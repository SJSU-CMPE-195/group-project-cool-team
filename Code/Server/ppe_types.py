"""Detection record and box geometry. Zero dependencies, deliberately.

Both ppe_detector.py (which needs numpy/cv2/onnxruntime) and ppe_rules.py
import this. Keeping it dependency-free means the rule engine -- the part most
worth testing -- can be unit tested with a bare Python interpreter.

Box convention throughout the project: (x1, y1, x2, y2) in FULL-FRAME pixel
coordinates, x right, y down, x1<=x2 and y1<=y2. The detector is responsible
for undoing letterbox padding before it hands boxes over; nothing downstream
should ever see letterboxed coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """One detected object in full-frame pixel coordinates."""

    cls_id: int
    name: str
    conf: float
    xyxy: Box

    @property
    def x1(self) -> float:
        return self.xyxy[0]

    @property
    def y1(self) -> float:
        return self.xyxy[1]

    @property
    def x2(self) -> float:
        return self.xyxy[2]

    @property
    def y2(self) -> float:
        return self.xyxy[3]

    @property
    def width(self) -> float:
        return max(0.0, self.xyxy[2] - self.xyxy[0])

    @property
    def height(self) -> float:
        return max(0.0, self.xyxy[3] - self.xyxy[1])

    def area(self) -> float:
        return self.width * self.height

    def center(self) -> tuple[float, float]:
        return (
            (self.xyxy[0] + self.xyxy[2]) / 2.0,
            (self.xyxy[1] + self.xyxy[3]) / 2.0,
        )

    def to_json(self) -> dict:
        return {
            "cls": self.name,
            "conf": round(float(self.conf), 4),
            "bbox": [round(float(v), 1) for v in self.xyxy],
        }


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def intersection_area(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def box_area(a: Box) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def iou(a: Box, b: Box) -> float:
    inter = intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def overlap_frac(inner: Box, outer: Box) -> float:
    """Fraction of `inner` that falls inside `outer`.

    This, NOT IoU, is the right measure for attaching a PPE item to a person.
    A helmet is tiny relative to a person, so a correctly-attached helmet has
    an IoU near 0.02 while its containment is ~1.0. Using IoU here would
    attach nothing to anyone.
    """
    a = box_area(inner)
    if a <= 0.0:
        return 0.0
    return intersection_area(inner, outer) / a


def pad_box(b: Box, frac: float, clip: tuple[int, int] | None = None) -> Box:
    """Grow a box by `frac` of its own size on every side.

    Person boxes get padded slightly before association because a helmet
    routinely pokes above the top edge of the person box the detector drew.
    """
    w, h = max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1])
    dx, dy = w * frac, h * frac
    x1, y1, x2, y2 = b[0] - dx, b[1] - dy, b[2] + dx, b[3] + dy
    if clip is not None:
        fw, fh = clip
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(fw), x2), min(float(fh), y2)
    return (x1, y1, x2, y2)


def vertical_band(inner: Box, outer: Box) -> tuple[float, float]:
    """Where `inner` sits vertically within `outer`, as (top_frac, bottom_frac).

    Used to reject anatomically impossible associations -- a "helmet" detected
    at a person's knees is a misdetection, not PPE. Returns fractions that may
    fall outside 0..1 when `inner` extends past `outer`.
    """
    h = max(1e-6, outer[3] - outer[1])
    return ((inner[1] - outer[1]) / h, (inner[3] - outer[1]) / h)


def touches_edge(b: Box, frame_wh: tuple[int, int], margin: int) -> bool:
    """True if the box comes within `margin` px of any frame edge.

    A person clipped by the frame edge cannot be judged on absence evidence:
    their torso may simply be outside the picture.
    """
    w, h = frame_wh
    return (
        b[0] <= margin
        or b[1] <= margin
        or b[2] >= w - margin
        or b[3] >= h - margin
    )
