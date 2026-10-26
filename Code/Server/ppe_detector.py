"""YOLOv8 PPE detector.

Runs the SH17 model through ONNX Runtime with hand-written pre/post-processing.
That is a deliberate trade: ~200 lines of letterbox/decode/NMS in exchange for
never installing torch or ultralytics on the Pi.

Why that matters on a Raspberry Pi 5 running Bookworm:

  * torch + torchvision for aarch64 is a 1.5-2.5GB install; onnxruntime is a
    single ~15MB wheel, imports in under a second instead of 4-8, and holds
    ~150MB RSS instead of ~1GB.
  * More importantly, `pip install ultralytics` drags in `opencv-python` and
    `numpy>=2`, and Bookworm's picamera2 stack is compiled against numpy 1.x.
    Letting pip win that argument breaks camera capture with ABI errors that
    look exactly like camera hardware faults. See docs/PI_SETUP.md.

The module imports with only numpy and cv2 present; onnxruntime is imported
lazily inside the session builder, so ppe_rules.py and the test suite can
import this file on a machine that has no inference runtime at all.

Four details that are easy to get wrong and silently ruin everything:

  1. Output is (1, 4+nc, N) = (1, 21, 8400). It must be TRANSPOSED to (N, 21).
  2. YOLOv8 has NO objectness channel. Score = max over the 17 class scores.
     Multiplying by a non-existent objectness term is a classic port bug.
  3. Boxes come out as cx,cy,w,h in LETTERBOXED pixel space. Un-pad before
     un-scaling, in that order.
  4. NMS must be CLASS-WISE. Agnostic NMS would suppress a `helmet` box
     overlapping a `head` box -- destroying precisely the evidence pair the
     rule engine depends on to tell a helmeted person from a bare-headed one.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ppe_classes import CLASS_NAMES, NUM_CLASSES
from ppe_types import Detection

PAD_VALUE = 114  # ultralytics' letterbox grey; must match training


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------


def letterbox(
    img: np.ndarray, new_shape: tuple[int, int], color: int = PAD_VALUE
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize preserving aspect ratio, pad to `new_shape` (w, h).

    Returns (padded_image, ratio, (pad_x, pad_y)) where pad_* is the padding
    added to the LEFT and TOP. Those three values are exactly what's needed to
    map boxes back to original coordinates.

    A 640x480 frame into a 640x640 box gives ratio 1.0 and pad (0, 80): pure
    padding, no resampling, which is why the camera runs at 640x480.
    """
    h, w = img.shape[:2]
    new_w, new_h = new_shape
    ratio = min(new_w / w, new_h / h)
    unpad_w, unpad_h = round(w * ratio), round(h * ratio)

    if (w, h) != (unpad_w, unpad_h):
        interp = cv2.INTER_LINEAR if ratio > 1 else cv2.INTER_AREA
        img = cv2.resize(img, (unpad_w, unpad_h), interpolation=interp)

    pad_x = (new_w - unpad_w) / 2.0
    pad_y = (new_h - unpad_h) / 2.0
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)
    out = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color,) * 3
    )
    return out, ratio, (pad_x, pad_y)


def unletterbox_boxes(
    boxes: np.ndarray,
    ratio: float,
    pad: tuple[float, float],
    orig_wh: tuple[int, int],
) -> np.ndarray:
    """Map xyxy boxes from letterboxed space back to original frame pixels.

    Un-pad first, then un-scale. Doing it the other way round is subtly wrong
    and the error grows with aspect-ratio mismatch, which makes it a nightmare
    to spot by eye on near-square images.
    """
    pad_x, pad_y = pad
    out = boxes.astype(np.float32, copy=True)
    out[:, [0, 2]] -= pad_x
    out[:, [1, 3]] -= pad_y
    out /= ratio
    w, h = orig_wh
    # Assign through __setitem__ rather than np.clip(..., out=...): fancy
    # indexing returns a COPY, so clipping into out[:, [0, 2]] would write to
    # a temporary and silently do nothing.
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0.0, float(w))
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0.0, float(h))
    return out


def preprocess(img_bgr: np.ndarray, size: tuple[int, int]):
    """BGR frame -> (NCHW float32 RGB /255, ratio, pad)."""
    padded, ratio, pad = letterbox(img_bgr, size)
    rgb = padded[:, :, ::-1]  # BGR -> RGB; getting this wrong tints everything
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
    chw /= 255.0
    return chw[None, ...], ratio, pad


# --------------------------------------------------------------------------
# postprocessing
# --------------------------------------------------------------------------


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Standard greedy NMS. Returns kept indices, highest score first."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_thr]
    return keep


def decode_v8(
    raw: np.ndarray,
    ratio: float,
    pad: tuple[float, float],
    orig_wh: tuple[int, int],
    conf_floor: float,
    iou_thr: float,
    class_conf: dict[str, float] | None = None,
    allowed: set[int] | None = None,
) -> list[Detection]:
    """Decode a YOLOv8 detection head into Detections.

    `raw` is the single model output, shape (1, 4+nc, N) or (4+nc, N).
    `conf_floor` admits candidates; `class_conf` then applies a per-class bar
    after NMS, so a tight per-class threshold never costs us NMS quality.
    """
    arr = raw[0] if raw.ndim == 3 else raw
    if arr.shape[0] == 4 + NUM_CLASSES:
        arr = arr.T  # (N, 4+nc) -- channels-first is the usual export layout
    elif arr.shape[1] != 4 + NUM_CLASSES:
        raise ValueError(
            f"unexpected model output {raw.shape}; expected a 4+{NUM_CLASSES} axis"
        )

    cls_scores = arr[:, 4:]
    cls_ids = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]

    mask = scores >= conf_floor
    if allowed is not None:
        mask &= np.isin(cls_ids, list(allowed))
    if not mask.any():
        return []

    boxes_cxcywh = arr[mask, :4]
    scores = scores[mask]
    cls_ids = cls_ids[mask]

    # cxcywh -> xyxy, still in letterboxed space
    xy = boxes_cxcywh[:, :2]
    wh = boxes_cxcywh[:, 2:4]
    boxes = np.concatenate([xy - wh / 2.0, xy + wh / 2.0], axis=1)
    boxes = unletterbox_boxes(boxes, ratio, pad, orig_wh)

    # Class-wise NMS. See module docstring, point 4: this is load-bearing.
    out: list[Detection] = []
    for cid in np.unique(cls_ids):
        sel = np.where(cls_ids == cid)[0]
        keep = nms_numpy(boxes[sel], scores[sel], iou_thr)
        name = CLASS_NAMES[int(cid)]
        bar = (class_conf or {}).get(name, 0.0)
        for k in keep:
            idx = sel[k]
            score = float(scores[idx])
            if score < bar:
                continue
            x1, y1, x2, y2 = (float(v) for v in boxes[idx])
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue
            out.append(
                Detection(cls_id=int(cid), name=name, conf=score, xyxy=(x1, y1, x2, y2))
            )

    out.sort(key=lambda d: d.conf, reverse=True)
    return out


# --------------------------------------------------------------------------
# detector
# --------------------------------------------------------------------------


class PpeDetector:
    """ONNX Runtime PPE detector.

    Thread-safety: an ORT session is safe to call from multiple threads, but
    this project drives it from the single inference thread. Don't share one
    instance across processes -- create it after any fork (see ppe_main).
    """

    def __init__(
        self,
        onnx_path: str | Path,
        conf: float = 0.30,
        iou: float = 0.45,
        class_conf: dict[str, float] | None = None,
        intra_op_threads: int = 3,
        providers: list[str] | None = None,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.is_file():
            raise FileNotFoundError(
                f"model not found: {self.onnx_path}\n"
                f"Export it first:  python Tools/export_onnx.py --imgsz 640"
            )
        self.conf = conf
        self.iou = iou
        self.class_conf = dict(class_conf or {})

        # Lazy import: keeps this module importable where onnxruntime isn't
        # installed, which is what lets the rule-engine tests run anywhere.
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "onnxruntime is required for inference.\n"
                "  Pi : ~/ppe-venv/bin/pip install onnxruntime\n"
                "  PC : pip install onnxruntime"
            ) from exc

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(intra_op_threads))
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=opts,
            providers=providers or ["CPUExecutionProvider"],
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        # Fixed-shape export: (1, 3, H, W). Fall back to 640 for dynamic axes.
        h = shape[2] if isinstance(shape[2], int) else 640
        w = shape[3] if isinstance(shape[3], int) else 640
        self.input_size: tuple[int, int] = (int(w), int(h))
        self.output_names = [o.name for o in self.session.get_outputs()]

        self._verify_class_vocabulary()

    def _verify_class_vocabulary(self) -> None:
        """Check the model's embedded class names against ours.

        A model swap that silently shifts label indices would turn every
        verdict into nonsense while looking perfectly healthy, so this is
        worth the few lines. Only warns when metadata is absent, since ONNX
        exports don't always carry it.
        """
        meta = self.session.get_modelmeta().custom_metadata_map or {}
        raw = meta.get("names")
        if not raw:
            print(
                f"[detector] note: {self.onnx_path.name} carries no class metadata; "
                f"assuming the {NUM_CLASSES}-class SH17 vocabulary"
            )
            return
        try:
            parsed: Any = eval(raw, {"__builtins__": {}}, {})  # noqa: S307
        except Exception:  # pragma: no cover
            return
        if isinstance(parsed, dict):
            got = tuple(parsed[k] for k in sorted(parsed, key=int))
            if got != CLASS_NAMES:
                raise ValueError(
                    "model class names do not match ppe_classes.CLASS_NAMES.\n"
                    f"  model: {got}\n  ours : {CLASS_NAMES}\n"
                    "Refusing to run: every PPE verdict would be mislabelled."
                )

    @classmethod
    def from_config(cls, cfg, which: str = "confirm") -> PpeDetector:
        """Build from a PPEConfig. `which` is "scan" or "confirm"."""
        model = cfg.model
        rel = model.scan_onnx if which == "scan" else model.confirm_onnx
        return cls(
            onnx_path=cfg.resolve_path(rel),
            conf=model.conf_default,
            iou=model.iou_nms,
            class_conf=model.class_conf,
            intra_op_threads=model.intra_op_threads,
        )

    # ------------------------------------------------------------------ infer

    def detect(
        self,
        frame_bgr: np.ndarray,
        allowed: set[int] | None = None,
        apply_class_conf: bool = True,
    ) -> list[Detection]:
        """Detect in a full frame. Boxes come back in frame pixel coordinates."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        h, w = frame_bgr.shape[:2]
        blob, ratio, pad = preprocess(frame_bgr, self.input_size)
        raw = self.session.run(self.output_names, {self.input_name: blob})[0]
        return decode_v8(
            raw,
            ratio,
            pad,
            (w, h),
            conf_floor=self.conf,
            iou_thr=self.iou,
            class_conf=self.class_conf if apply_class_conf else None,
            allowed=allowed,
        )

    def detect_crop(
        self,
        frame_bgr: np.ndarray,
        xyxy: tuple[float, float, float, float],
        pad_frac: float = 0.15,
    ) -> list[Detection]:
        """Run a second pass on a padded crop, boxes mapped back to the frame.

        The model was trained at 640. A person 180px tall in the frame has a
        helmet maybe 25px across -- near this nano model's floor. Letterboxing
        that person's crop up to 640 puts small PPE back near its training
        scale, which is the cheapest recall win available. Only worth doing
        while stationary, where there's time to spare.
        """
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = xyxy
        dx, dy = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
        cx1 = max(0, int(x1 - dx))
        cy1 = max(0, int(y1 - dy))
        cx2 = min(w, int(x2 + dx))
        cy2 = min(h, int(y2 + dy))
        if cx2 - cx1 < 16 or cy2 - cy1 < 16:
            return []

        crop = frame_bgr[cy1:cy2, cx1:cx2]
        dets = self.detect(crop)
        return [
            Detection(
                cls_id=d.cls_id,
                name=d.name,
                conf=d.conf,
                xyxy=(d.x1 + cx1, d.y1 + cy1, d.x2 + cx1, d.y2 + cy1),
            )
            for d in dets
        ]

    def warmup(self, n: int = 2) -> None:
        """Run the graph on dummy input so the first real frame isn't slow."""
        w, h = self.input_size
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(max(1, n)):
            self.detect(dummy)

    def benchmark(self, frame_bgr: np.ndarray | None = None, runs: int = 30) -> dict:
        """Measure latency. The number that sets the whole patrol timing budget."""
        if frame_bgr is None:
            w, h = self.input_size
            frame_bgr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        self.warmup(3)
        times: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            self.detect(frame_bgr)
            times.append((time.perf_counter() - t0) * 1000.0)
        times.sort()
        return {
            "model": self.onnx_path.name,
            "imgsz": self.input_size,
            "runs": runs,
            "p50_ms": round(times[len(times) // 2], 1),
            "p90_ms": round(times[int(len(times) * 0.9)], 1),
            "min_ms": round(times[0], 1),
            "max_ms": round(times[-1], 1),
            "mean_ms": round(sum(times) / len(times), 1),
            "fps_p50": round(1000.0 / max(times[len(times) // 2], 1e-6), 1),
        }

    def close(self) -> None:
        self.session = None


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

_COLOURS = {
    "person": (255, 190, 80),
    "helmet": (80, 220, 110),
    "head": (70, 90, 255),
    "safety-vest": (80, 220, 110),
    "gloves": (80, 220, 110),
    "hands": (70, 90, 255),
    "shoes": (80, 220, 110),
    "foot": (70, 90, 255),
}
_DEFAULT_COLOUR = (190, 190, 190)


def annotate(
    frame_bgr: np.ndarray,
    dets: list[Detection],
    observations: list | None = None,
    show_labels: bool = True,
) -> np.ndarray:
    """Draw boxes and, if given, each person's PPE verdict. Returns a copy."""
    out = frame_bgr.copy()

    for d in dets:
        if d.name == "person":
            continue
        colour = _COLOURS.get(d.name, _DEFAULT_COLOUR)
        p1 = (int(d.x1), int(d.y1))
        p2 = (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, colour, 1)
        if show_labels:
            cv2.putText(
                out, f"{d.name} {d.conf:.2f}", (p1[0], max(10, p1[1] - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA,
            )

    if observations:
        for obs in observations:
            missing = obs.missing_items()
            colour = (60, 60, 240) if missing else (90, 210, 120)
            p1 = (int(obs.person.x1), int(obs.person.y1))
            p2 = (int(obs.person.x2), int(obs.person.y2))
            cv2.rectangle(out, p1, p2, colour, 2)
            label = (
                "NO " + ", ".join(m.upper() for m in missing)
                if missing
                else "PPE OK"
            )
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            ty = max(th + 4, p1[1])
            cv2.rectangle(out, (p1[0], ty - th - 4), (p1[0] + tw + 6, ty + 2), colour, -1)
            cv2.putText(
                out, label, (p1[0] + 3, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA,
            )
    else:
        for d in dets:
            if d.name != "person":
                continue
            cv2.rectangle(
                out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)),
                _COLOURS["person"], 2,
            )
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _main() -> int:
    import argparse

    from ppe_config import load_config

    ap = argparse.ArgumentParser(description="PPE detector: test, benchmark, inspect.")
    ap.add_argument("--image", help="detect in an image file")
    ap.add_argument("--onnx", help="model path (default: config's confirm model)")
    ap.add_argument("--which", default="confirm", choices=("scan", "confirm"))
    ap.add_argument("--config", help="config path")
    ap.add_argument("--bench", action="store_true", help="measure latency")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--threads", type=int, help="override intra_op_num_threads")
    ap.add_argument("--rules", action="store_true", help="also apply the rule engine")
    ap.add_argument("--out", default="detect_annotated.jpg")
    args = ap.parse_args()

    cfg = load_config(args.config)
    path = args.onnx or cfg.resolve_path(
        cfg.model.scan_onnx if args.which == "scan" else cfg.model.confirm_onnx
    )
    det = PpeDetector(
        onnx_path=path,
        conf=cfg.model.conf_default,
        iou=cfg.model.iou_nms,
        class_conf=cfg.model.class_conf,
        intra_op_threads=args.threads or cfg.model.intra_op_threads,
    )
    print(f"model {det.onnx_path.name}  input {det.input_size}")

    if args.bench:
        frame = cv2.imread(args.image) if args.image else None
        stats = det.benchmark(frame, runs=args.runs)
        width = max(len(k) for k in stats)
        for k, v in stats.items():
            print(f"  {k:<{width}} : {v}")
        try:  # Pi only
            import subprocess

            for cmd in (["vcgencmd", "measure_temp"], ["vcgencmd", "get_throttled"]):
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if res.returncode == 0:
                    print(f"  {cmd[1]:<{width}} : {res.stdout.strip()}")
        except Exception:
            pass
        return 0

    if not args.image:
        ap.error("pass --image, or --bench")

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"cannot read image: {args.image}")
        return 1

    t0 = time.perf_counter()
    dets = det.detect(frame)
    ms = (time.perf_counter() - t0) * 1000.0
    h, w = frame.shape[:2]
    print(f"{args.image}  {w}x{h}  {len(dets)} detections in {ms:.1f}ms")
    for d in dets:
        print(f"  {d.name:14s} {d.conf:.3f}  {[round(v) for v in d.xyxy]}")

    observations = None
    if args.rules:
        from ppe_rules import RuleEngine, summarize

        observations = RuleEngine(cfg.rules, cfg.model).evaluate_frame(dets, (w, h))
        print(summarize(observations))

    cv2.imwrite(args.out, annotate(frame, dets, observations))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
