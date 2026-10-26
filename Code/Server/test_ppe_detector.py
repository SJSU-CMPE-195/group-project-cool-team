"""Tests for detector pre/post-processing.

Needs numpy and cv2 but NOT onnxruntime -- these cover the hand-written
letterbox / decode / NMS maths, which is the part of the detector that can be
wrong without being obviously broken. A box drawn 20px off looks plausible on
screen; a round-trip test catches it exactly.

Run: python test_ppe_detector.py
"""

from __future__ import annotations

import sys

import numpy as np

from ppe_classes import CLASS_IDS, NUM_CLASSES
from ppe_detector import (
    decode_v8,
    letterbox,
    nms_numpy,
    preprocess,
    unletterbox_boxes,
)


# --------------------------------------------------------------------------
# letterbox
# --------------------------------------------------------------------------


def test_letterbox_480p_into_640_is_pad_only():
    """The reason the camera runs at 640x480.

    640x480 into 640x640 must be ratio 1.0 with 80px top/bottom padding and
    zero resampling -- free accuracy for a model trained at 640.
    """
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out, ratio, pad = letterbox(img, (640, 640))
    assert out.shape == (640, 640, 3)
    assert ratio == 1.0, f"expected no scaling, got ratio {ratio}"
    assert pad == (0.0, 80.0), f"expected (0, 80) padding, got {pad}"


def test_letterbox_preserves_aspect_ratio():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out, ratio, pad = letterbox(img, (448, 448))
    assert out.shape == (448, 448, 3)
    assert abs(ratio - 448 / 640) < 1e-6
    # 480 * 0.7 = 336, so 56px of padding top and bottom
    assert abs(pad[1] - 56.0) < 1.0
    assert pad[0] == 0.0


def test_letterbox_pads_with_the_training_grey():
    img = np.full((100, 200, 3), 255, dtype=np.uint8)
    out, _, _ = letterbox(img, (640, 640))
    assert out[0, 0, 0] == 114, "pad value must match ultralytics' 114"


def test_letterbox_square_input_needs_no_padding():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    out, ratio, pad = letterbox(img, (640, 640))
    assert out.shape == (640, 640, 3)
    assert pad == (0.0, 0.0)
    assert abs(ratio - 640 / 300) < 1e-6


# --------------------------------------------------------------------------
# the round trip -- the critical one
# --------------------------------------------------------------------------


def _forward(box, ratio, pad):
    """Map an original-frame box into letterboxed space (what the model sees)."""
    x1, y1, x2, y2 = box
    return [
        x1 * ratio + pad[0],
        y1 * ratio + pad[1],
        x2 * ratio + pad[0],
        y2 * ratio + pad[1],
    ]


def test_unletterbox_round_trip_480p():
    """Forward then back must land on the original coordinates."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, ratio, pad = letterbox(img, (640, 640))
    originals = [
        (10.0, 20.0, 110.0, 220.0),
        (250.0, 100.0, 390.0, 400.0),
        (0.0, 0.0, 640.0, 480.0),
        (300.5, 200.25, 320.75, 240.5),
    ]
    boxes = np.array([_forward(b, ratio, pad) for b in originals], dtype=np.float32)
    back = unletterbox_boxes(boxes, ratio, pad, (640, 480))
    for want, got in zip(originals, back):
        for w, g in zip(want, got):
            assert abs(w - g) < 0.01, f"round trip drifted: {want} -> {tuple(got)}"


def test_unletterbox_round_trip_with_downscale():
    """Non-unit ratio AND padding together -- where order-of-operations bugs show."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, ratio, pad = letterbox(img, (448, 448))
    assert ratio != 1.0 and pad[1] > 0
    originals = [(33.0, 44.0, 333.0, 444.0), (1.0, 2.0, 639.0, 479.0)]
    boxes = np.array([_forward(b, ratio, pad) for b in originals], dtype=np.float32)
    back = unletterbox_boxes(boxes, ratio, pad, (640, 480))
    for want, got in zip(originals, back):
        for w, g in zip(want, got):
            assert abs(w - g) < 0.1, f"round trip drifted: {want} -> {tuple(got)}"


def test_unletterbox_order_matters():
    """Pin the bug this guards against.

    Un-scaling before un-padding (the natural-looking mistake) gives a
    measurably different answer. If this ever stops differing, the
    implementation has regressed into the wrong order.
    """
    ratio, pad = 0.7, (0.0, 56.0)
    boxes = np.array([[70.0, 126.0, 140.0, 196.0]], dtype=np.float32)
    correct = unletterbox_boxes(boxes, ratio, pad, (640, 480))
    wrong = boxes.copy() / ratio
    wrong[:, [1, 3]] -= pad[1]
    assert abs(correct[0][1] - wrong[0][1]) > 1.0


def test_unletterbox_clips_to_frame():
    boxes = np.array([[-50.0, -50.0, 900.0, 900.0]], dtype=np.float32)
    out = unletterbox_boxes(boxes, 1.0, (0.0, 0.0), (640, 480))
    assert out[0][0] >= 0 and out[0][1] >= 0
    assert out[0][2] <= 640 and out[0][3] <= 480


# --------------------------------------------------------------------------
# preprocess
# --------------------------------------------------------------------------


def test_preprocess_shape_and_range():
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    blob, ratio, pad = preprocess(img, (640, 640))
    assert blob.shape == (1, 3, 640, 640), f"want NCHW, got {blob.shape}"
    assert blob.dtype == np.float32
    assert 0.0 <= blob.min() and blob.max() <= 1.0


def test_preprocess_converts_bgr_to_rgb():
    """cv2 gives BGR, the model wants RGB, so the channels must swap.

    Distinct per-channel values make the mapping unambiguous: a BGR frame of
    (10, 20, 30) must come out as RGB planes (30, 20, 10). Get this backwards
    and detections quietly degrade on every single frame, with nothing but a
    blue-tinted annotated image as a hint.
    """
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[:, :, 0] = 10   # B
    img[:, :, 1] = 20   # G
    img[:, :, 2] = 30   # R
    blob, _, _ = preprocess(img, (640, 640))
    assert abs(blob[0, 0].mean() - 30 / 255) < 1e-3, "plane 0 must be R"
    assert abs(blob[0, 1].mean() - 20 / 255) < 1e-3, "plane 1 must be G"
    assert abs(blob[0, 2].mean() - 10 / 255) < 1e-3, "plane 2 must be B"


def test_preprocess_is_contiguous():
    """ORT silently copies non-contiguous input; keep it free."""
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    blob, _, _ = preprocess(img, (640, 640))
    assert blob.flags["C_CONTIGUOUS"]


# --------------------------------------------------------------------------
# NMS
# --------------------------------------------------------------------------


def test_nms_suppresses_overlapping_and_keeps_best():
    boxes = np.array(
        [[0, 0, 100, 100], [5, 5, 105, 105], [500, 500, 600, 600]], dtype=np.float32
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms_numpy(boxes, scores, 0.45)
    assert keep == [0, 2]


def test_nms_keeps_distinct_boxes():
    boxes = np.array(
        [[0, 0, 50, 50], [100, 100, 150, 150], [300, 300, 350, 350]], dtype=np.float32
    )
    scores = np.array([0.5, 0.9, 0.7], dtype=np.float32)
    assert sorted(nms_numpy(boxes, scores, 0.45)) == [0, 1, 2]


def test_nms_empty_and_single():
    assert nms_numpy(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), 0.45) == []
    one = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert nms_numpy(one, np.array([0.5], np.float32), 0.45) == [0]


# --------------------------------------------------------------------------
# decode
# --------------------------------------------------------------------------


def _raw(entries: list[tuple[str, float, tuple[float, float, float, float]]]):
    """Build a synthetic (1, 4+nc, N) head output from cxcywh boxes."""
    arr = np.zeros((1, 4 + NUM_CLASSES, len(entries)), dtype=np.float32)
    for i, (name, conf, (cx, cy, w, h)) in enumerate(entries):
        arr[0, 0, i], arr[0, 1, i], arr[0, 2, i], arr[0, 3, i] = cx, cy, w, h
        arr[0, 4 + CLASS_IDS[name], i] = conf
    return arr


def test_decode_basic():
    raw = _raw([("person", 0.9, (320.0, 240.0, 100.0, 300.0))])
    dets = decode_v8(raw, 1.0, (0.0, 0.0), (640, 480), 0.25, 0.45)
    assert len(dets) == 1
    d = dets[0]
    assert d.name == "person"
    assert abs(d.conf - 0.9) < 1e-5
    # cxcywh 320,240,100,300 -> xyxy 270,90,370,390
    for want, got in zip((270.0, 90.0, 370.0, 390.0), d.xyxy):
        assert abs(want - got) < 0.01


def test_decode_accepts_transposed_layout():
    """Some exports emit (N, 4+nc) instead of (4+nc, N)."""
    raw = _raw([("helmet", 0.8, (100.0, 100.0, 40.0, 40.0))])
    a = decode_v8(raw, 1.0, (0.0, 0.0), (640, 480), 0.25, 0.45)
    b = decode_v8(raw[0].T, 1.0, (0.0, 0.0), (640, 480), 0.25, 0.45)
    assert len(a) == len(b) == 1
    assert a[0].xyxy == b[0].xyxy


def test_decode_applies_conf_floor():
    raw = _raw(
        [
            ("person", 0.90, (320.0, 240.0, 100.0, 300.0)),
            ("person", 0.10, (100.0, 100.0, 50.0, 50.0)),
        ]
    )
    dets = decode_v8(raw, 1.0, (0.0, 0.0), (640, 480), 0.25, 0.45)
    assert len(dets) == 1


def test_decode_applies_per_class_bar_after_nms():
    raw = _raw([("head", 0.45, (100.0, 100.0, 40.0, 40.0))])
    assert decode_v8(raw, 1.0, (0.0, 0.0), (640, 480), 0.25, 0.45) != []
    assert decode_v8(
        raw, 1.0, (0.0, 0.0), (640, 480), 0.25, 0.45, class_conf={"head": 0.50}
    ) == []


def test_decode_nms_is_class_wise_not_agnostic():
    """The single most consequential detail in the decoder.

    A helmet sits on a head, so the two boxes overlap heavily. Agnostic NMS
    would suppress one of them, and the rule engine would lose the pair it
    needs to distinguish a helmeted person from a bare-headed one.
    """
    overlapping = (320.0, 120.0, 50.0, 50.0)
    raw = _raw([("head", 0.90, overlapping), ("helmet", 0.85, overlapping)])
    dets = decode_v8(raw, 1.0, (0.0, 0.0), (640, 480), 0.25, 0.45)
    names = {d.name for d in dets}
    assert names == {"head", "helmet"}, (
        f"class-wise NMS must keep both, got {names}"
    )


def test_decode_still_suppresses_duplicates_within_a_class():
    raw = _raw(
        [
            ("person", 0.90, (320.0, 240.0, 100.0, 300.0)),
            ("person", 0.85, (322.0, 242.0, 100.0, 300.0)),
        ]
    )
    dets = decode_v8(raw, 1.0, (0.0, 0.0), (640, 480), 0.25, 0.45)
    assert len(dets) == 1
    assert abs(dets[0].conf - 0.90) < 1e-5


def test_decode_respects_allowed_classes():
    """The scan pass only cares about people."""
    raw = _raw(
        [
            ("person", 0.90, (320.0, 240.0, 100.0, 300.0)),
            ("helmet", 0.88, (320.0, 120.0, 40.0, 40.0)),
        ]
    )
    dets = decode_v8(
        raw, 1.0, (0.0, 0.0), (640, 480), 0.25, 0.45,
        allowed={CLASS_IDS["person"]},
    )
    assert [d.name for d in dets] == ["person"]


def test_decode_maps_through_letterbox():
    """Decode must undo padding, not just hand back letterboxed coordinates."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, ratio, pad = letterbox(img, (640, 640))
    # A person at 270,90..370,390 in the frame, expressed in letterboxed space
    cx, cy = 320.0 * ratio + pad[0], 240.0 * ratio + pad[1]
    raw = _raw([("person", 0.9, (cx, cy, 100.0 * ratio, 300.0 * ratio))])
    dets = decode_v8(raw, ratio, pad, (640, 480), 0.25, 0.45)
    for want, got in zip((270.0, 90.0, 370.0, 390.0), dets[0].xyxy):
        assert abs(want - got) < 0.5


def test_decode_drops_degenerate_boxes():
    raw = _raw([("person", 0.9, (320.0, 240.0, 0.4, 0.4))])
    assert decode_v8(raw, 1.0, (0.0, 0.0), (640, 480), 0.25, 0.45) == []


def test_decode_empty_output():
    arr = np.zeros((1, 4 + NUM_CLASSES, 0), dtype=np.float32)
    assert decode_v8(arr, 1.0, (0.0, 0.0), (640, 480), 0.25, 0.45) == []


def test_decode_rejects_wrong_channel_count():
    bad = np.zeros((1, 85, 100), dtype=np.float32)  # a COCO-shaped head
    try:
        decode_v8(bad, 1.0, (0.0, 0.0), (640, 480), 0.25, 0.45)
    except ValueError as exc:
        assert "4+17" in str(exc) or "expected" in str(exc)
    else:
        raise AssertionError("must reject a non-17-class model output")


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------


def _run() -> int:
    tests = [
        (n, f) for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except BaseException as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    print("PPE detector pre/post-processing tests\n")
    sys.exit(_run())
