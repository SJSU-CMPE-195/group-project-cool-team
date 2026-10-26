"""Export yolo8n.pt to ONNX, and verify our decoder against ultralytics.

Run this ON THE PC, once. The Pi never needs torch or ultralytics -- it runs
the exported .onnx through onnxruntime alone.

    # one-off throwaway env (torch is ~2GB and we only need it to export)
    py -3.12 -m venv %TEMP%\\yolo-export
    %TEMP%\\yolo-export\\Scripts\\pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cpu
    %TEMP%\\yolo-export\\Scripts\\pip install "ultralytics==8.2.38" onnx onnxsim onnxruntime

    %TEMP%\\yolo-export\\Scripts\\python Tools/export_onnx.py --imgsz 640
    %TEMP%\\yolo-export\\Scripts\\python Tools/export_onnx.py --imgsz 448
    %TEMP%\\yolo-export\\Scripts\\python Tools/export_onnx.py --imgsz 640 --verify path/to/photo.jpg

Two pins that matter:

  ultralytics==8.2.38  the version that produced this checkpoint. Later
                       versions have renamed internal attributes and may fail
                       to unpickle it.
  torch==2.4.x         torch 2.6 flipped `torch.load(weights_only=True)` to
                       the default, which breaks loading ultralytics
                       checkpoints of this era.

--verify is the important part, and the single most valuable test in the whole
project: it runs ultralytics' own predict() and our onnxruntime path on the
same image and compares the boxes. ppe_detector.py reimplements letterboxing,
YOLOv8 head decoding and NMS by hand, and a subtle error there (wrong channel
order, un-scaling before un-padding, agnostic instead of class-wise NMS) does
not crash -- it just quietly degrades every detection forever. Once the
numbers agree, torch is never needed again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO / "yolo8n.pt"
MODEL_DIR = REPO / "Code" / "Server" / "models"

EXPECTED_CLASSES = (
    "person", "ear", "ear-mufs", "face", "face-guard", "face-mask", "foot",
    "tool", "glasses", "gloves", "helmet", "hands", "head", "medical-suit",
    "shoes", "safety-suit", "safety-vest",
)


def sha256_12(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def export(weights: Path, imgsz: int, opset: int, simplify: bool) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "ultralytics is not installed in this interpreter.\n"
            "See the module docstring for the throwaway export venv."
        )

    print(f"loading {weights}")
    model = YOLO(str(weights))

    names = tuple(model.names[k] for k in sorted(model.names, key=int))
    if names != EXPECTED_CLASSES:
        sys.exit(
            f"unexpected class vocabulary in {weights.name}:\n"
            f"  got    {names}\n  expect {EXPECTED_CLASSES}\n"
            "ppe_classes.py would mislabel every detection. Aborting."
        )
    print(f"{len(names)} classes verified against ppe_classes.CLASS_NAMES")

    # Fixed shape (dynamic=False) lets ORT fully fold the graph; measurably
    # faster than a dynamic-axis export, and we only ever feed one size.
    print(f"exporting ONNX imgsz={imgsz} opset={opset} simplify={simplify}")
    produced = Path(
        model.export(
            format="onnx", imgsz=imgsz, opset=opset, simplify=simplify,
            dynamic=False, half=False,
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = MODEL_DIR / f"ppe_yolo8n_{imgsz}.onnx"
    if produced.resolve() != target.resolve():
        shutil.move(str(produced), str(target))
    print(f"wrote {target.relative_to(REPO)}  ({target.stat().st_size / 1e6:.1f} MB)")

    meta = {
        "names": {str(i): n for i, n in enumerate(names)},
        "imgsz": imgsz,
        "opset": opset,
        "simplified": simplify,
        "source_pt": weights.name,
        "pt_sha256_12": sha256_12(weights),
        "onnx_sha256_12": sha256_12(target),
        "dataset": "SH17 (safe_human.yaml)",
        "license": "AGPL-3.0 (inherited from the ultralytics weights)",
    }
    meta_path = target.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {meta_path.relative_to(REPO)}")
    return target


def verify(onnx_path: Path, weights: Path, image: Path, conf: float, iou: float) -> int:
    """Compare ultralytics predict() against our onnxruntime decoder.

    Passes when both find the same number of detections, and every box pairs
    up with IoU > 0.95, the same class, and confidence within 0.02.
    """
    import cv2

    sys.path.insert(0, str(REPO / "Code" / "Server"))
    from ppe_detector import PpeDetector  # noqa: E402
    from ppe_types import iou as box_iou  # noqa: E402

    frame = cv2.imread(str(image))
    if frame is None:
        sys.exit(f"cannot read image: {image}")
    print(f"\nverifying on {image.name} ({frame.shape[1]}x{frame.shape[0]})")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    imgsz = PpeDetector(onnx_path).input_size[0]
    res = model.predict(
        source=str(image), imgsz=imgsz, conf=conf, iou=iou, verbose=False
    )[0]
    ref = []
    for b in res.boxes:
        cid = int(b.cls.item())
        ref.append((model.names[cid], float(b.conf.item()), tuple(b.xyxy[0].tolist())))
    ref.sort(key=lambda t: -t[1])

    # Match ultralytics' thresholds exactly, and skip per-class gating, so we
    # are comparing decoders rather than comparing policy.
    det = PpeDetector(onnx_path, conf=conf, iou=iou, class_conf=None)
    ours = [
        (d.name, d.conf, d.xyxy)
        for d in det.detect(frame, apply_class_conf=False)
    ]
    ours.sort(key=lambda t: -t[1])

    print(f"\n  ultralytics : {len(ref)} detections")
    print(f"  ours (onnx) : {len(ours)} detections\n")

    width = max([len(n) for n, _, _ in ref + ours] + [10])
    print(f"  {'class':<{width}}  {'ultra':>6}  {'ours':>6}  {'IoU':>6}  verdict")
    print(f"  {'-' * width}  {'-' * 6}  {'-' * 6}  {'-' * 6}  -------")

    unmatched = list(ours)
    failures = 0
    for name, conf_r, box_r in ref:
        best, best_iou = None, 0.0
        for cand in unmatched:
            if cand[0] != name:
                continue
            score = box_iou(box_r, cand[2])
            if score > best_iou:
                best, best_iou = cand, score
        if best is None or best_iou <= 0.95 or abs(best[1] - conf_r) > 0.02:
            failures += 1
            got = f"{best[1]:.3f}" if best else "  --  "
            print(f"  {name:<{width}}  {conf_r:6.3f}  {got:>6}  {best_iou:6.3f}  MISMATCH")
        else:
            unmatched.remove(best)
            print(f"  {name:<{width}}  {conf_r:6.3f}  {best[1]:6.3f}  {best_iou:6.3f}  ok")

    for name, conf_o, _ in unmatched:
        failures += 1
        print(f"  {name:<{width}}  {'  --  ':>6}  {conf_o:6.3f}  {0.0:6.3f}  EXTRA")

    print()
    if failures:
        print(
            f"FAIL: {failures} mismatch(es).\n"
            "Our decoder disagrees with ultralytics. Check, in order:\n"
            "  - BGR->RGB channel order in preprocess()\n"
            "  - un-pad BEFORE un-scale in unletterbox_boxes()\n"
            "  - class-wise (not agnostic) NMS in decode_v8()\n"
            "  - no objectness channel: score = max over class logits\n"
        )
        return 1

    print(f"PASS: all {len(ref)} detections agree. The ONNX path is trustworthy;")
    print("torch/ultralytics are no longer needed for this project.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export the PPE model to ONNX and verify our decoder."
    )
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=640, help="640 confirm, 448 scan")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-simplify", action="store_true")
    ap.add_argument(
        "--verify", type=Path, metavar="IMAGE",
        help="compare against ultralytics on this image (do this at least once)",
    )
    ap.add_argument("--verify-only", action="store_true", help="skip the export")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    args = ap.parse_args()

    if not args.weights.is_file():
        sys.exit(f"weights not found: {args.weights}")

    if args.verify_only:
        onnx_path = MODEL_DIR / f"ppe_yolo8n_{args.imgsz}.onnx"
        if not onnx_path.is_file():
            sys.exit(f"no export at {onnx_path}; run without --verify-only first")
    else:
        onnx_path = export(
            args.weights, args.imgsz, args.opset, not args.no_simplify
        )

    if args.verify:
        return verify(onnx_path, args.weights, args.verify, args.conf, args.iou)

    print(
        "\nNo --verify image given. Run at least once with a photo of a person "
        "(ideally one with a hard hat and one without) before trusting the "
        "detector:\n"
        f"  python Tools/export_onnx.py --imgsz {args.imgsz} --verify-only "
        f"--verify yourphoto.jpg"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
