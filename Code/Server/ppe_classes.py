"""Class vocabulary for the SH17 PPE model (yolo8n.pt).

Dependency-free on purpose: both ppe_config.py and ppe_detector.py import this,
so it must not pull in numpy, cv2 or onnxruntime.

The order below IS the model's output channel order, read directly out of the
checkpoint's pickle (`best/data.pkl`, the `names` dict). Do not reorder: index
i in the ONNX class logits means CLASS_NAMES[i] and nothing else.

Model provenance: yolov8n fine-tuned on the SH17 safety dataset
(safe_human.yaml, run "safehuman_nano_v0_14", ultralytics 8.2.38, imgsz=640,
200 epochs). The weights carry ultralytics' AGPL-3.0 metadata.
"""

CLASS_NAMES: tuple[str, ...] = (
    "person",        # 0
    "ear",           # 1
    "ear-mufs",      # 2
    "face",          # 3
    "face-guard",    # 4
    "face-mask",     # 5
    "foot",          # 6
    "tool",          # 7
    "glasses",       # 8
    "gloves",        # 9
    "helmet",        # 10
    "hands",         # 11
    "head",          # 12
    "medical-suit",  # 13
    "shoes",         # 14
    "safety-suit",   # 15
    "safety-vest",   # 16
)

NUM_CLASSES = len(CLASS_NAMES)

CLASS_IDS: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

PERSON = "person"

# The model has no "no-helmet" class. Violations are inferred from the geometry
# of a frame, and this dataset's key property is that it labels bare body parts
# and the PPE that covers them as SEPARATE classes. So for these items a
# violation rests on POSITIVE evidence -- we see an uncovered body part --
# rather than on the absence of a detection.
#
#   required item -> (classes that satisfy it, classes that prove it's missing)
#
# An item whose "missing" tuple is empty can only ever be judged by absence,
# which is far weaker evidence: no box might mean no PPE, or might mean the
# person turned around. ppe_rules.py holds those to a stricter standard.
PPE_ITEMS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "helmet":          (("helmet",),                                     ("head",)),
    "gloves":          (("gloves",),                                     ("hands",)),
    "shoes":           (("shoes",),                                      ("foot",)),
    "face-mask":       (("face-mask", "face-guard"),                     ("face",)),
    "eye-protection":  (("glasses", "face-guard"),                       ("face",)),
    "ear-protection":  (("ear-mufs",),                                   ("ear",)),
    # Absence-only: the dataset has no "bare torso" class.
    "safety-vest":     (("safety-vest", "safety-suit", "medical-suit"),  ()),
}


def validate_class_name(name: str) -> str:
    """Raise if `name` is not a real model class. Catches config typos at load."""
    if name not in CLASS_IDS:
        raise ValueError(
            f"unknown model class {name!r}; expected one of {', '.join(CLASS_NAMES)}"
        )
    return name


def validate_item_name(name: str) -> str:
    """Raise if `name` is not a requirable PPE item."""
    if name not in PPE_ITEMS:
        raise ValueError(
            f"unknown PPE item {name!r}; expected one of {', '.join(PPE_ITEMS)}"
        )
    return name


def is_positive_evidence_item(item: str) -> bool:
    """True when a violation of `item` can be proven by an uncovered body part.

    These are the trustworthy ones. `safety-vest` is the notable False.
    """
    return bool(PPE_ITEMS[validate_item_name(item)][1])


if __name__ == "__main__":
    print(f"{NUM_CLASSES} classes:")
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {i:2d}  {n}")
    print("\nrequirable PPE items:")
    for item, (present, missing) in PPE_ITEMS.items():
        kind = "positive-evidence" if missing else "ABSENCE-ONLY (weak)"
        print(f"  {item:16s} satisfied_by={list(present)}")
        print(f"  {'':16s} missing_if_bare={list(missing) or '-'}  [{kind}]")
