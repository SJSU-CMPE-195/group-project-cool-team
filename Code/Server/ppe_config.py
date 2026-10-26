"""Configuration for the PPE patrol.

Loads ppe_config.json into frozen dataclasses and validates every class/item
name against the model vocabulary at load time, so a typo fails loudly at
startup instead of silently disabling a rule halfway through a patrol.

Layering, highest priority last:
    ppe_config.json          committed defaults
    ppe_config.local.json    gitignored; portal IP, API key, per-robot tweaks
    $PPE_CONFIG              explicit path, replaces the pair above

The merge is recursive over dicts, so a local file need only name the keys it
overrides.

Hardware identity (Connect_Version / Pcb_Version / Pi_Version) is NOT here --
that stays in params.json via parameter.py, which has a different lifecycle.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from ppe_classes import validate_class_name, validate_item_name

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "ppe_config.json"
LOCAL_CONFIG_PATH = HERE / "ppe_config.local.json"


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Detector backend and thresholds.

    Three distinct confidence gates, which are easy to confuse:

    class_conf    minimum to be considered a detection at all.
    evidence_conf minimum to be USED AS PROOF that PPE is missing. Higher.
                  Seeing a `head` at 0.35 is fine; declaring a violation on
                  the strength of it is not.
    veto_conf     minimum for a PPE detection to clear a person outright.
                  Deliberately low -- positive evidence should win easily.
    """

    scan_onnx: str = "models/ppe_yolo8n_448.onnx"
    confirm_onnx: str = "models/ppe_yolo8n_640.onnx"
    pt_path: str = "../../yolo8n.pt"
    backend: str = "onnx"  # "onnx" | "ultralytics" (PC validation only)
    conf_default: float = 0.30
    iou_nms: float = 0.45
    intra_op_threads: int = 3  # leave a core for the camera encoder + drive loop
    class_conf: dict[str, float] = field(
        default_factory=lambda: {
            "person": 0.45,
            "helmet": 0.40,
            "head": 0.50,
            "safety-vest": 0.40,
            "hands": 0.45,
            "gloves": 0.40,
            "foot": 0.45,
            "shoes": 0.40,
            "face": 0.45,
        }
    )
    evidence_conf: dict[str, float] = field(
        default_factory=lambda: {
            "head": 0.60,
            "hands": 0.55,
            "foot": 0.55,
            "face": 0.55,
            "ear": 0.60,
        }
    )
    veto_conf: dict[str, float] = field(
        default_factory=lambda: {
            "helmet": 0.35,
            "safety-vest": 0.35,
            "safety-suit": 0.35,
            "medical-suit": 0.35,
            "gloves": 0.35,
            "shoes": 0.35,
            "face-mask": 0.35,
            "face-guard": 0.35,
            "glasses": 0.35,
            "ear-mufs": 0.35,
        }
    )

    def __post_init__(self) -> None:
        for mapping in (self.class_conf, self.evidence_conf, self.veto_conf):
            for name in mapping:
                validate_class_name(name)
        if self.backend not in ("onnx", "ultralytics"):
            raise ValueError(f"model.backend must be onnx|ultralytics, got {self.backend!r}")

    def conf_for(self, cls_name: str) -> float:
        return self.class_conf.get(cls_name, self.conf_default)

    def evidence_threshold(self, cls_name: str) -> float:
        """Bar for treating a bare-body-part detection as proof of a violation."""
        return self.evidence_conf.get(cls_name, max(self.conf_for(cls_name), 0.55))

    def veto_threshold(self, cls_name: str) -> float:
        """Bar for a PPE detection to clear a person. Low by design."""
        return self.veto_conf.get(cls_name, 0.35)


@dataclass(frozen=True)
class RulesConfig:
    """Per-frame violation inference.

    `required_ppe` defaults to helmet only. That is the strongest signal this
    model offers: `head` and `helmet` are separate classes, so a bare head is
    affirmative proof rather than a missing detection. Add safety-vest only
    after measuring precision from the portal's review flags -- it is
    absence-only and will generate false positives on anyone turned away.
    """

    required_ppe: list[str] = field(default_factory=lambda: ["helmet"])
    assoc_containment_min: float = 0.60
    assoc_ambiguity_margin: float = 0.15
    person_box_pad: float = 0.05
    ppe_cover_overlap_min: float = 0.30
    # Vertical band within the person box where each item may legitimately
    # appear, as (top_frac, bottom_frac). A "helmet" at knee height is a false
    # detection, not PPE.
    region_gates: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "helmet": (0.00, 0.40),
            "head": (0.00, 0.40),
            "face": (0.00, 0.40),
            "ear": (0.00, 0.40),
            "safety-vest": (0.15, 0.75),
            "gloves": (0.20, 0.90),
            "hands": (0.20, 0.90),
            "shoes": (0.70, 1.00),
            "foot": (0.70, 1.00),
        }
    )
    min_person_height_frac: float = 0.35
    min_person_area_frac: float = 0.02
    edge_margin_px: int = 8
    absence_requires_full_view: bool = True

    def __post_init__(self) -> None:
        for item in self.required_ppe:
            validate_item_name(item)
        for name, band in self.region_gates.items():
            validate_class_name(name)
            lo, hi = band
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError(f"region_gates[{name}] must be 0<=lo<hi<=1, got {band}")

    def region_gate(self, cls_name: str) -> tuple[float, float]:
        return self.region_gates.get(cls_name, (0.0, 1.0))


@dataclass(frozen=True)
class TemporalConfig:
    """N-of-M confirmation.

    Only frames captured while the robot is stopped and the camera has settled
    are fed to this -- see patrol's camera_steady handshake. N-of-M over frames
    that pan across a scene would be meaningless.
    """

    window_m: int = 10
    positive_n: int = 5  # bare-body-part proof: 5 of 10
    absence_n: int = 8   # absence-only items: stricter, 8 of 10
    min_track_hits: int = 4

    def __post_init__(self) -> None:
        if not 0 < self.positive_n <= self.window_m:
            raise ValueError("temporal.positive_n must be in 1..window_m")
        if not 0 < self.absence_n <= self.window_m:
            raise ValueError("temporal.absence_n must be in 1..window_m")


@dataclass(frozen=True)
class CooldownConfig:
    """Stops one person in view generating a hundred snapshots.

    Three layers, because each defeats a different failure: per-track cooldown
    handles the normal case; histogram re-ID catches a track that was lost and
    reacquired under a new id; the global cap is a backstop for tracker churn.
    """

    per_person_s: float = 90.0
    global_min_interval_s: float = 8.0
    max_per_hour: int = 60
    reid_similarity: float = 0.85
    reid_window_s: float = 120.0


@dataclass(frozen=True)
class PatrolConfig:
    """Drive behaviour.

    Pan angles are 60/90/120, NOT upstream's 30/90/150 (car.py:72). The
    factory servo test only sweeps pan 50-110 (test.py:91-96), so upstream's
    own autonomous mode drives the bracket past its stated safe range.

    The ultrasonic sensor shares the pan/tilt bracket with the camera, so
    sweeping for obstacle bearings necessarily swings the camera. Sweeping
    therefore only happens while stopped, with camera_steady false.
    """

    cruise_duty: int = 700
    pivot_duty: int = 1200
    reverse_duty: int = -1300
    stop_distance_cm: float = 35.0
    stuck_distance_cm: float = 20.0
    pan_center: int = 90
    pan_left: int = 60
    pan_right: int = 120
    tilt: int = 100
    pan_settle_s: float = 0.35
    hold_max_s: float = 8.0
    post_hold_ignore_s: float = 5.0
    tick_s: float = 0.2
    scan_interval_s: float = 0.15
    person_trigger_conf: float = 0.50
    min_battery_v: float = 6.6
    thermal_throttle_c: float = 80.0
    buzz_on_violation: bool = True
    buzzer_on_violation_s: float = 0.4

    # servo.py does no angle clamping, and a servo driven past its mechanical
    # stop stalls, draws current and gets hot. The factory test sweeps pan
    # 50-110 (test.py:91-96) while upstream's own ultrasonic mode drives
    # 30-150 (car.py:72), so the bracket evidently tolerates more than the
    # tested range. We default to a 60/120 sweep -- wide enough for useful
    # obstacle bearings, comfortably inside what upstream already does -- and
    # only warn outside 40-140, where we would be exceeding both.
    PAN_SANE_MIN = 40
    PAN_SANE_MAX = 140

    def __post_init__(self) -> None:
        for name in ("pan_center", "pan_left", "pan_right"):
            angle = getattr(self, name)
            if not self.PAN_SANE_MIN <= angle <= self.PAN_SANE_MAX:
                print(
                    f"[ppe_config] WARNING: patrol.{name}={angle} is outside "
                    f"{self.PAN_SANE_MIN}-{self.PAN_SANE_MAX}; servo.py does not "
                    f"clamp angles and a stalled servo overheats"
                )
        if not self.pan_left < self.pan_center < self.pan_right:
            raise ValueError(
                f"patrol pan angles must satisfy left<center<right, got "
                f"{self.pan_left}/{self.pan_center}/{self.pan_right}"
            )


@dataclass(frozen=True)
class CameraConfig:
    """640x480 is deliberate, not arbitrary.

    The model was trained at imgsz=640. A 640x480 frame letterboxes into
    640x640 with padding only and ZERO downscaling, which is the cheapest
    accuracy available. Upstream's 400x300 stream (main.py:46) would force a
    downscale and cost small-object recall -- exactly what a helmet is at
    patrol distance.
    """

    stream_size: tuple[int, int] = (640, 480)
    hflip: bool = False
    vflip: bool = False
    buffer_count: int = 4
    mjpeg_port: int = 8080
    mjpeg_quality: int = 70


@dataclass(frozen=True)
class SnapshotConfig:
    dir: str = "snapshots"
    jpeg_quality: int = 85
    max_pending_mb: int = 2000
    sent_retention_days: int = 7
    min_free_disk_mb: int = 1024


@dataclass(frozen=True)
class PortalConfig:
    url: str = "http://127.0.0.1:5050"
    api_key: str = "change-me"
    timeout_connect_s: float = 3.0
    timeout_read_s: float = 10.0
    backoff_max_s: float = 300.0
    poll_interval_s: float = 2.0

    def resolved_api_key(self) -> str:
        """Allow "env:VARNAME" so the real secret need not sit in a file."""
        if self.api_key.startswith("env:"):
            var = self.api_key[4:]
            value = os.environ.get(var)
            if not value:
                raise RuntimeError(
                    f"portal.api_key points at ${var} but that variable is unset"
                )
            return value
        return self.api_key

    @property
    def ingest_url(self) -> str:
        return self.url.rstrip("/") + "/api/v1/violations"

    @property
    def health_url(self) -> str:
        return self.url.rstrip("/") + "/api/v1/health"


@dataclass(frozen=True)
class PPEConfig:
    robot_id: str = "freenove-01"
    zone_label: str = "unspecified"
    dry_run: bool = False  # detect and snapshot locally, never upload
    model: ModelConfig = field(default_factory=ModelConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    cooldown: CooldownConfig = field(default_factory=CooldownConfig)
    patrol: PatrolConfig = field(default_factory=PatrolConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    snapshots: SnapshotConfig = field(default_factory=SnapshotConfig)
    portal: PortalConfig = field(default_factory=PortalConfig)

    def resolve_path(self, maybe_relative: str) -> Path:
        """Resolve a config path relative to Code/Server, not to the cwd.

        parameter.py's PARAM_FILE is cwd-relative and that is a known trap;
        model paths should not inherit it.
        """
        p = Path(maybe_relative)
        return p if p.is_absolute() else (HERE / p).resolve()


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

_SECTIONS: dict[str, type] = {
    "model": ModelConfig,
    "rules": RulesConfig,
    "temporal": TemporalConfig,
    "cooldown": CooldownConfig,
    "patrol": PatrolConfig,
    "camera": CameraConfig,
    "snapshots": SnapshotConfig,
    "portal": PortalConfig,
}


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _build_section(cls: type, raw: dict[str, Any], section: str):
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) in config section {section!r}: {', '.join(sorted(unknown))}"
        )
    kwargs = dict(raw)
    # JSON has no tuples; coerce the fields that want them.
    for f in fields(cls):
        if f.name not in kwargs:
            continue
        if f.name == "stream_size" and isinstance(kwargs[f.name], list):
            kwargs[f.name] = tuple(kwargs[f.name])
        if f.name == "region_gates" and isinstance(kwargs[f.name], dict):
            kwargs[f.name] = {k: tuple(v) for k, v in kwargs[f.name].items()}
    return cls(**kwargs)


def load_config(path: str | os.PathLike[str] | None = None) -> PPEConfig:
    """Load, merge and validate configuration.

    Never raises for a missing file -- built-in defaults are a valid config,
    which keeps unit tests and the PC-side tools free of setup. It does raise
    for malformed JSON, unknown keys and invalid class/item names, because
    those are always mistakes.
    """
    raw: dict[str, Any] = {}

    explicit = path or os.environ.get("PPE_CONFIG")
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"config not found: {p}")
        raw = json.loads(p.read_text(encoding="utf-8"))
    else:
        for candidate in (DEFAULT_CONFIG_PATH, LOCAL_CONFIG_PATH):
            if candidate.is_file():
                try:
                    layer = json.loads(candidate.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{candidate.name} is not valid JSON: {exc}") from exc
                raw = _deep_merge(raw, layer)

    top = {k: v for k, v in raw.items() if k not in _SECTIONS}
    unknown_top = set(top) - {f.name for f in fields(PPEConfig)}
    if unknown_top:
        raise ValueError(f"unknown top-level config key(s): {', '.join(sorted(unknown_top))}")

    sections = {
        name: _build_section(cls, raw.get(name, {}) or {}, name)
        for name, cls in _SECTIONS.items()
    }
    return PPEConfig(**top, **sections)


def as_dict(obj: Any) -> Any:
    """Recursive dataclass -> plain dict, for printing and for event metadata."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: as_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: as_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_dict(v) for v in obj]
    return obj


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Print the resolved PPE patrol config.")
    ap.add_argument("--config", help="explicit config path (overrides layering)")
    ap.add_argument("--section", help="print only this section")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = as_dict(cfg)
    if args.section:
        if args.section not in data:
            raise SystemExit(f"no such section: {args.section}; have {', '.join(data)}")
        data = data[args.section]
    print(json.dumps(data, indent=2))

    print("\n--- derived ---")
    print(f"required PPE      : {cfg.rules.required_ppe}")
    print(f"scan model        : {cfg.resolve_path(cfg.model.scan_onnx)}")
    print(f"confirm model     : {cfg.resolve_path(cfg.model.confirm_onnx)}")
    print(f"portal ingest     : {cfg.portal.ingest_url}")
    print(f"snapshot dir      : {cfg.resolve_path(cfg.snapshots.dir)}")
