"""On-disk queue for violation snapshots.

The disk IS the queue. No SQLite on the robot, no in-memory buffer: a
confirmed violation becomes two files, and the uploader drains them. That
gives offline resilience for free -- the robot patrols perfectly happily with
the portal switched off, accumulates evidence, and backfills when it returns,
surviving reboots without any extra machinery.

    snapshots/pending/   <stamp>_<event_id>.jpg + .json   waiting to upload
    snapshots/sent/      successfully delivered (pruned after N days)
    snapshots/failed/    permanently rejected (bad payload); kept for inspection

Atomicity matters here because the robot can lose power mid-write at any
moment. Both files are written to .tmp and then os.replace'd, and **the JSON
is the commit marker**: iter_pending only yields a pair once the .json exists,
so a half-written event is invisible rather than corrupt. An orphaned .jpg
from an interrupted write is reaped by prune().

The stored image is the RAW frame, deliberately not annotated. The portal
draws the bounding box as a CSS overlay from the recorded coordinates, which
keeps the evidence pristine and re-reviewable, and saves the Pi the work of
drawing.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PendingEvent:
    event_id: str
    image: Path
    meta: Path

    def load_meta(self) -> dict:
        return json.loads(self.meta.read_text(encoding="utf-8"))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clock_is_synced() -> bool:
    """Whether NTP has set the clock.

    A Pi 5 has no battery-backed RTC by default, so a robot that boots without
    a network has a meaningless wall clock and would stamp every snapshot with
    a nonsense time. The portal shows a warning badge when this is false.
    """
    try:
        import subprocess

        res = subprocess.run(
            ["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
            capture_output=True, text=True, timeout=3,
        )
        return res.returncode == 0 and res.stdout.strip().lower() == "yes"
    except Exception:
        return False  # not Linux, or timedatectl missing: don't claim synced


def cpu_temp_c() -> float | None:
    """Pi CPU temperature, for spotting thermal throttling in the portal."""
    try:
        raw = Path("/sys/class/thermal/thermal_zone0/temp").read_text()
        return round(int(raw.strip()) / 1000.0, 1)
    except Exception:
        return None


def throttled_flags() -> str | None:
    """`vcgencmd get_throttled`. Bit 0 set means undervoltage was detected.

    Worth recording: a motor stall current spike coinciding with an inference
    spike can brown out the regulator and reboot the Pi, which otherwise looks
    exactly like a software crash.
    """
    try:
        import subprocess

        res = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=3
        )
        if res.returncode == 0:
            return res.stdout.strip().split("=")[-1]
    except Exception:
        pass
    return None


class SnapshotStore:
    """Atomic, bounded, crash-safe event queue on the filesystem."""

    def __init__(
        self,
        root: str | Path,
        jpeg_quality: int = 85,
        max_pending_mb: int = 2000,
        sent_retention_days: int = 7,
        min_free_disk_mb: int = 1024,
    ) -> None:
        self.root = Path(root)
        self.pending = self.root / "pending"
        self.sent = self.root / "sent"
        self.failed = self.root / "failed"
        for folder in (self.pending, self.sent, self.failed):
            folder.mkdir(parents=True, exist_ok=True)
        self.jpeg_quality = jpeg_quality
        self.max_pending_mb = max_pending_mb
        self.sent_retention_days = sent_retention_days
        self.min_free_disk_mb = min_free_disk_mb

    # ------------------------------------------------------------------ write

    def write_event(self, frame_bgr, payload: dict) -> str:
        """Persist one violation. Returns the event_id.

        `frame_bgr` is a raw BGR numpy frame; `payload` is the metadata dict,
        which must carry an event_id (one is generated if absent).
        """
        import cv2

        event_id = str(payload.get("event_id") or uuid.uuid4().hex)
        payload.setdefault("event_id", event_id)
        payload.setdefault("schema", SCHEMA_VERSION)
        payload.setdefault("captured_at", utc_now_iso())

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")[:-3]
        base = f"{stamp}_{event_id[:12]}"
        img_path = self.pending / f"{base}.jpg"
        meta_path = self.pending / f"{base}.json"

        ok, buf = cv2.imencode(
            ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("cv2.imencode failed for violation frame")
        payload["image_bytes"] = int(buf.nbytes)

        # Image first, then the JSON. The JSON is the commit marker, so an
        # interrupted write leaves an orphan jpg (ignored, later reaped) rather
        # than a metadata record pointing at a file that does not exist.
        img_tmp = img_path.with_suffix(".jpg.tmp")
        img_tmp.write_bytes(buf.tobytes())
        os.replace(img_tmp, img_path)

        meta_tmp = meta_path.with_suffix(".json.tmp")
        meta_tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(meta_tmp, meta_path)

        self.enforce_limits()
        return event_id

    # ------------------------------------------------------------------- read

    def iter_pending(self) -> Iterator[PendingEvent]:
        """Oldest first, so the portal receives events in the order they happened."""
        for meta_path in sorted(self.pending.glob("*.json")):
            img_path = meta_path.with_suffix(".jpg")
            if not img_path.is_file():
                continue  # metadata without an image: reaped by prune()
            try:
                event_id = json.loads(meta_path.read_text(encoding="utf-8"))["event_id"]
            except Exception:
                self._quarantine(meta_path, img_path, "unreadable metadata")
                continue
            yield PendingEvent(event_id=str(event_id), image=img_path, meta=meta_path)

    def pending_count(self) -> int:
        return sum(1 for _ in self.pending.glob("*.json"))

    def pending_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.pending.iterdir() if f.is_file())

    # ----------------------------------------------------------- transitions

    def mark_sent(self, event: PendingEvent) -> None:
        self._move(event, self.sent)

    def mark_failed(self, event: PendingEvent, reason: str) -> None:
        """Permanent rejection (a 4xx that will never succeed on retry)."""
        try:
            meta = event.load_meta()
            meta["upload_failure"] = {"reason": reason, "at": utc_now_iso()}
            event.meta.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        except Exception:
            pass
        self._move(event, self.failed)
        print(f"[snapshots] permanently failed {event.event_id[:12]}: {reason}")

    def _quarantine(self, meta_path: Path, img_path: Path, reason: str) -> None:
        print(f"[snapshots] quarantining {meta_path.name}: {reason}")
        for src in (meta_path, img_path):
            if src.is_file():
                try:
                    shutil.move(str(src), str(self.failed / src.name))
                except Exception:
                    src.unlink(missing_ok=True)

    def _move(self, event: PendingEvent, dest_dir: Path) -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in (event.image, event.meta):
            if not src.is_file():
                continue
            dest = dest_dir / src.name
            try:
                os.replace(src, dest)
            except OSError:
                shutil.move(str(src), str(dest))

    # ------------------------------------------------------------ maintenance

    def enforce_limits(self) -> None:
        """Keep the queue bounded.

        A robot that cannot reach the portal for a week must not fill its SD
        card and wedge itself. Losing the oldest evidence is strictly better
        than a robot that stops working, so when the cap is hit the oldest
        pending events are dropped -- loudly.
        """
        free_mb = self._free_mb()
        over_size = self.pending_bytes() > self.max_pending_mb * 1024 * 1024
        low_disk = free_mb is not None and free_mb < self.min_free_disk_mb
        if not (over_size or low_disk):
            return

        why = "pending queue over cap" if over_size else f"only {free_mb}MB free"
        events = list(self.iter_pending())
        drop = max(1, len(events) // 10)
        print(
            f"[snapshots] WARNING: {why}; dropping the {drop} oldest pending "
            f"event(s) of {len(events)} to keep the robot running"
        )
        for event in events[:drop]:
            event.image.unlink(missing_ok=True)
            event.meta.unlink(missing_ok=True)

    def prune(self) -> int:
        """Delete expired `sent` events and orphaned temp/image files."""
        removed = 0
        cutoff = time.time() - self.sent_retention_days * 86400
        for f in self.sent.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
                removed += 1

        for f in self.pending.iterdir():
            if not f.is_file():
                continue
            # Orphan .jpg (no committing .json) or a leftover .tmp from an
            # interrupted write.
            if f.suffix == ".tmp" or (
                f.suffix == ".jpg" and not f.with_suffix(".json").is_file()
            ):
                if time.time() - f.stat().st_mtime > 60:  # let live writes finish
                    f.unlink(missing_ok=True)
                    removed += 1
        return removed

    def _free_mb(self) -> int | None:
        try:
            return shutil.disk_usage(self.root).free // (1024 * 1024)
        except Exception:
            return None

    def stats(self) -> dict:
        return {
            "pending": self.pending_count(),
            "pending_mb": round(self.pending_bytes() / 1e6, 2),
            "sent": sum(1 for _ in self.sent.glob("*.json")),
            "failed": sum(1 for _ in self.failed.glob("*.json")),
            "free_mb": self._free_mb(),
        }


def build_payload(
    event,
    cfg,
    frame_shape: tuple[int, int],
    battery_v: float | None = None,
    patrol_state: str | None = None,
    pan_angle: int | None = None,
    sonic_cm: float | None = None,
    extra: dict | None = None,
) -> dict:
    """Assemble the metadata payload for a confirmed ViolationEvent.

    On "location": this chassis has no wheel encoders, no IMU and no map, so
    there is genuinely no position to report. `zone_label` from config is the
    only trustworthy spatial field; anything else would imply localisation the
    robot does not have.
    """
    observation = event.observation
    payload: dict = {
        "schema": SCHEMA_VERSION,
        "event_id": uuid.uuid4().hex,
        "robot_id": cfg.robot_id,
        "zone_label": cfg.zone_label,
        "captured_at": utc_now_iso(),
        "clock_synced": clock_is_synced(),
        "monotonic_s": round(time.monotonic(), 3),
        "required_ppe": list(cfg.rules.required_ppe),
        "missing_ppe": list(event.items),
        "track_id": event.track_id,
        "person_conf": round(float(observation.person.conf), 4),
        "person_bbox": [round(float(v), 1) for v in observation.person.xyxy],
        "frame_size": [int(frame_shape[0]), int(frame_shape[1])],
        "evidence": {
            item: {
                **observation.items[item].to_json(),
                "frames_agree": event.frames_agree.get(item),
            }
            for item in event.items
            if item in observation.items
        },
        "detections": [
            d.to_json()
            for dets in observation.attached.values()
            for d in dets
        ],
        "detector": {
            "model": Path(cfg.model.confirm_onnx).name,
            "conf": cfg.model.conf_default,
            "iou": cfg.model.iou_nms,
        },
        "temporal": {
            "window_m": cfg.temporal.window_m,
            "positive_n": cfg.temporal.positive_n,
            "absence_n": cfg.temporal.absence_n,
        },
        "patrol": {
            "state": patrol_state,
            "pan_angle": pan_angle,
            "sonic_center_cm": sonic_cm,
        },
        "battery_v": round(battery_v, 2) if battery_v else None,
        "cpu_temp_c": cpu_temp_c(),
        "throttled": throttled_flags(),
    }
    if extra:
        payload.update(extra)
    return payload


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect or maintain the snapshot queue.")
    ap.add_argument("--dir", default=None, help="queue root (default: from config)")
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument(
        "--fake", type=int, default=0, metavar="N",
        help="write N synthetic events, to test the uploader without a robot",
    )
    args = ap.parse_args()

    from ppe_config import load_config

    cfg = load_config()
    root = args.dir or cfg.resolve_path(cfg.snapshots.dir)
    store = SnapshotStore(
        root,
        jpeg_quality=cfg.snapshots.jpeg_quality,
        max_pending_mb=cfg.snapshots.max_pending_mb,
        sent_retention_days=cfg.snapshots.sent_retention_days,
    )
    print(f"queue at {store.root}")

    if args.fake:
        import numpy as np

        for i in range(args.fake):
            frame = np.full((480, 640, 3), 60, dtype=np.uint8)
            frame[100:400, 250:390] = (100, 110, 125)
            eid = store.write_event(
                frame,
                {
                    "robot_id": cfg.robot_id,
                    "zone_label": cfg.zone_label,
                    "missing_ppe": ["helmet"],
                    "person_conf": 0.9,
                    "person_bbox": [250, 100, 390, 400],
                    "frame_size": [640, 480],
                    "clock_synced": True,
                    "evidence": {
                        "helmet": {
                            "status": "MISSING_POSITIVE",
                            "support": {"head": 0.87},
                            "frames_agree": "6/10",
                            "reason": "synthetic test event",
                        }
                    },
                },
            )
            print(f"  wrote {eid[:12]}")

    if args.prune:
        print(f"pruned {store.prune()} file(s)")

    if args.list:
        for event in store.iter_pending():
            meta = event.load_meta()
            print(f"  {event.event_id[:12]}  {meta.get('captured_at', '?')[:19]}  "
                  f"{','.join(meta.get('missing_ppe', []))}")

    for key, value in store.stats().items():
        print(f"  {key:12s} {value}")
