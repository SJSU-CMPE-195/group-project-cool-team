"""Populate the portal with synthetic violations, so you can see it working.

Generates plausible-looking frames (a simple drawn scene, not real photos) and
POSTs them through the real ingest endpoint -- so this also exercises auth,
validation, idempotency, thumbnailing and the gallery end to end.

    python Portal/app.py                 # terminal 1
    python Portal/seed_demo.py           # terminal 2
    # then open http://127.0.0.1:5050

    python Portal/seed_demo.py --clear   # remove seeded rows afterwards
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone

SEED_ZONE = "demo-seed"

SCENARIOS = [
    # (missing items, evidence status, reason, person conf, bg colour)
    (["helmet"], "MISSING_POSITIVE",
     "bare head at {c:.2f}, no overlapping helmet", 0.93, (58, 64, 74)),
    (["helmet"], "MISSING_POSITIVE",
     "bare head at {c:.2f}; helmet detected but not covering it (not worn)",
     0.88, (70, 62, 58)),
    (["helmet", "safety-vest"], "MISSING_POSITIVE",
     "bare head at {c:.2f}, no overlapping helmet", 0.91, (52, 60, 68)),
    (["safety-vest"], "MISSING_ABSENT",
     "no safety-vest detected on a clearly-visible person", 0.85, (64, 68, 60)),
]


def make_frame(colour, bbox, label: str, width=640, height=480) -> bytes:
    """Draw a stand-in 'camera frame'. Deliberately synthetic, not a fake photo."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), colour)
    d = ImageDraw.Draw(img)

    # Floor line and a couple of background shapes, so frames differ visibly.
    d.rectangle([0, int(height * 0.78), width, height],
                fill=tuple(max(0, c - 14) for c in colour))
    for _ in range(3):
        x = random.randint(0, width - 60)
        w = random.randint(30, 90)
        d.rectangle([x, int(height * 0.5), x + w, int(height * 0.78)],
                    fill=tuple(min(255, c + 10) for c in colour))

    # The "person": a simple figure inside the reported bbox.
    x1, y1, x2, y2 = bbox
    d.rectangle([x1, y1 + (y2 - y1) * 0.22, x2, y2], fill=(96, 104, 120))
    head_r = (x2 - x1) * 0.19
    cx = (x1 + x2) / 2
    cy = y1 + head_r
    d.ellipse([cx - head_r, cy - head_r, cx + head_r, cy + head_r], fill=(188, 152, 126))

    d.text((8, 8), "SYNTHETIC DEMO FRAME", fill=(230, 230, 230))
    d.text((8, 22), label, fill=(230, 180, 180))

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82)
    return buf.getvalue()


def build(index: int, when: datetime) -> tuple[dict, bytes]:
    missing, _status, reason_tpl, person_conf, colour = random.choice(SCENARIOS)
    support_conf = round(random.uniform(0.62, 0.95), 3)

    x1 = random.randint(120, 330)
    y1 = random.randint(70, 130)
    bbox = [x1, y1, x1 + random.randint(120, 175), y1 + random.randint(250, 310)]

    # Status is decided by the ITEM, not the scenario. safety-vest is
    # absence-only -- the dataset has no "bare torso" class, so a vest
    # violation can never be proven by seeing uncovered skin, and labelling it
    # MISSING_POSITIVE would misrepresent what the real engine produces.
    # It also carries the stricter 8-of-10 agreement bar.
    absence_only = {"safety-vest"}
    evidence = {}
    for item in missing:
        if item in absence_only:
            evidence[item] = {
                "status": "MISSING_ABSENT",
                "support": {"person": person_conf},
                "frames_agree": f"{random.randint(8, 10)}/10",
                "reason": f"no {item} detected on a clearly-visible person",
            }
        else:
            evidence[item] = {
                "status": "MISSING_POSITIVE",
                "support": {"head": support_conf, "person": person_conf},
                "frames_agree": f"{random.randint(5, 10)}/10",
                "reason": reason_tpl.format(c=support_conf),
            }

    meta = {
        "schema": 1,
        "event_id": uuid.uuid4().hex,
        "robot_id": "freenove-01" if index % 7 else "freenove-02",
        "zone_label": SEED_ZONE,
        "captured_at": when.isoformat(),
        "clock_synced": index % 11 != 0,
        "missing_ppe": missing,
        "person_conf": person_conf,
        "person_bbox": bbox,
        "frame_size": [640, 480],
        "track_id": random.randint(1, 60),
        "battery_v": round(random.uniform(6.8, 8.3), 2),
        "cpu_temp_c": round(random.uniform(52, 78), 1),
        "evidence": evidence,
        "detector": {
            "model": "ppe_yolo8n_640.onnx",
            "imgsz": 640,
            "conf": 0.30,
            "iou": 0.45,
        },
        "patrol": {"state": "HOLD", "pan_angle": 90,
                   "sonic_center_cm": round(random.uniform(60, 240), 1)},
        "note": "Seeded demo data, not a real detection.",
    }
    return meta, make_frame(colour, bbox, " + ".join(missing).upper())


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed the portal with demo violations.")
    ap.add_argument("--url", default="http://127.0.0.1:5050")
    ap.add_argument("--key", default=None, help="API key (default: $PPE_API_KEY or change-me)")
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--days", type=int, default=6, help="spread over this many days")
    ap.add_argument("--clear", action="store_true",
                    help="delete previously seeded rows instead of adding more")
    args = ap.parse_args()

    import os

    import requests

    key = args.key or os.environ.get("PPE_API_KEY", "change-me")

    if args.clear:
        return clear(args.url)

    try:
        health = requests.get(f"{args.url}/api/v1/health", timeout=4)
        health.raise_for_status()
    except Exception as exc:
        print(f"portal not reachable at {args.url}: {exc}")
        print("start it first:  python Portal/app.py")
        return 1
    print(f"portal is up, {health.json()['count']} existing violations")

    now = datetime.now(timezone.utc)
    created = dupes = failed = 0
    for i in range(args.count):
        when = now - timedelta(
            days=random.uniform(0, args.days), minutes=random.uniform(0, 1400)
        )
        meta, image = build(i, when)
        try:
            res = requests.post(
                f"{args.url}/api/v1/violations",
                headers={"X-API-Key": key},
                files={
                    "metadata": ("meta.json", json.dumps(meta), "application/json"),
                    "image": (f"{meta['event_id']}.jpg", image, "image/jpeg"),
                },
                timeout=10,
            )
        except Exception as exc:
            print(f"  [{i + 1:2d}] request failed: {exc}")
            failed += 1
            continue

        if res.status_code == 201:
            created += 1
            print(f"  [{i + 1:2d}] created id={res.json()['id']:<4} "
                  f"{','.join(meta['missing_ppe'])}  {meta['captured_at'][:16]}")
        elif res.status_code == 409:
            dupes += 1
        else:
            failed += 1
            print(f"  [{i + 1:2d}] HTTP {res.status_code}: {res.text[:160]}")

    print(f"\ncreated {created}, duplicate {dupes}, failed {failed}")

    # Re-POST one event to demonstrate idempotency, which is what makes the
    # robot's offline retry safe.
    if created:
        meta, image = build(0, now)
        files = {
            "metadata": ("meta.json", json.dumps(meta), "application/json"),
            "image": (f"{meta['event_id']}.jpg", image, "image/jpeg"),
        }
        first = requests.post(f"{args.url}/api/v1/violations",
                              headers={"X-API-Key": key}, files=files, timeout=10)
        files["metadata"] = ("meta.json", json.dumps(meta), "application/json")
        files["image"] = (f"{meta['event_id']}.jpg", image, "image/jpeg")
        second = requests.post(f"{args.url}/api/v1/violations",
                               headers={"X-API-Key": key}, files=files, timeout=10)
        print(f"idempotency check: first POST {first.status_code}, "
              f"repeat POST {second.status_code} "
              f"({'correct' if second.status_code == 409 else 'UNEXPECTED'})")

    print(f"\nopen {args.url}")
    return 0


def clear(url: str) -> int:
    """Delete seeded rows straight from the database (no destructive HTTP route)."""
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import db as dbmod
    from app import DB_PATH, IMAGE_ROOT, THUMB_ROOT

    if not DB_PATH.is_file():
        print("no database yet, nothing to clear")
        return 0

    conn = dbmod.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, image_path, thumb_path FROM violations WHERE zone_label = ?",
        (SEED_ZONE,),
    ).fetchall()
    if not rows:
        print("no seeded rows found")
        return 0

    for row in rows:
        (IMAGE_ROOT / row["image_path"]).unlink(missing_ok=True)
        if row["thumb_path"]:
            (THUMB_ROOT / row["thumb_path"]).unlink(missing_ok=True)
    with conn:
        conn.execute("DELETE FROM violations WHERE zone_label = ?", (SEED_ZONE,))
        conn.execute("DELETE FROM robots")
    conn.close()
    print(f"removed {len(rows)} seeded violations and their images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
