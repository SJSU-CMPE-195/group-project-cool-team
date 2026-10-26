"""End-to-end tests for the admin portal.

Uses Flask's test client against a temporary data directory, so it needs no
network, no robot and no running server, and it never touches your real
portal.sqlite3.

Covers the ingest contract the robot depends on -- especially idempotency,
which is what makes the Pi's retry-after-lost-ack safe -- plus auth, upload
validation and the gallery/detail/review flow.

Run: python test_portal.py
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

API_KEY = "test-key-123"
ADMIN_PASSWORD = "hunter2"


def make_jpeg(width: int = 640, height: int = 480, colour=(40, 90, 150)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buf, "JPEG", quality=80)
    return buf.getvalue()


def meta(
    event_id: str | None = None,
    missing=("helmet",),
    captured_at: str | None = None,
    **extra,
) -> dict:
    payload = {
        "schema": 1,
        "event_id": event_id or uuid.uuid4().hex,
        "robot_id": "freenove-01",
        "zone_label": "workshop",
        "captured_at": captured_at or datetime.now(timezone.utc).isoformat(),
        "clock_synced": True,
        "missing_ppe": list(missing),
        "person_conf": 0.91,
        "person_bbox": [250, 100, 390, 400],
        "frame_size": [640, 480],
        "track_id": 7,
        "battery_v": 7.84,
        "cpu_temp_c": 62.3,
        "evidence": {
            "helmet": {
                "status": "MISSING_POSITIVE",
                "support": {"head": 0.88, "person": 0.91},
                "frames_agree": "6/10",
                "reason": "bare head at 0.88, no overlapping helmet",
            }
        },
    }
    payload.update(extra)
    return payload


def post(client, payload=None, image=None, key=API_KEY, omit=None):
    data = {}
    if omit != "metadata":
        data["metadata"] = json.dumps(payload if payload is not None else meta())
    if omit != "image":
        data["image"] = (io.BytesIO(image if image is not None else make_jpeg()), "f.jpg")
    headers = {"X-API-Key": key} if key is not None else {}
    return client.post(
        "/api/v1/violations", data=data, headers=headers,
        content_type="multipart/form-data",
    )


class Harness:
    """A portal instance on a throwaway data dir."""

    def __init__(self, with_auth: bool = False):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)

        import app as appmod

        self.appmod = appmod
        self._saved = (
            appmod.DATA_DIR, appmod.IMAGE_ROOT,
            appmod.THUMB_ROOT, appmod.INSTANCE_DIR, appmod.DB_PATH,
        )
        appmod.DATA_DIR = root / "data"
        appmod.IMAGE_ROOT = root / "data" / "images"
        appmod.THUMB_ROOT = root / "data" / "thumbs"
        appmod.INSTANCE_DIR = root / "instance"
        appmod.DB_PATH = root / "instance" / "portal.sqlite3"

        import os

        os.environ["PPE_API_KEY"] = API_KEY
        if with_auth:
            os.environ["PPE_ADMIN_PASSWORD"] = ADMIN_PASSWORD
        else:
            os.environ.pop("PPE_ADMIN_PASSWORD", None)

        self.app = appmod.create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def login(self):
        self.client.post("/login", data={"password": ADMIN_PASSWORD})

    def close(self):
        (
            self.appmod.DATA_DIR, self.appmod.IMAGE_ROOT,
            self.appmod.THUMB_ROOT, self.appmod.INSTANCE_DIR,
            self.appmod.DB_PATH,
        ) = self._saved
        try:
            self.tmp.cleanup()
        except Exception:
            pass  # Windows sometimes holds the sqlite file briefly

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# --------------------------------------------------------------------------
# ingest: the contract the robot relies on
# --------------------------------------------------------------------------


def test_ingest_creates_a_violation():
    with Harness() as h:
        res = post(h.client)
        assert res.status_code == 201, res.get_data(as_text=True)
        body = res.get_json()
        assert body["status"] == "created"
        assert isinstance(body["id"], int)


def test_ingest_writes_image_and_thumbnail_to_disk():
    with Harness() as h:
        payload = meta(captured_at="2026-10-22T15:30:12.123456+00:00")
        res = post(h.client, payload)
        assert res.status_code == 201

        rel = f"2026/10/22/{payload['event_id']}.jpg"
        assert (h.appmod.IMAGE_ROOT / rel).is_file(), "image should be on disk by date"
        assert (h.appmod.THUMB_ROOT / rel).is_file(), "thumbnail should be generated"
        # No temp files left behind.
        assert not list(h.appmod.IMAGE_ROOT.rglob("*.tmp"))


def test_duplicate_event_id_returns_409_not_a_second_row():
    """The guarantee that makes the robot's retry safe.

    A POST whose response was lost gets retried. The portal must recognise it
    and report the existing row rather than storing the same violation twice.
    """
    with Harness() as h:
        payload = meta()
        first = post(h.client, payload)
        assert first.status_code == 201

        again = post(h.client, payload)
        assert again.status_code == 409
        body = again.get_json()
        assert body["status"] == "duplicate"
        assert body["id"] == first.get_json()["id"]

        assert h.client.get("/api/v1/health").get_json()["count"] == 1


def test_rejects_wrong_api_key():
    with Harness() as h:
        assert post(h.client, key="nope").status_code == 401
        assert post(h.client, key=None).status_code == 401
        assert h.client.get("/api/v1/health").get_json()["count"] == 0


def test_rejects_a_non_jpeg_upload():
    """Never trust the filename -- a renamed text file must not be stored."""
    with Harness() as h:
        res = post(h.client, image=b"this is definitely not a jpeg")
        assert res.status_code == 400
        assert "JPEG" in res.get_json()["error"]


def test_rejects_truncated_jpeg_header():
    with Harness() as h:
        assert post(h.client, image=b"\xff\xd8\xff").status_code == 400


def test_rejects_oversize_upload():
    with Harness() as h:
        huge = b"\xff\xd8\xff" + b"\x00" * (13 * 1024 * 1024)
        res = post(h.client, image=huge)
        assert res.status_code == 413


def test_rejects_malformed_and_incomplete_metadata():
    with Harness() as h:
        bad = h.client.post(
            "/api/v1/violations",
            data={"metadata": "{not json", "image": (io.BytesIO(make_jpeg()), "f.jpg")},
            headers={"X-API-Key": API_KEY},
            content_type="multipart/form-data",
        )
        assert bad.status_code == 400

        incomplete = meta()
        del incomplete["captured_at"]
        assert post(h.client, incomplete).status_code == 400

        empty_missing = meta()
        empty_missing["missing_ppe"] = []
        assert post(h.client, empty_missing).status_code == 400

        wrong_schema = meta()
        wrong_schema["schema"] = 99
        assert post(h.client, wrong_schema).status_code == 400


def test_rejects_missing_parts():
    with Harness() as h:
        assert post(h.client, omit="image").status_code == 400
        assert post(h.client, omit="metadata").status_code == 400


def test_metadata_is_stored_verbatim():
    """Full payload is kept so a later schema change loses nothing."""
    with Harness() as h:
        payload = meta(custom_future_field={"nested": [1, 2, 3]})
        res = post(h.client, payload)
        vid = res.get_json()["id"]
        page = h.client.get(f"/v/{vid}").get_data(as_text=True)
        assert "custom_future_field" in page


def test_per_item_rows_are_written():
    with Harness() as h:
        post(h.client, meta(missing=["helmet", "safety-vest"]))
        page = h.client.get("/?item=safety-vest").get_data(as_text=True)
        assert "helmet + safety-vest" in page
        assert "1 violation" in page


# --------------------------------------------------------------------------
# gallery, detail, review
# --------------------------------------------------------------------------


def test_empty_gallery_has_an_empty_state():
    with Harness() as h:
        page = h.client.get("/").get_data(as_text=True)
        assert "No violations" in page


def test_gallery_lists_and_detail_renders():
    with Harness() as h:
        res = post(h.client)
        vid = res.get_json()["id"]

        gallery = h.client.get("/")
        assert gallery.status_code == 200
        assert "1 violation" in gallery.get_data(as_text=True)

        detail = h.client.get(f"/v/{vid}")
        assert detail.status_code == 200
        text = detail.get_data(as_text=True)
        assert "Missing: helmet" in text
        assert "missing positive" in text
        assert "bare head at 0.88" in text


def test_detail_overlays_the_bbox_as_a_percentage():
    """Image is stored raw; the box is CSS, computed from the stored coords."""
    with Harness() as h:
        vid = post(h.client).get_json()["id"]
        text = h.client.get(f"/v/{vid}").get_data(as_text=True)
        # bbox 250..390 of 640 wide -> left 39.062%, width 21.875%
        assert "left:39.062%" in text.replace(" ", "")
        assert "width:21.875%" in text.replace(" ", "")


def test_detail_404s_for_unknown_id():
    with Harness() as h:
        assert h.client.get("/v/9999").status_code == 404


def test_images_are_served():
    with Harness() as h:
        payload = meta(captured_at="2026-10-22T15:30:12+00:00")
        post(h.client, payload)
        rel = f"2026/10/22/{payload['event_id']}.jpg"
        full = h.client.get(f"/images/{rel}")
        assert full.status_code == 200
        assert full.data[:3] == b"\xff\xd8\xff"
        assert h.client.get(f"/thumbs/{rel}").status_code == 200


def test_image_path_traversal_is_blocked():
    with Harness() as h:
        for attempt in ("../instance/portal.sqlite3", "....//....//etc/passwd"):
            assert h.client.get(f"/images/{attempt}").status_code in (400, 403, 404)


def test_review_round_trip():
    with Harness() as h:
        vid = post(h.client).get_json()["id"]
        h.client.get(f"/v/{vid}")  # as a browser would, before submitting
        with h.client.session_transaction() as sess:
            token = sess["csrf_token"]
        res = h.client.post(
            f"/v/{vid}/review",
            data={"reviewed": "2", "notes": "mannequin, not a person",
                  "csrf_token": token},
            follow_redirects=True,
        )
        assert res.status_code == 200
        text = res.get_data(as_text=True)
        assert "mannequin, not a person" in text
        assert h.client.get("/api/v1/stats").get_json()["false_positive"] == 1


def test_review_requires_a_csrf_token():
    with Harness() as h:
        vid = post(h.client).get_json()["id"]
        res = h.client.post(f"/v/{vid}/review", data={"reviewed": "1"})
        assert res.status_code == 400


def test_review_rejects_an_out_of_range_value():
    with Harness() as h:
        vid = post(h.client).get_json()["id"]
        h.client.get(f"/v/{vid}")
        with h.client.session_transaction() as sess:
            token = sess["csrf_token"]
        res = h.client.post(
            f"/v/{vid}/review", data={"reviewed": "7", "csrf_token": token}
        )
        assert res.status_code == 400


# --------------------------------------------------------------------------
# filters, stats, export
# --------------------------------------------------------------------------


def test_filters_narrow_the_gallery():
    with Harness() as h:
        now = datetime.now(timezone.utc)
        post(h.client, meta(missing=["helmet"], captured_at=now.isoformat()))
        post(h.client, meta(
            missing=["safety-vest"],
            captured_at=(now - timedelta(days=5)).isoformat(),
            robot_id="freenove-02",
        ))

        assert "2 violations" in h.client.get("/").get_data(as_text=True)
        assert "1 violation" in h.client.get("/?item=helmet").get_data(as_text=True)
        assert "1 violation" in h.client.get(
            "/?robot=freenove-02"
        ).get_data(as_text=True)
        assert "2 violations" in h.client.get(
            "/?reviewed=0"
        ).get_data(as_text=True)
        assert "No violations" in h.client.get(
            "/?reviewed=1"
        ).get_data(as_text=True)

        yesterday = (now - timedelta(days=1)).date().isoformat()
        assert "1 violation" in h.client.get(
            f"/?from={yesterday}"
        ).get_data(as_text=True)


def test_date_to_filter_includes_the_whole_day():
    """A bare date must not silently exclude everything after midnight."""
    with Harness() as h:
        post(h.client, meta(captured_at="2026-10-22T15:30:00+00:00"))
        page = h.client.get("/?to=2026-10-22").get_data(as_text=True)
        assert "1 violation" in page


def test_stats_reports_precision_from_reviews():
    with Harness() as h:
        ids = [post(h.client).get_json()["id"] for _ in range(4)]
        h.client.get("/")
        with h.client.session_transaction() as sess:
            token = sess["csrf_token"]
        for vid, verdict in zip(ids, ["1", "1", "1", "2"]):
            h.client.post(
                f"/v/{vid}/review", data={"reviewed": verdict, "csrf_token": token}
            )
        s = h.client.get("/api/v1/stats").get_json()
        assert s["total"] == 4
        assert s["confirmed"] == 3
        assert s["false_positive"] == 1
        assert s["precision"] == 0.75
        assert s["by_item"]["helmet"] == 4

        page = h.client.get("/stats").get_data(as_text=True)
        assert "75.0%" in page


def test_stats_precision_is_none_before_any_review():
    with Harness() as h:
        post(h.client)
        assert h.client.get("/api/v1/stats").get_json()["precision"] is None
        assert "Nothing reviewed yet" in h.client.get("/stats").get_data(as_text=True)


def test_csv_export():
    with Harness() as h:
        post(h.client)
        res = h.client.get("/export.csv")
        assert res.status_code == 200
        assert "text/csv" in res.headers["Content-Type"]
        text = res.get_data(as_text=True)
        assert "event_id" in text.splitlines()[0]
        assert "freenove-01" in text


def test_robot_telemetry_is_tracked():
    with Harness() as h:
        post(h.client)
        post(h.client, meta(battery_v=7.1, cpu_temp_c=71.0))
        robots = h.client.get("/api/v1/stats").get_json()["robots"]
        assert len(robots) == 1
        assert robots[0]["total_events"] == 2
        assert robots[0]["last_battery_v"] == 7.1


def test_unsynced_clock_is_flagged():
    """A Pi with no RTC can report a nonsense timestamp; surface it."""
    with Harness() as h:
        post(h.client, meta(clock_synced=False))
        assert "clock?" in h.client.get("/").get_data(as_text=True)


def test_health_endpoint_needs_no_auth():
    """The robot probes this before uploading, with no session."""
    with Harness(with_auth=True) as h:
        res = h.client.get("/api/v1/health")
        assert res.status_code == 200
        assert res.get_json()["ok"] is True


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------


def test_gallery_requires_login_when_a_password_is_set():
    with Harness(with_auth=True) as h:
        assert h.client.get("/").status_code == 302
        assert h.client.get("/stats").status_code == 302
        h.login()
        assert h.client.get("/").status_code == 200


def test_wrong_password_does_not_authenticate():
    with Harness(with_auth=True) as h:
        h.client.post("/login", data={"password": "wrong"})
        assert h.client.get("/").status_code == 302


def test_ingest_works_without_a_session():
    """Auth protects browsing; the robot authenticates with the API key only."""
    with Harness(with_auth=True) as h:
        assert post(h.client).status_code == 201


def test_logout_clears_the_session():
    with Harness(with_auth=True) as h:
        h.login()
        assert h.client.get("/").status_code == 200
        with h.client.session_transaction() as sess:
            token = sess["csrf_token"]
        h.client.post("/logout", data={"csrf_token": token})
        assert h.client.get("/").status_code == 302


# --------------------------------------------------------------------------
# live proxy
# --------------------------------------------------------------------------


def test_live_proxy_reports_robot_offline_cleanly():
    """No robot on the LAN must give a clear 503, not a traceback."""
    with Harness() as h:
        h.app.config["ROBOT_MJPEG_URL"] = "http://127.0.0.1:9/stream.mjpg"
        res = h.client.get("/live/stream.mjpg")
        assert res.status_code == 503
        assert "unreachable" in res.get_data(as_text=True)


def test_live_page_renders():
    with Harness() as h:
        text = h.client.get("/live").get_data(as_text=True)
        assert "Live view" in text
        assert "Robot offline" in text  # the hidden fallback element


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
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    print("PPE portal tests\n")
    sys.exit(_run())
