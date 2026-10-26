"""PPE compliance admin portal.

Runs on the PC (not the robot). The Pi POSTs confirmed violations here; you
browse them in a web page. Also proxies the robot's live MJPEG feed so one URL
gives you both the live view and the violation history.

    pip install -r Portal/requirements.txt
    set PPE_API_KEY=some-shared-secret
    set PPE_ADMIN_PASSWORD=something
    python Portal/app.py

SECURITY, stated plainly rather than implied:

The ingest endpoint requires a shared bearer token, compared in constant time.
Browsing requires one password, if PPE_ADMIN_PASSWORD is set; if it isn't, the
portal runs wide open and says so loudly at startup.

That is adequate for a trusted home or lab LAN and nothing more. Traffic is
plain HTTP, so the API key and the session cookie cross the network in
cleartext and anyone who can capture packets on that network gets both. There
is no per-robot key rotation and no transport integrity.

Do not port-forward this. The images are photographs of identifiable people.
If it ever needs to leave the LAN, put it behind a TLS-terminating proxy, or
better, a WireGuard/Tailscale tunnel so there is no public listener at all.
"""

from __future__ import annotations

import csv
import hmac
import io
import json
import os
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import (
    Flask,
    Response,
    abort,
    current_app,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

import db as dbmod

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
IMAGE_ROOT = DATA_DIR / "images"
THUMB_ROOT = DATA_DIR / "thumbs"
INSTANCE_DIR = HERE / "instance"
DB_PATH = INSTANCE_DIR / "portal.sqlite3"

PAGE_SIZE = 48
THUMB_WIDTH = 360
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

REQUIRED_META_KEYS = ("event_id", "captured_at", "missing_ppe")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = dbmod.connect(DB_PATH)
    return g.db


def _secret_key() -> bytes:
    """Persist the session key so logins survive a restart."""
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    path = INSTANCE_DIR / "secret_key"
    if not path.exists():
        path.write_bytes(secrets.token_bytes(32))
    return path.read_bytes()


def login_required(view):
    """Gate a view behind the admin password, when one is configured."""
    from functools import wraps

    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_app.config["ADMIN_PASSWORD_HASH"] and not session.get("authed"):
            return redirect(url_for("login", next=request.full_path))
        return view(*args, **kwargs)

    return wrapped


def check_csrf() -> None:
    sent = request.form.get("csrf_token", "")
    expected = session.get("csrf_token", "")
    if not expected or not hmac.compare_digest(sent, expected):
        abort(400, "CSRF token mismatch; reload the page and try again")


def is_jpeg(data: bytes) -> bool:
    """Verify the upload really is a JPEG.

    Magic bytes plus a real decode. Never trust the filename or the
    client-supplied content type -- a renamed .txt would otherwise be written
    into the image tree and served back to browsers.
    """
    if len(data) < 4 or data[:3] != b"\xff\xd8\xff":
        return False
    try:
        from PIL import Image

        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False


def write_thumbnail(src_bytes: bytes, dest: Path) -> bool:
    try:
        from PIL import Image

        dest.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(io.BytesIO(src_bytes))
        img = img.convert("RGB")
        ratio = THUMB_WIDTH / max(1, img.width)
        if ratio < 1:
            img = img.resize(
                (THUMB_WIDTH, max(1, round(img.height * ratio))), Image.LANCZOS
            )
        img.save(dest, "JPEG", quality=78, optimize=True)
        return True
    except Exception as exc:  # thumbnails are a nicety, not worth failing on
        print(f"[portal] thumbnail failed for {dest.name}: {exc}")
        return False


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=_secret_key(),
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        API_KEY=os.environ.get("PPE_API_KEY", "change-me"),
        ROBOT_MJPEG_URL=os.environ.get(
            "PPE_ROBOT_MJPEG", "http://192.168.129.51:8080/stream.mjpg"
        ),
    )

    admin_password = os.environ.get("PPE_ADMIN_PASSWORD", "")
    app.config["ADMIN_PASSWORD_HASH"] = (
        generate_password_hash(admin_password) if admin_password else ""
    )

    for folder in (IMAGE_ROOT, THUMB_ROOT, INSTANCE_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    dbmod.init_db(DB_PATH)

    if app.config["API_KEY"] == "change-me":
        print(
            "[portal] WARNING: PPE_API_KEY is unset, so the ingest endpoint "
            "accepts the default key 'change-me'."
        )
    if not app.config["ADMIN_PASSWORD_HASH"]:
        print(
            "[portal] WARNING: PPE_ADMIN_PASSWORD is unset, so anyone on this "
            "network can browse the violation gallery without logging in."
        )

    @app.teardown_appcontext
    def _close_db(_exc):
        conn = g.pop("db", None)
        if conn is not None:
            conn.close()

    @app.before_request
    def _ensure_csrf_token():
        """Establish the CSRF token per session, before any view runs.

        Doing this in the template context processor instead would mean the
        token only exists once a page has rendered -- so a form POST from a
        session that had not yet loaded a page would have nothing to compare
        against. A security token should not depend on a rendering side
        effect. Skipped for /api/, which authenticates by key and should not
        be handed a session cookie.
        """
        if request.path.startswith("/api/"):
            return
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_urlsafe(32)

    @app.context_processor
    def _inject():
        return {
            "csrf_token": session.get("csrf_token", ""),
            "review_labels": dbmod.REVIEW_LABELS,
            "auth_enabled": bool(app.config["ADMIN_PASSWORD_HASH"]),
            "authed": session.get("authed", False),
        }

    # ------------------------------------------------------------------ auth

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if not app.config["ADMIN_PASSWORD_HASH"]:
            return redirect(url_for("gallery"))
        error = None
        if request.method == "POST":
            if check_password_hash(
                app.config["ADMIN_PASSWORD_HASH"], request.form.get("password", "")
            ):
                session["authed"] = True
                session.permanent = True
                nxt = request.args.get("next") or url_for("gallery")
                return redirect(nxt)
            error = "Incorrect password."
        return render_template("login.html", error=error)

    @app.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        return redirect(url_for("login"))

    # --------------------------------------------------------------- gallery

    @app.route("/")
    @login_required
    def gallery():
        conn = get_db()
        page = max(1, request.args.get("page", 1, type=int))
        filters = {
            "item": request.args.get("item") or None,
            "robot": request.args.get("robot") or None,
            "date_from": request.args.get("from") or None,
            "date_to": request.args.get("to") or None,
            "reviewed": request.args.get("reviewed"),
        }
        total = dbmod.count_violations(conn, **filters)
        rows = dbmod.list_violations(
            conn, limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE, **filters
        )
        return render_template(
            "gallery.html",
            rows=rows,
            total=total,
            page=page,
            pages=max(1, -(-total // PAGE_SIZE)),
            filters=filters,
            all_items=dbmod.distinct_items(conn),
            all_robots=dbmod.distinct_robots(conn),
            summary=dbmod.stats(conn),
        )

    @app.route("/v/<int:violation_id>")
    @login_required
    def detail(violation_id: int):
        conn = get_db()
        row = dbmod.get_violation(conn, violation_id)
        if row is None:
            abort(404)
        newer, older = dbmod.neighbours(conn, violation_id)
        try:
            meta = json.loads(row["metadata_json"])
        except Exception:
            meta = {}
        return render_template(
            "detail.html",
            v=row,
            items=dbmod.get_items(conn, violation_id),
            meta=meta,
            meta_pretty=json.dumps(meta, indent=2, sort_keys=True),
            newer=newer,
            older=older,
        )

    @app.route("/v/<int:violation_id>/review", methods=["POST"])
    @login_required
    def review(violation_id: int):
        check_csrf()
        conn = get_db()
        if dbmod.get_violation(conn, violation_id) is None:
            abort(404)
        try:
            reviewed = int(request.form.get("reviewed", 0))
        except ValueError:
            abort(400, "reviewed must be 0, 1 or 2")
        if reviewed not in dbmod.REVIEW_LABELS:
            abort(400, "reviewed must be 0, 1 or 2")
        dbmod.set_review(conn, violation_id, reviewed, request.form.get("notes") or None)
        flash(f"Marked as {dbmod.REVIEW_LABELS[reviewed]}.")
        return redirect(request.form.get("next") or url_for("detail", violation_id=violation_id))

    @app.route("/stats")
    @login_required
    def stats_page():
        return render_template("stats.html", s=dbmod.stats(get_db()))

    @app.route("/live")
    @login_required
    def live_page():
        return render_template("live.html", robot_url=app.config["ROBOT_MJPEG_URL"])

    @app.route("/export.csv")
    @login_required
    def export_csv():
        conn = get_db()
        filters = {
            "item": request.args.get("item") or None,
            "robot": request.args.get("robot") or None,
            "date_from": request.args.get("from") or None,
            "date_to": request.args.get("to") or None,
            "reviewed": request.args.get("reviewed"),
        }
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(
            ["id", "event_id", "robot", "zone", "captured_at", "received_at",
             "missing_ppe", "person_conf", "battery_v", "cpu_temp_c",
             "review", "notes"]
        )
        for r in dbmod.export_rows(conn, **filters):
            writer.writerow(
                [r["id"], r["event_id"], r["robot_id"], r["zone_label"],
                 r["captured_at"], r["received_at"], r["missing_ppe"],
                 r["person_conf"], r["battery_v"], r["cpu_temp_c"],
                 dbmod.REVIEW_LABELS.get(r["reviewed"], r["reviewed"]), r["notes"]]
            )
        return Response(
            buf.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=ppe_violations.csv"},
        )

    # ---------------------------------------------------------------- images

    @app.route("/images/<path:rel>")
    @login_required
    def image(rel: str):
        # send_from_directory rejects traversal; don't hand-roll path joining.
        return send_from_directory(IMAGE_ROOT, rel, max_age=3600)

    @app.route("/thumbs/<path:rel>")
    @login_required
    def thumb(rel: str):
        return send_from_directory(THUMB_ROOT, rel, max_age=3600)

    # ------------------------------------------------------------- live feed

    @app.route("/live/stream.mjpg")
    @login_required
    def live_stream():
        """Proxy the robot's MJPEG feed.

        Keeps the portal as the single address to remember, and means the
        browser needs no route to the Pi. Streamed, so memory is bounded.
        """
        import requests

        url = app.config["ROBOT_MJPEG_URL"]
        try:
            upstream = requests.get(url, stream=True, timeout=(3, 10))
            upstream.raise_for_status()
        except Exception as exc:
            return Response(f"robot unreachable at {url}: {exc}", status=503,
                            mimetype="text/plain")

        content_type = upstream.headers.get(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )

        def pump():
            try:
                for chunk in upstream.iter_content(chunk_size=8192):
                    if chunk:
                        yield chunk
            except Exception:
                return
            finally:
                upstream.close()

        return Response(pump(), mimetype=content_type)

    # ------------------------------------------------------------------- api

    @app.route("/api/v1/health")
    def health():
        """The robot's uploader probes this before trying to POST."""
        conn = get_db()
        return jsonify(
            {
                "ok": True,
                "count": dbmod.count_violations(conn),
                "server_time": utc_now_iso(),
            }
        )

    @app.route("/api/v1/violations", methods=["POST"])
    def ingest():
        # 1. auth, constant time
        provided = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(provided, app.config["API_KEY"]):
            return jsonify({"error": "bad or missing X-API-Key"}), 401

        # 2. metadata
        raw_meta = request.form.get("metadata")
        if raw_meta is None and "metadata" in request.files:
            raw_meta = request.files["metadata"].read().decode("utf-8", "replace")
        if not raw_meta:
            return jsonify({"error": "missing 'metadata' part"}), 400
        try:
            meta = json.loads(raw_meta)
        except json.JSONDecodeError as exc:
            return jsonify({"error": f"metadata is not valid JSON: {exc}"}), 400
        if not isinstance(meta, dict):
            return jsonify({"error": "metadata must be a JSON object"}), 400
        if meta.get("schema", 1) != 1:
            return jsonify({"error": f"unsupported schema {meta.get('schema')}"}), 400
        missing_keys = [k for k in REQUIRED_META_KEYS if not meta.get(k)]
        if missing_keys:
            return jsonify({"error": f"metadata missing {missing_keys}"}), 400
        if not isinstance(meta.get("missing_ppe"), list) or not meta["missing_ppe"]:
            return jsonify({"error": "missing_ppe must be a non-empty list"}), 400

        event_id = str(meta["event_id"])[:64]

        # 3. idempotency. A POST that succeeded but whose response was lost
        #    will be retried; 409 lets the robot treat that as done instead of
        #    accumulating duplicates.
        conn = get_db()
        existing = dbmod.find_by_event_id(conn, event_id)
        if existing is not None:
            return (
                jsonify(
                    {
                        "status": "duplicate",
                        "id": existing["id"],
                        "event_id": event_id,
                    }
                ),
                409,
            )

        # 4. image
        upload = request.files.get("image")
        if upload is None:
            return jsonify({"error": "missing 'image' part"}), 400
        data = upload.read()
        if not data:
            return jsonify({"error": "image part is empty"}), 400
        if not is_jpeg(data):
            return jsonify({"error": "image is not a readable JPEG"}), 400

        # 5. write the file first, then the row. A crash in between leaves an
        #    orphan file (harmless, reapable) rather than a row pointing at
        #    nothing (a broken gallery tile).
        captured = str(meta["captured_at"])
        try:
            stamp = datetime.fromisoformat(captured.replace("Z", "+00:00"))
        except ValueError:
            stamp = datetime.now(timezone.utc)
        rel = f"{stamp:%Y/%m/%d}/{event_id or uuid.uuid4().hex}.jpg"

        dest = IMAGE_ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".jpg.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, dest)

        thumb_rel = rel if write_thumbnail(data, THUMB_ROOT / rel) else None

        try:
            violation_id = dbmod.insert_violation(
                conn,
                meta=meta,
                image_path=rel,
                thumb_path=thumb_rel,
                image_bytes=len(data),
                received_at=utc_now_iso(),
            )
        except sqlite3.IntegrityError:
            # Two concurrent POSTs of the same event; the other one won.
            existing = dbmod.find_by_event_id(conn, event_id)
            return (
                jsonify({"status": "duplicate",
                         "id": existing["id"] if existing else None,
                         "event_id": event_id}),
                409,
            )
        except Exception as exc:
            dest.unlink(missing_ok=True)
            (THUMB_ROOT / rel).unlink(missing_ok=True)
            return jsonify({"error": f"insert failed: {exc}"}), 500

        return (
            jsonify({"status": "created", "id": violation_id, "event_id": event_id}),
            201,
        )

    @app.route("/api/v1/violations", methods=["GET"])
    @login_required
    def api_list():
        conn = get_db()
        rows = dbmod.list_violations(
            conn,
            limit=min(500, request.args.get("limit", 100, type=int)),
            date_from=request.args.get("since") or None,
        )
        return jsonify(
            [
                {
                    "id": r["id"],
                    "event_id": r["event_id"],
                    "captured_at": r["captured_at"],
                    "robot_id": r["robot_id"],
                    "missing_ppe": r["missing_ppe"].split(","),
                    "reviewed": r["reviewed"],
                    "image": url_for("image", rel=r["image_path"]),
                }
                for r in rows
            ]
        )

    @app.route("/api/v1/stats")
    @login_required
    def api_stats():
        return jsonify(dbmod.stats(get_db()))

    # ---------------------------------------------------------------- errors

    @app.errorhandler(404)
    def _404(_e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "not found"}), 404
        return render_template("error.html", code=404,
                               message="No such page."), 404

    @app.errorhandler(413)
    def _413(_e):
        return jsonify(
            {"error": f"upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB"}
        ), 413

    return app


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="PPE admin portal")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    application = create_app()
    print(f"\n  PPE portal  ->  http://127.0.0.1:{args.port}")
    print(f"  ingest      ->  POST /api/v1/violations")
    print(f"  db          ->  {DB_PATH}")
    print(f"  images      ->  {IMAGE_ROOT}")
    if args.host == "0.0.0.0":
        print(
            "\n  Reachable from the LAN. On Windows you may need to allow "
            "inbound TCP\n  on this port for the Private profile, or the Pi's "
            "uploads will be\n  silently dropped and look like a robot fault."
        )
    print()
    application.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
