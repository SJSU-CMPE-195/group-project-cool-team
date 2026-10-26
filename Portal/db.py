"""SQLite storage for the PPE admin portal.

Images live on disk; the database holds metadata and a relative path. Putting
JPEGs in BLOBs would bloat the file, make VACUUM and backups painful, and
defeat the OS page cache -- and `send_from_directory` handles conditional GETs
for free.

Concurrency: one writer (the robot, via the ingest endpoint) and a few readers
(browsers). WAL mode plus a busy timeout covers that comfortably. Connections
are per-request and cached on `flask.g`, never shared between threads.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS violations (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id       TEXT    NOT NULL UNIQUE,   -- robot-generated idempotency key
  robot_id       TEXT    NOT NULL,
  zone_label     TEXT,
  captured_at    TEXT    NOT NULL,          -- ISO8601 UTC, from the robot
  received_at    TEXT    NOT NULL,          -- ISO8601 UTC, from the portal
  clock_synced   INTEGER NOT NULL DEFAULT 1,
  missing_ppe    TEXT    NOT NULL,          -- "helmet" or "helmet,safety-vest"
  person_conf    REAL,
  bbox_x1 INTEGER, bbox_y1 INTEGER, bbox_x2 INTEGER, bbox_y2 INTEGER,
  frame_w INTEGER, frame_h INTEGER,
  battery_v      REAL,
  cpu_temp_c     REAL,
  image_path     TEXT    NOT NULL,          -- relative to data/images
  thumb_path     TEXT,                      -- relative to data/thumbs
  image_bytes    INTEGER,
  metadata_json  TEXT    NOT NULL,          -- full payload, verbatim
  reviewed       INTEGER NOT NULL DEFAULT 0, -- 0 new, 1 confirmed, 2 false pos
  notes          TEXT
);
CREATE INDEX IF NOT EXISTS idx_v_captured ON violations(captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_v_reviewed ON violations(reviewed);
CREATE INDEX IF NOT EXISTS idx_v_robot    ON violations(robot_id, captured_at DESC);

-- Normalized so filtering by item is an index lookup rather than a LIKE scan
-- over the denormalized missing_ppe column.
CREATE TABLE IF NOT EXISTS violation_items (
  violation_id INTEGER NOT NULL REFERENCES violations(id) ON DELETE CASCADE,
  item         TEXT    NOT NULL,
  status       TEXT    NOT NULL,            -- MISSING_POSITIVE | MISSING_ABSENT
  confidence   REAL,
  frames_agree TEXT,
  reason       TEXT,
  PRIMARY KEY (violation_id, item)
);
CREATE INDEX IF NOT EXISTS idx_items_item ON violation_items(item);

CREATE TABLE IF NOT EXISTS robots (
  robot_id       TEXT PRIMARY KEY,
  last_seen      TEXT,
  last_battery_v REAL,
  last_cpu_temp_c REAL,
  total_events   INTEGER NOT NULL DEFAULT 0
);
"""

REVIEW_LABELS = {0: "New", 1: "Confirmed", 2: "False positive"}


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str | Path) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------
# writes
# --------------------------------------------------------------------------


def find_by_event_id(conn: sqlite3.Connection, event_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT id, event_id, image_path FROM violations WHERE event_id = ?",
        (event_id,),
    ).fetchone()


def insert_violation(
    conn: sqlite3.Connection,
    *,
    meta: dict[str, Any],
    image_path: str,
    thumb_path: str | None,
    image_bytes: int,
    received_at: str,
) -> int:
    """Insert a violation and its per-item rows in one transaction.

    Raises sqlite3.IntegrityError if event_id already exists; the caller turns
    that into a 409, which the robot treats as success.
    """
    bbox = meta.get("person_bbox") or [None, None, None, None]
    frame = meta.get("frame_size") or [None, None]
    missing = meta.get("missing_ppe") or []
    evidence = meta.get("evidence") or {}

    with conn:  # transaction: both tables or neither
        cur = conn.execute(
            """
            INSERT INTO violations (
              event_id, robot_id, zone_label, captured_at, received_at,
              clock_synced, missing_ppe, person_conf,
              bbox_x1, bbox_y1, bbox_x2, bbox_y2, frame_w, frame_h,
              battery_v, cpu_temp_c, image_path, thumb_path, image_bytes,
              metadata_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                meta["event_id"],
                meta.get("robot_id", "unknown"),
                meta.get("zone_label"),
                meta["captured_at"],
                received_at,
                1 if meta.get("clock_synced", True) else 0,
                ",".join(missing),
                meta.get("person_conf"),
                _int(bbox[0]), _int(bbox[1]), _int(bbox[2]), _int(bbox[3]),
                _int(frame[0]), _int(frame[1]),
                meta.get("battery_v"),
                meta.get("cpu_temp_c"),
                image_path,
                thumb_path,
                image_bytes,
                json.dumps(meta, separators=(",", ":")),
            ),
        )
        violation_id = int(cur.lastrowid)

        for item in missing:
            ev = evidence.get(item) or {}
            support = ev.get("support") or {}
            conn.execute(
                """
                INSERT OR REPLACE INTO violation_items
                  (violation_id, item, status, confidence, frames_agree, reason)
                VALUES (?,?,?,?,?,?)
                """,
                (
                    violation_id,
                    item,
                    ev.get("status", "MISSING_ABSENT"),
                    max(support.values()) if support else None,
                    ev.get("frames_agree"),
                    ev.get("reason"),
                ),
            )

        conn.execute(
            """
            INSERT INTO robots (robot_id, last_seen, last_battery_v,
                                last_cpu_temp_c, total_events)
            VALUES (?,?,?,?,1)
            ON CONFLICT(robot_id) DO UPDATE SET
              last_seen       = excluded.last_seen,
              last_battery_v  = excluded.last_battery_v,
              last_cpu_temp_c = excluded.last_cpu_temp_c,
              total_events    = total_events + 1
            """,
            (
                meta.get("robot_id", "unknown"),
                received_at,
                meta.get("battery_v"),
                meta.get("cpu_temp_c"),
            ),
        )
    return violation_id


def set_review(
    conn: sqlite3.Connection, violation_id: int, reviewed: int, notes: str | None
) -> None:
    with conn:
        conn.execute(
            "UPDATE violations SET reviewed = ?, notes = ? WHERE id = ?",
            (reviewed, notes, violation_id),
        )


def delete_violation(conn: sqlite3.Connection, violation_id: int) -> sqlite3.Row | None:
    row = conn.execute(
        "SELECT image_path, thumb_path FROM violations WHERE id = ?", (violation_id,)
    ).fetchone()
    if row is None:
        return None
    with conn:
        conn.execute("DELETE FROM violations WHERE id = ?", (violation_id,))
    return row


# --------------------------------------------------------------------------
# reads
# --------------------------------------------------------------------------


def _int(value: Any) -> int | None:
    return None if value is None else int(round(float(value)))


def build_filters(
    item: str | None = None,
    robot: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    reviewed: str | None = None,
) -> tuple[str, list[Any]]:
    """Shared WHERE clause so the gallery, CSV export and counts agree."""
    where: list[str] = []
    params: list[Any] = []

    if item:
        where.append(
            "v.id IN (SELECT violation_id FROM violation_items WHERE item = ?)"
        )
        params.append(item)
    if robot:
        where.append("v.robot_id = ?")
        params.append(robot)
    if date_from:
        where.append("v.captured_at >= ?")
        params.append(date_from)
    if date_to:
        # Inclusive of the whole day when given a bare date.
        where.append("v.captured_at <= ?")
        params.append(date_to if "T" in date_to else date_to + "T23:59:59.999999+00:00")
    if reviewed not in (None, "", "all"):
        where.append("v.reviewed = ?")
        params.append(int(reviewed))

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    return clause, params


def list_violations(
    conn: sqlite3.Connection,
    *,
    limit: int = 48,
    offset: int = 0,
    **filters: Any,
) -> list[sqlite3.Row]:
    clause, params = build_filters(**filters)
    return conn.execute(
        f"""
        SELECT v.* FROM violations v
        {clause}
        ORDER BY v.captured_at DESC, v.id DESC
        LIMIT ? OFFSET ?
        """,
        [*params, limit, offset],
    ).fetchall()


def count_violations(conn: sqlite3.Connection, **filters: Any) -> int:
    clause, params = build_filters(**filters)
    row = conn.execute(
        f"SELECT COUNT(*) AS n FROM violations v {clause}", params
    ).fetchone()
    return int(row["n"])


def get_violation(conn: sqlite3.Connection, violation_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM violations WHERE id = ?", (violation_id,)
    ).fetchone()


def get_items(conn: sqlite3.Connection, violation_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM violation_items WHERE violation_id = ? ORDER BY item",
        (violation_id,),
    ).fetchall()


def neighbours(
    conn: sqlite3.Connection, violation_id: int
) -> tuple[int | None, int | None]:
    """(newer_id, older_id) for prev/next links on the detail page."""
    row = get_violation(conn, violation_id)
    if row is None:
        return None, None
    newer = conn.execute(
        """SELECT id FROM violations
           WHERE (captured_at, id) > (?, ?)
           ORDER BY captured_at ASC, id ASC LIMIT 1""",
        (row["captured_at"], violation_id),
    ).fetchone()
    older = conn.execute(
        """SELECT id FROM violations
           WHERE (captured_at, id) < (?, ?)
           ORDER BY captured_at DESC, id DESC LIMIT 1""",
        (row["captured_at"], violation_id),
    ).fetchone()
    return (newer["id"] if newer else None, older["id"] if older else None)


def distinct_robots(conn: sqlite3.Connection) -> list[str]:
    return [r["robot_id"] for r in conn.execute(
        "SELECT robot_id FROM robots ORDER BY robot_id"
    )]


def distinct_items(conn: sqlite3.Connection) -> list[str]:
    return [r["item"] for r in conn.execute(
        "SELECT DISTINCT item FROM violation_items ORDER BY item"
    )]


def stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Headline numbers, including a review-derived precision estimate.

    Precision here is confirmed / (confirmed + false positive) over reviewed
    events only. It is the number to look at before tightening thresholds or
    before enabling an absence-only item like safety-vest.
    """
    total = int(conn.execute("SELECT COUNT(*) n FROM violations").fetchone()["n"])
    by_review = {
        int(r["reviewed"]): int(r["n"])
        for r in conn.execute(
            "SELECT reviewed, COUNT(*) n FROM violations GROUP BY reviewed"
        )
    }
    by_item = {
        r["item"]: int(r["n"])
        for r in conn.execute(
            "SELECT item, COUNT(*) n FROM violation_items GROUP BY item ORDER BY n DESC"
        )
    }
    by_day = [
        {"day": r["day"], "n": int(r["n"])}
        for r in conn.execute(
            """SELECT substr(captured_at, 1, 10) day, COUNT(*) n
               FROM violations GROUP BY day ORDER BY day DESC LIMIT 30"""
        )
    ]
    confirmed = by_review.get(1, 0)
    false_pos = by_review.get(2, 0)
    judged = confirmed + false_pos
    return {
        "total": total,
        "new": by_review.get(0, 0),
        "confirmed": confirmed,
        "false_positive": false_pos,
        "reviewed_total": judged,
        "precision": round(confirmed / judged, 3) if judged else None,
        "by_item": by_item,
        "by_day": by_day,
        "robots": [dict(r) for r in conn.execute("SELECT * FROM robots")],
    }


def export_rows(conn: sqlite3.Connection, **filters: Any) -> Iterable[sqlite3.Row]:
    clause, params = build_filters(**filters)
    return conn.execute(
        f"""SELECT v.id, v.event_id, v.robot_id, v.zone_label, v.captured_at,
                   v.received_at, v.missing_ppe, v.person_conf, v.battery_v,
                   v.cpu_temp_c, v.reviewed, v.notes
            FROM violations v {clause}
            ORDER BY v.captured_at DESC""",
        params,
    )
