"""Background uploader: drains the snapshot queue to the admin portal.

Runs as its own thread so a slow or absent portal never stalls detection or
driving. It shares nothing with the vision pipeline except the on-disk queue.

Offline resilience is the whole point. The portal lives on a PC that may be
switched off, asleep, or behind a firewall that drops the connection without a
word. In every one of those cases the robot must keep patrolling, keep
accumulating evidence, and backfill once the portal reappears.

Response handling, which is where this kind of thing usually goes wrong:

    201  created      -> mark sent
    409  duplicate    -> ALSO mark sent. This is the important one. If a POST
                         succeeds but the response is lost, the retry must not
                         create a second row, and the robot must not retry it
                         forever. The portal dedupes on our event_id and
                         answers 409, which we treat as success.
    429  rate limited -> keep, back off
    4xx  rejected     -> move to failed/. A malformed payload will never
                         succeed, so retrying it forever would block the queue.
    5xx / timeout /
    connection error  -> keep, exponential backoff, retry indefinitely.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from ppe_snapshot import PendingEvent, SnapshotStore

# A connection error means the PC is off or unreachable; a read timeout means
# it answered but is slow. Separate timeouts so the former fails fast.
DEFAULT_TIMEOUT = (3.0, 10.0)


class Uploader(threading.Thread):
    """Drains `store.pending` into the portal, forever, politely."""

    def __init__(
        self,
        store: SnapshotStore,
        ingest_url: str,
        health_url: str,
        api_key: str,
        timeout: tuple[float, float] = DEFAULT_TIMEOUT,
        backoff_max_s: float = 300.0,
        poll_interval_s: float = 2.0,
        dry_run: bool = False,
    ) -> None:
        super().__init__(name="ppe-uploader", daemon=True)
        self.store = store
        self.ingest_url = ingest_url
        self.health_url = health_url
        self.api_key = api_key
        self.timeout = timeout
        self.backoff_max_s = backoff_max_s
        self.poll_interval_s = poll_interval_s
        self.dry_run = dry_run

        self._stop_event = threading.Event()
        self._backoff = 0.0
        self.online: bool | None = None  # None until the first probe
        self.sent_count = 0
        self.failed_count = 0
        self.last_error: str | None = None

    # ------------------------------------------------------------------- loop

    def run(self) -> None:
        if self.dry_run:
            print("[uploader] dry_run: snapshots stay on disk, nothing is uploaded")
            return
        print(f"[uploader] started, target {self.ingest_url}")
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:  # never let the thread die
                self.last_error = f"{type(exc).__name__}: {exc}"
                print(f"[uploader] unexpected error: {self.last_error}")
                self._sleep(5.0)
        print(f"[uploader] stopped ({self.sent_count} sent, {self.failed_count} failed)")

    def _tick(self) -> None:
        events = list(self.store.iter_pending())
        if not events:
            self._backoff = 0.0
            self._sleep(self.poll_interval_s)
            return

        if not self.health_ok():
            wait = self._next_backoff()
            if self.online is not False:
                self.online = False
                print(
                    f"[uploader] portal unreachable ({self.last_error}); "
                    f"{len(events)} event(s) queued, retrying in {wait:.0f}s. "
                    f"Patrol continues regardless."
                )
            self._sleep(wait)
            return

        if self.online is not True:
            self.online = True
            print(f"[uploader] portal reachable; backfilling {len(events)} event(s)")
        self._backoff = 0.0

        for event in events:
            if self._stop_event.is_set():
                return
            ok, permanent = self.post_event(event)
            if ok:
                self.store.mark_sent(event)
                self.sent_count += 1
            elif permanent:
                self.store.mark_failed(event, self.last_error or "rejected")
                self.failed_count += 1
            else:
                # Transient: stop this pass and back off, preserving order.
                self._sleep(self._next_backoff())
                return

        self.store.prune()

    # ------------------------------------------------------------------- http

    def health_ok(self) -> bool:
        """Probe before uploading, so a dead portal costs one cheap GET."""
        try:
            import requests

            res = requests.get(self.health_url, timeout=self.timeout)
            if res.status_code == 200:
                return True
            self.last_error = f"health returned HTTP {res.status_code}"
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
        return False

    def post_event(self, event: PendingEvent) -> tuple[bool, bool]:
        """POST one event. Returns (delivered, permanently_rejected)."""
        import requests

        try:
            meta_bytes = event.meta.read_bytes()
            image_bytes = event.image.read_bytes()
        except FileNotFoundError:
            # Pruned or dropped underneath us; treat as done.
            return True, False

        try:
            res = requests.post(
                self.ingest_url,
                headers={"X-API-Key": self.api_key},
                files={
                    "metadata": ("meta.json", meta_bytes, "application/json"),
                    "image": (f"{event.event_id}.jpg", image_bytes, "image/jpeg"),
                },
                timeout=self.timeout,
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False, False

        if res.status_code in (200, 201):
            print(f"[uploader] sent {event.event_id[:12]} -> HTTP {res.status_code}")
            return True, False

        if res.status_code == 409:
            # Already stored. A retry after a lost ack lands here; treating it
            # as success is what stops the queue looping forever.
            print(f"[uploader] {event.event_id[:12]} already on the portal (409)")
            return True, False

        if res.status_code == 429:
            self.last_error = "portal rate limited us (429)"
            return False, False

        body = res.text[:200].replace("\n", " ")
        self.last_error = f"HTTP {res.status_code}: {body}"

        if res.status_code in (401, 403):
            # Not permanent in the useful sense: the key is probably
            # misconfigured, and discarding real evidence over a typo would be
            # worse than retrying. Keep it and complain.
            print(
                f"[uploader] auth rejected ({res.status_code}). Check PPE_API_KEY "
                f"matches the portal's. Keeping {event.event_id[:12]} queued."
            )
            return False, False

        if 400 <= res.status_code < 500:
            return False, True  # malformed; will never succeed

        return False, False  # 5xx: portal's problem, retry

    # ------------------------------------------------------------------ utils

    def _next_backoff(self) -> float:
        self._backoff = min(
            self.backoff_max_s, 1.0 if self._backoff <= 0 else self._backoff * 2
        )
        return self._backoff

    def _sleep(self, seconds: float) -> None:
        self._stop_event.wait(seconds)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout)

    def status(self) -> dict:
        return {
            "online": self.online,
            "sent": self.sent_count,
            "failed": self.failed_count,
            "backoff_s": round(self._backoff, 1),
            "last_error": self.last_error,
            **self.store.stats(),
        }


def from_config(cfg, store: SnapshotStore | None = None) -> Uploader:
    if store is None:
        store = SnapshotStore(
            cfg.resolve_path(cfg.snapshots.dir),
            jpeg_quality=cfg.snapshots.jpeg_quality,
            max_pending_mb=cfg.snapshots.max_pending_mb,
            sent_retention_days=cfg.snapshots.sent_retention_days,
            min_free_disk_mb=cfg.snapshots.min_free_disk_mb,
        )
    return Uploader(
        store=store,
        ingest_url=cfg.portal.ingest_url,
        health_url=cfg.portal.health_url,
        api_key=cfg.portal.resolved_api_key(),
        timeout=(cfg.portal.timeout_connect_s, cfg.portal.timeout_read_s),
        backoff_max_s=cfg.portal.backoff_max_s,
        poll_interval_s=cfg.portal.poll_interval_s,
        dry_run=cfg.dry_run,
    )


if __name__ == "__main__":
    import argparse

    from ppe_config import load_config

    ap = argparse.ArgumentParser(description="Drain the snapshot queue to the portal.")
    ap.add_argument("--config", help="config path")
    ap.add_argument("--url", help="override portal URL")
    ap.add_argument("--key", help="override API key")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--health", action="store_true", help="probe the portal and exit")
    ap.add_argument("--status", action="store_true", help="print queue stats and exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = SnapshotStore(
        cfg.resolve_path(cfg.snapshots.dir),
        jpeg_quality=cfg.snapshots.jpeg_quality,
    )

    base = (args.url or cfg.portal.url).rstrip("/")
    uploader = Uploader(
        store=store,
        ingest_url=f"{base}/api/v1/violations",
        health_url=f"{base}/api/v1/health",
        api_key=args.key or cfg.portal.resolved_api_key(),
        timeout=(cfg.portal.timeout_connect_s, cfg.portal.timeout_read_s),
        backoff_max_s=cfg.portal.backoff_max_s,
    )

    if args.status:
        for key, value in uploader.status().items():
            print(f"  {key:12s} {value}")
        raise SystemExit(0)

    if args.health:
        ok = uploader.health_ok()
        print(f"  portal {base}: {'reachable' if ok else 'UNREACHABLE'}")
        if not ok:
            print(f"  reason: {uploader.last_error}")
            print(
                "\n  If the portal is running, this is most often the Windows\n"
                "  firewall dropping inbound TCP on the Private profile. The\n"
                "  Flask dev server logs nothing when packets are silently\n"
                "  dropped, so it looks like a robot fault."
            )
        raise SystemExit(0 if ok else 1)

    if args.once:
        events = list(store.iter_pending())
        print(f"  {len(events)} pending event(s)")
        if events and not uploader.health_ok():
            print(f"  portal unreachable: {uploader.last_error}")
            raise SystemExit(1)
        for event in events:
            ok, permanent = uploader.post_event(event)
            if ok:
                store.mark_sent(event)
                print(f"  sent     {event.event_id[:12]}")
            elif permanent:
                store.mark_failed(event, uploader.last_error or "rejected")
                print(f"  rejected {event.event_id[:12]}: {uploader.last_error}")
            else:
                print(f"  deferred {event.event_id[:12]}: {uploader.last_error}")
        for key, value in store.stats().items():
            print(f"  {key:12s} {value}")
        raise SystemExit(0)

    uploader.start()
    print("  uploading continuously; Ctrl-C to stop")
    try:
        while True:
            time.sleep(10)
            print(f"  {uploader.status()}")
    except KeyboardInterrupt:
        uploader.stop()
