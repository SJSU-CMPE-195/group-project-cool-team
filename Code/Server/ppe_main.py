"""PPE patrol entry point.

Run this ON THE PI, from Code/Server (params.json and the model paths are
resolved relative to it):

    ~/ppe-venv/bin/python ppe_main.py                 # full patrol
    ~/ppe-venv/bin/python ppe_main.py --no-motors      # bench test on blocks
    ~/ppe-venv/bin/python ppe_main.py --simulate -v    # no hardware at all

On a PC, for developing the vision stack against a webcam:

    python ppe_main.py --simulate --source webcam -v

This is a SEPARATE entry point from upstream's main.py, deliberately. Stock
main.py keeps working untouched for manual driving and the PyQt client;
nothing in Code/Server/*.py from upstream is modified. Picamera2 is exclusive,
so run one or the other, never both at once.

Four threads, sharing one small lock-protected SharedState:

    vision    camera -> detect -> rules -> confirm -> snapshot queue
    drive     the patrol state machine (motors, pan servo, sonar)
    uploader  drains the on-disk queue to the portal, with backoff
    mjpeg     serves annotated frames for the portal's live view

The wheels stop on every exit path: clean shutdown, Ctrl-C, SIGTERM, an
unhandled exception, and a watchdog if the vision thread dies.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def preflight(cfg, args) -> list[str]:
    """Check the things that otherwise fail in confusing ways. Returns problems."""
    problems: list[str] = []

    if not args.simulate:
        # parameter.py prompts on stdin via input() when params.json is
        # missing or invalid (parameter.py:96-114), and BOTH Led() and ADC()
        # construct a ParameterManager. Under systemd or a detached shell that
        # is an invisible prompt and the process hangs forever.
        if not (Path.cwd() / "params.json").is_file():
            problems.append(
                "params.json is missing from the working directory.\n"
                "      parameter.py will block on an invisible stdin prompt.\n"
                "      Fix: run once interactively from Code/Server to create it,\n"
                "      or pass --simulate to skip hardware entirely."
            )

    for which, rel in (("confirm", cfg.model.confirm_onnx),):
        path = cfg.resolve_path(rel)
        if not path.is_file():
            problems.append(
                f"{which} model not found at {path}\n"
                f"      Export it on the PC, then commit it:\n"
                f"      python Tools/export_onnx.py --imgsz 640"
            )

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        problems.append(
            "onnxruntime is not installed.\n"
            "      Pi: ~/ppe-venv/bin/pip install onnxruntime\n"
            "      PC: pip install onnxruntime"
        )

    try:
        import numpy

        if not args.simulate and numpy.__version__.startswith("2."):
            print(
                f"[preflight] NOTE: numpy {numpy.__version__} is installed. "
                "Bookworm's picamera2 stack is built against numpy 1.x; if "
                "camera capture fails with ABI errors, that is why."
            )
    except ImportError:
        problems.append("numpy is not installed")

    if cfg.portal.api_key == "change-me" and not cfg.dry_run:
        print(
            "[preflight] NOTE: portal.api_key is still 'change-me'. Set it in "
            "ppe_config.local.json or via env:PPE_TOKEN."
        )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Autonomous PPE compliance patrol.")
    ap.add_argument("--config", help="config path (default: ppe_config.json)")
    ap.add_argument("--no-motors", action="store_true",
                    help="run everything but keep the wheels inert (bench test)")
    ap.add_argument("--simulate", action="store_true",
                    help="no hardware at all; simulated car")
    ap.add_argument("--source", default="auto",
                    choices=("auto", "picamera", "webcam", "file"),
                    help="camera backend")
    ap.add_argument("--index", type=int, default=0, help="webcam index")
    ap.add_argument("--path", help="video/image path for --source file")
    ap.add_argument("--no-upload", action="store_true",
                    help="queue snapshots locally but never upload")
    ap.add_argument("--no-live", action="store_true", help="disable the MJPEG server")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after N seconds (0 = run until interrupted)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="per-frame state machine trace")
    args = ap.parse_args()

    from ppe_config import load_config

    cfg = load_config(args.config)

    print("=" * 66)
    print("  PPE compliance patrol")
    print("=" * 66)
    print(f"  robot        {cfg.robot_id}  zone={cfg.zone_label}")
    print(f"  required PPE {cfg.rules.required_ppe}")
    print(f"  confirm      {cfg.temporal.positive_n}/{cfg.temporal.window_m} frames "
          f"(absence {cfg.temporal.absence_n}/{cfg.temporal.window_m})")
    print(f"  portal       {cfg.portal.url}")
    print(f"  snapshots    {cfg.resolve_path(cfg.snapshots.dir)}")
    if cfg.dry_run or args.no_upload:
        print("  UPLOADS OFF  snapshots stay on disk")
    print()

    problems = preflight(cfg, args)
    if problems:
        print("Cannot start:\n")
        for problem in problems:
            print(f"  [!] {problem}\n")
        return 1

    from ppe_camera import open_camera
    from ppe_mjpeg import FrameBuffer, MjpegServer
    from ppe_patrol import PatrolDrive, PatrolVision, SharedState, open_car
    from ppe_snapshot import SnapshotStore
    from ppe_uploader import from_config as uploader_from_config

    shared = SharedState()
    store = SnapshotStore(
        cfg.resolve_path(cfg.snapshots.dir),
        jpeg_quality=cfg.snapshots.jpeg_quality,
        max_pending_mb=cfg.snapshots.max_pending_mb,
        sent_retention_days=cfg.snapshots.sent_retention_days,
        min_free_disk_mb=cfg.snapshots.min_free_disk_mb,
    )
    pending = store.pending_count()
    if pending:
        print(f"  {pending} snapshot(s) already queued from a previous run\n")

    camera = open_camera(cfg, args.source, args.index, args.path)
    car = open_car(simulate=args.simulate, no_motors=args.no_motors)

    frame_buffer = None
    mjpeg = None
    if not args.no_live:
        frame_buffer = FrameBuffer()
        mjpeg = MjpegServer(frame_buffer, port=cfg.camera.mjpeg_port)

    vision = PatrolVision(
        camera, shared, cfg, store=store,
        frame_buffer=frame_buffer, verbose=args.verbose,
    )
    drive = PatrolDrive(car, shared, cfg)
    uploader = None
    if not (args.no_upload or cfg.dry_run):
        uploader = uploader_from_config(cfg, store)

    stopping = threading.Event()

    def shutdown(reason: str) -> None:
        if stopping.is_set():
            return
        stopping.set()
        print(f"\n[main] shutting down ({reason})")
        # Drive first: stopping the wheels is the only urgent part.
        drive.stop()
        vision.stop()
        if uploader is not None:
            uploader.stop()
        if mjpeg is not None:
            mjpeg.stop()
        camera.close()
        car.close()

    def on_signal(signum, _frame):
        shutdown(f"signal {signal.Signals(signum).name}")

    signal.signal(signal.SIGINT, on_signal)
    try:
        signal.signal(signal.SIGTERM, on_signal)
    except (AttributeError, ValueError):
        pass  # not available on every platform

    try:
        camera.start()
    except Exception as exc:
        print(f"[main] camera failed to start: {exc}")
        car.close()
        return 1

    started = time.monotonic()
    try:
        if mjpeg is not None:
            mjpeg.start()
        vision.start()

        # Wait for the detector to load before the wheels turn, so the robot
        # never drives blind. A failure here is fatal and reported.
        for _ in range(600):  # up to 60s for a cold ORT session load
            if vision.error or not vision.is_alive() or shared.snapshot()["frames"] > 0:
                break
            time.sleep(0.1)
        if vision.error:
            print(f"[main] vision thread failed: {vision.error}")
            shutdown("vision failed to start")
            return 1

        drive.start()
        if uploader is not None:
            uploader.start()

        print("\n[main] patrol running. Ctrl-C to stop.")
        if mjpeg is not None:
            print(f"[main] live view: http://<this-host>:{cfg.camera.mjpeg_port}/")
        print()

        last_report = 0.0
        while not stopping.is_set():
            time.sleep(0.5)

            # Watchdog: if vision dies, the robot is driving blind, so stop.
            if not vision.is_alive():
                print("[main] vision thread died; stopping for safety")
                shutdown("vision thread died")
                break
            if not drive.is_alive() and not stopping.is_set():
                print("[main] drive thread died; stopping")
                shutdown("drive thread died")
                break

            now = time.monotonic()
            if args.duration and now - started >= args.duration:
                shutdown(f"--duration {args.duration:.0f}s reached")
                break

            if not args.verbose and now - last_report >= 15.0:
                last_report = now
                state = shared.snapshot()
                queue = store.stats()
                online = (
                    "portal:online" if uploader and uploader.online
                    else "portal:offline" if uploader else "portal:off"
                )
                print(
                    f"  {state['drive_state']:<13} "
                    f"sonic={state['sonic_cm']:>5.0f}cm "
                    f"batt={state['battery_v']:.2f}V "
                    f"{state['infer_ms']:>4.0f}ms "
                    f"people={'Y' if state['person_present'] else 'n'} "
                    f"violations={state['violations']} "
                    f"queued={queue['pending']} {online}"
                )
    except Exception as exc:
        import traceback

        print(f"\n[main] unhandled error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        shutdown("unhandled exception")
        return 1
    finally:
        shutdown("exit")

    state = shared.snapshot()
    queue = store.stats()
    print("\n" + "=" * 66)
    print(f"  ran for            {time.monotonic() - started:.0f}s")
    print(f"  frames processed   {state['frames']} "
          f"({state['recorded']} counted toward confirmation)")
    print(f"  violations found   {state['violations']}")
    print(f"  queue             pending={queue['pending']} sent={queue['sent']} "
          f"failed={queue['failed']}")
    if uploader is not None:
        print(f"  uploaded           {uploader.sent_count}")
        if uploader.last_error:
            print(f"  last upload error  {uploader.last_error}")
    if queue["pending"]:
        print(
            f"\n  {queue['pending']} snapshot(s) still queued. They will upload on "
            f"the next run, or now with:\n    python ppe_uploader.py --once"
        )
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
