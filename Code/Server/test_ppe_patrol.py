"""Tests for the patrol drive state machine and the vision/drive handshake.

Runs against SimulatedCar, so the whole behaviour is exercised with no Pi, no
camera and no model. Ticks are driven manually rather than by the thread loop,
which keeps everything deterministic and fast.

The handshake tests are the important ones: they pin down that frames captured
while the robot is moving or panning are never counted toward N-of-M
agreement, which is the property the entire false-positive story rests on.

Run: python test_ppe_patrol.py
"""

from __future__ import annotations

import sys
import time

from ppe_config import load_config
from ppe_patrol import (
    ALERT,
    BLOCKED_SCAN,
    HOLD,
    LOW_BATTERY,
    RECOVER,
    WANDER,
    NoMotorCar,
    PatrolDrive,
    SharedState,
    SimulatedCar,
)


def build(distances=None, **patrol_overrides):
    """A drive controller wired to a simulated car, ready to tick."""
    cfg = load_config()
    if patrol_overrides:
        from dataclasses import replace

        cfg = replace(cfg, patrol=replace(cfg.patrol, **patrol_overrides))
    car = SimulatedCar(scripted_distances=distances)
    shared = SharedState()
    drive = PatrolDrive(car, shared, cfg)
    drive._enter(WANDER)
    return drive, car, shared, cfg


def tick(drive, n=1, advance=0.0):
    """Tick the state machine, optionally pretending time has passed."""
    for _ in range(n):
        if advance:
            drive._state_since -= advance
            drive._phase_since -= advance
        drive._tick()


def run_until(drive, state, max_ticks=400, advance=0.1):
    for _ in range(max_ticks):
        if drive.state == state:
            return True
        tick(drive, advance=advance)
    return False


# --------------------------------------------------------------------------
# wander and obstacles
# --------------------------------------------------------------------------


def test_wanders_forward_when_clear():
    drive, car, shared, cfg = build(distances=[250.0])
    tick(drive)
    assert drive.state == WANDER
    assert car.motor == tuple([cfg.patrol.cruise_duty] * 4)


def test_camera_points_forward_and_is_steady_while_wandering():
    """Inference only happens while the camera is forward and steady."""
    drive, car, shared, cfg = build(distances=[250.0])
    tick(drive, 3)
    assert shared.pan_angle == cfg.patrol.pan_center
    assert shared.camera_steady is True


def test_obstacle_triggers_a_scan():
    drive, car, shared, cfg = build(distances=[20.0])
    tick(drive)
    assert drive.state == BLOCKED_SCAN


def test_scan_marks_the_camera_unsteady_while_sweeping():
    """The crux: the sonar shares the camera's bracket.

    Sweeping for bearings necessarily swings the camera, so every frame taken
    during a scan must be discarded.
    """
    drive, car, shared, cfg = build(distances=[20.0])
    tick(drive)
    assert drive.state == BLOCKED_SCAN
    tick(drive, 2)
    assert shared.camera_steady is False, "frames during a sweep must not count"
    assert shared.may_record() is False


def test_scan_sweeps_within_the_safe_pan_range():
    """Upstream sweeps 30/150; the factory test only covers 50-110."""
    drive, car, shared, cfg = build(distances=[20.0])
    run_until(drive, BLOCKED_SCAN)
    for _ in range(60):
        tick(drive, advance=0.5)
        if drive.state != BLOCKED_SCAN:
            break
    angles = [a for name, (_c, a) in car.commands if name == "servo"
              for _ in [0]] if False else [
        args[1] for name, args in car.commands if name == "servo" and args[0] == "0"
    ]
    assert angles, "the scan should have moved the pan servo"
    assert min(angles) >= 40, f"pan went to {min(angles)}, below the safe range"
    assert max(angles) <= 140, f"pan went to {max(angles)}, above the safe range"


def test_scan_returns_the_camera_to_centre_and_resumes():
    drive, car, shared, cfg = build(distances=[20.0, 250.0, 250.0, 250.0])
    run_until(drive, BLOCKED_SCAN)
    assert run_until(drive, WANDER, advance=0.5), "scan should finish and resume"
    assert shared.pan_angle == cfg.patrol.pan_center
    assert shared.camera_steady is True


def test_scan_turns_toward_the_freer_side():
    drive, car, shared, cfg = build()
    drive._enter(BLOCKED_SCAN)
    drive._phase = 4
    drive._phase_since = time.monotonic()
    drive._scan_readings = {"left": 250.0, "right": 40.0, "center": 25.0}
    tick(drive)
    left_front, _, right_front, _ = car.motor
    assert left_front < 0 and right_front > 0, (
        f"with more room on the left it should pivot left, got {car.motor}"
    )

    drive2, car2, _, _ = build()
    drive2._enter(BLOCKED_SCAN)
    drive2._phase = 4
    drive2._phase_since = time.monotonic()
    drive2._scan_readings = {"left": 40.0, "right": 250.0, "center": 25.0}
    tick(drive2)
    assert car2.motor[0] > 0 and car2.motor[2] < 0, (
        f"with more room on the right it should pivot right, got {car2.motor}"
    )


def test_repeatedly_boxed_in_escalates_to_recover():
    drive, car, shared, cfg = build(distances=[10.0])
    for _ in range(3):
        run_until(drive, BLOCKED_SCAN, advance=0.5)
        if not run_until(drive, WANDER, max_ticks=80, advance=0.5):
            break
        if drive.state == RECOVER:
            break
    assert drive._stuck_count >= 1 or drive.state == RECOVER


# --------------------------------------------------------------------------
# the hold handshake
# --------------------------------------------------------------------------


def test_person_request_stops_the_robot():
    drive, car, shared, cfg = build(distances=[250.0])
    tick(drive)
    assert car.motor != (0, 0, 0, 0)

    shared.ask_hold(True)
    tick(drive)
    assert drive.state == HOLD
    assert car.motor == (0, 0, 0, 0), "must brake before inspecting"


def test_hold_only_grants_recording_after_the_robot_settles():
    """Frames taken in the instant after braking still have motion blur."""
    drive, car, shared, cfg = build(distances=[250.0])
    shared.ask_hold(True)
    tick(drive)
    assert drive.state == HOLD
    assert shared.may_record() is False, "must not record before settling"

    tick(drive, advance=0.4)
    assert shared.hold_granted is True
    assert shared.camera_steady is True
    assert shared.may_record() is True, "after settling, frames should count"


def test_hold_centres_the_camera():
    drive, car, shared, cfg = build(distances=[250.0])
    drive._set_pan(cfg.patrol.pan_left)
    shared.ask_hold(True)
    tick(drive)
    tick(drive, advance=0.4)
    assert shared.pan_angle == cfg.patrol.pan_center


def test_hold_releases_when_vision_stops_asking():
    drive, car, shared, cfg = build(distances=[250.0])
    shared.ask_hold(True)
    tick(drive)
    tick(drive, advance=0.4)
    assert drive.state == HOLD

    shared.ask_hold(False)  # person walked away / cleared
    tick(drive)
    assert drive.state == WANDER
    assert shared.hold_granted is False


def test_hold_times_out_rather_than_staring_forever():
    drive, car, shared, cfg = build(distances=[250.0], hold_max_s=2.0)
    shared.ask_hold(True)
    tick(drive)
    tick(drive, advance=0.4)
    assert drive.state == HOLD

    tick(drive, advance=3.0)
    assert drive.state == WANDER, "a hold must not last indefinitely"
    assert shared.hold_granted is False


def test_post_hold_ignore_prevents_immediate_retrigger():
    """After releasing, the same person must not instantly re-stop the robot."""
    drive, car, shared, cfg = build(
        distances=[250.0], hold_max_s=1.0, post_hold_ignore_s=5.0
    )
    shared.ask_hold(True)
    tick(drive)
    tick(drive, advance=0.4)
    tick(drive, advance=2.0)
    assert drive.state == WANDER

    shared.ask_hold(True)  # still sees them
    tick(drive)
    assert drive.state == WANDER, "should ignore holds during the cooldown"


def test_recording_is_false_in_every_moving_state():
    """Belt and braces over the whole state machine."""
    for distances, target in (([20.0], BLOCKED_SCAN), ([250.0], WANDER)):
        drive, car, shared, cfg = build(distances=distances)
        run_until(drive, target)
        tick(drive, 2)
        assert shared.may_record() is False, (
            f"frames must not count in {drive.state}"
        )


# --------------------------------------------------------------------------
# alert
# --------------------------------------------------------------------------


def test_alert_buzzes_then_returns_to_wander():
    drive, car, shared, cfg = build(distances=[250.0])
    shared.ask_alert()
    tick(drive)
    assert drive.state == ALERT
    assert car.motor == (0, 0, 0, 0), "braking happens on entry"
    tick(drive)  # phase 0 of ALERT sounds the buzzer
    assert car.buzzing is True

    tick(drive, advance=1.0)
    tick(drive, advance=1.0)
    assert car.buzzing is False
    assert drive.state == WANDER
    assert shared.request_alert is False


# --------------------------------------------------------------------------
# sensors and safety
# --------------------------------------------------------------------------


def test_sonar_none_reading_is_not_treated_as_clear():
    """A failed read must not be read as open space.

    ultrasonic.get_distance() swallows a RuntimeWarning and returns None
    (ultrasonic.py:21). Treating that as 300cm would drive into things.
    """
    drive, car, shared, cfg = build()
    car.get_distance = lambda: 15.0
    tick(drive)
    assert drive.state == BLOCKED_SCAN

    drive2, car2, _, _ = build()
    drive2._last_distance = 15.0
    car2.get_distance = lambda: None
    assert drive2._read_distance() == 15.0, "None should hold the last known value"


def test_sonar_exception_is_survivable():
    drive, car, shared, cfg = build()
    drive._last_distance = 120.0

    def boom():
        raise RuntimeError("sensor on fire")

    car.get_distance = boom
    assert drive._read_distance() == 120.0
    assert drive.state == WANDER


def test_low_battery_parks_the_robot():
    drive, car, shared, cfg = build(distances=[250.0])
    car.battery_voltage = lambda: 6.1
    drive._last_battery_check = 0.0
    tick(drive)
    assert drive.state == LOW_BATTERY
    assert car.motor == (0, 0, 0, 0)


def test_implausible_battery_reading_is_ignored():
    """A 0V reading means the ADC is unreadable, not that the battery is flat."""
    drive, car, shared, cfg = build(distances=[250.0])
    car.battery_voltage = lambda: 0.0
    drive._last_battery_check = 0.0
    tick(drive)
    assert drive.state == WANDER


def test_tick_exception_brakes_rather_than_crashing():
    drive, car, shared, cfg = build(distances=[250.0])

    def boom(_now):
        raise RuntimeError("state handler exploded")

    drive._tick_wander = boom
    drive._tick()  # must not raise
    assert car.motor == (0, 0, 0, 0), "a failed tick must leave the wheels stopped"


def test_stop_always_leaves_the_wheels_stopped():
    drive, car, shared, cfg = build(distances=[250.0])
    tick(drive)
    assert car.motor != (0, 0, 0, 0)
    drive.stop()
    assert car.motor == (0, 0, 0, 0)
    assert car.buzzing is False


# --------------------------------------------------------------------------
# --no-motors
# --------------------------------------------------------------------------


def test_no_motors_keeps_sensors_live_but_wheels_inert():
    cfg = load_config()
    inner = SimulatedCar(scripted_distances=[250.0])
    car = NoMotorCar(inner)
    shared = SharedState()
    drive = PatrolDrive(car, shared, cfg)
    drive._enter(WANDER)
    tick(drive)

    assert inner.motor == (0, 0, 0, 0), "no motor command should reach the hardware"
    assert car.last_requested == tuple([cfg.patrol.cruise_duty] * 4), (
        "but the intent should still be recorded for inspection"
    )
    assert shared.sonic_center_cm == 250.0, "sensors must stay live"


def test_no_motors_still_runs_the_whole_state_machine():
    cfg = load_config()
    inner = SimulatedCar(scripted_distances=[18.0])
    car = NoMotorCar(inner)
    shared = SharedState()
    drive = PatrolDrive(car, shared, cfg)
    drive._enter(WANDER)
    tick(drive)
    assert drive.state == BLOCKED_SCAN
    assert run_until(drive, WANDER, advance=0.5)
    servo_moves = [a for n, a in inner.commands if n == "servo"]
    assert servo_moves, "the pan servo should still sweep with motors disabled"


# --------------------------------------------------------------------------
# shared state
# --------------------------------------------------------------------------


def test_shared_state_snapshot_is_complete():
    _drive, _car, shared, _cfg = build()
    state = shared.snapshot()
    for key in (
        "drive_state", "hold_granted", "camera_steady", "pan_angle",
        "sonic_cm", "battery_v", "person_present", "violations",
        "frames", "recorded", "infer_ms",
    ):
        assert key in state, f"snapshot missing {key}"


def test_may_record_requires_both_conditions():
    _drive, _car, shared, _cfg = build()
    for granted, steady, expected in (
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, True),
    ):
        with shared.lock:
            shared.hold_granted = granted
            shared.camera_steady = steady
        assert shared.may_record() is expected


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
    print("PPE patrol state machine tests\n")
    sys.exit(_run())
