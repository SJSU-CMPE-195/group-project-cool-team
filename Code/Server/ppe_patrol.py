"""Patrol behaviour: the vision thread, the drive state machine, and the
handshake between them.

The central problem this module solves is that **the ultrasonic sensor and the
camera are bolted to the same pan/tilt bracket.** They physically cannot look
in different directions at the same time. Upstream's obstacle-avoidance mode
sweeps the pan servo across three bearings every 0.2s (car.py:69-88), which
would swing the camera continuously -- and inference on a camera that is
panning across a room produces garbage.

That is not an arbitration problem, it is a scheduling one, and it is resolved
by making the two activities take turns:

    sweeping for obstacle bearings happens ONLY while stopped,
    with camera_steady false and every frame discarded.

Which leads to the second reason the robot stops: temporal confirmation is
only meaningful across frames of the same scene. N-of-M agreement over frames
taken while driving compares different views and means nothing. So the vision
thread *requests a hold* and the drive thread *grants* it:

    vision sees a person          -> request_hold = True
    drive brakes, waits to settle -> camera_steady = True, hold_granted = True
    vision accumulates N-of-M     -> only while hold_granted and camera_steady
    violation confirmed           -> snapshot, request_alert = True
    drive releases after hold_max_s

Stopping to look is also what removes motion blur, which is the dominant
cause of both missed detections and false positives on a robot driving over a
hard floor. One mechanism, three benefits.

A SimulatedCar stands in for the hardware, so the entire state machine runs
and is testable on a PC with no Pi attached.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

# Drive states. Plain strings rather than an enum so they land in JSON
# metadata and log lines without conversion.
WANDER = "WANDER"
BLOCKED_SCAN = "BLOCKED_SCAN"
HOLD = "HOLD"
ALERT = "ALERT"
RECOVER = "RECOVER"
LOW_BATTERY = "LOW_BATTERY"
STOPPED = "STOPPED"

SONIC_CLEAR_CM = 300.0  # gpiozero reports max distance when nothing echoes


# --------------------------------------------------------------------------
# shared state
# --------------------------------------------------------------------------


@dataclass
class SharedState:
    """The vision/drive handshake. Every field is touched under `lock`.

    Kept small and dumb on purpose: two threads sharing a lot of mutable state
    is how robots acquire unreproducible bugs.
    """

    lock: threading.Lock = field(default_factory=threading.Lock)

    # vision -> drive
    request_hold: bool = False
    request_alert: bool = False

    # drive -> vision
    hold_granted: bool = False
    camera_steady: bool = True
    drive_state: str = STOPPED
    pan_angle: int = 90
    sonic_center_cm: float = SONIC_CLEAR_CM
    battery_v: float = 0.0
    motors_enabled: bool = True

    # telemetry, for logs and the live view
    person_present: bool = False
    largest_person_frac: float = 0.0
    last_violation_at: float = 0.0
    violations_total: int = 0
    frames_total: int = 0
    frames_recorded: int = 0
    infer_ms: float = 0.0

    # ------------------------------------------------------------- accessors

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "drive_state": self.drive_state,
                "hold_granted": self.hold_granted,
                "camera_steady": self.camera_steady,
                "request_hold": self.request_hold,
                "pan_angle": self.pan_angle,
                "sonic_cm": self.sonic_center_cm,
                "battery_v": self.battery_v,
                "person_present": self.person_present,
                "person_frac": round(self.largest_person_frac, 3),
                "violations": self.violations_total,
                "frames": self.frames_total,
                "recorded": self.frames_recorded,
                "infer_ms": round(self.infer_ms, 1),
            }

    def may_record(self) -> bool:
        """Whether the current frame counts toward N-of-M agreement.

        The single most important predicate in the project: it is what keeps
        temporal confirmation honest.
        """
        with self.lock:
            return self.hold_granted and self.camera_steady

    def set_person(self, present: bool, frac: float) -> None:
        with self.lock:
            self.person_present = present
            self.largest_person_frac = frac

    def ask_hold(self, want: bool) -> None:
        with self.lock:
            self.request_hold = want

    def ask_alert(self) -> None:
        with self.lock:
            self.request_alert = True


# --------------------------------------------------------------------------
# hardware
# --------------------------------------------------------------------------


class CarInterface(Protocol):
    def set_motor_model(self, d1: int, d2: int, d3: int, d4: int) -> None: ...
    def set_servo_pwm(self, channel: str, angle: int) -> None: ...
    def get_distance(self) -> float | None: ...
    def battery_voltage(self) -> float: ...
    def buzz(self, on: bool) -> None: ...
    def close(self) -> None: ...


class RealCar:
    """Adapter over upstream's `Car`, which owns all the device objects.

    Nothing in car.py is modified. In particular `mode_ultrasonic` is never
    called -- its 0.2s servo sweep is incompatible with inference -- but its
    tuned duty values are reused in the state machine below.
    """

    def __init__(self) -> None:
        from car import Car

        self.car = Car()
        self._pcb_version = 2
        try:
            from parameter import ParameterManager

            self._pcb_version = ParameterManager().get_pcb_version()
        except Exception as exc:
            print(f"[patrol] could not read Pcb_Version ({exc}); assuming 2")
        self._buzzer = None
        try:
            from buzzer import Buzzer

            self._buzzer = Buzzer()
        except Exception as exc:
            print(f"[patrol] buzzer unavailable: {exc}")

    def set_motor_model(self, d1: int, d2: int, d3: int, d4: int) -> None:
        self.car.motor.set_motor_model(d1, d2, d3, d4)

    def set_servo_pwm(self, channel: str, angle: int) -> None:
        # servo.py does not clamp, and a servo driven past its stop stalls.
        self.car.servo.set_servo_pwm(str(channel), int(max(20, min(160, angle))))

    def get_distance(self) -> float | None:
        return self.car.sonic.get_distance()

    def battery_voltage(self) -> float:
        # Multiplier depends on Pcb_Version, not Pi_Version (main.py:124).
        raw = self.car.adc.read_adc(2)
        return float(raw) * (3.0 if self._pcb_version == 1 else 2.0)

    def buzz(self, on: bool) -> None:
        if self._buzzer is not None:
            self._buzzer.set_state(bool(on))

    def close(self) -> None:
        try:
            self.car.close()
        finally:
            if self._buzzer is not None:
                try:
                    self._buzzer.set_state(False)
                    self._buzzer.close()
                except Exception:
                    pass


class SimulatedCar:
    """Stands in for the hardware so the state machine runs anywhere.

    Models the two sensor quirks that matter: `get_distance()` can return None,
    and "nothing there" reads as 300.0 rather than a large-but-finite number.
    """

    def __init__(self, scripted_distances: list[float] | None = None,
                 verbose: bool = False) -> None:
        self.verbose = verbose
        self.motor = (0, 0, 0, 0)
        self.servo_angles: dict[str, int] = {"0": 90, "1": 90}
        self.buzzing = False
        self.closed = False
        self._script = list(scripted_distances or [])
        self._script_pos = 0
        self.commands: list[tuple[str, tuple]] = []

    def _log(self, name: str, args: tuple) -> None:
        self.commands.append((name, args))
        if self.verbose:
            print(f"    [sim] {name}{args}")

    def set_motor_model(self, d1: int, d2: int, d3: int, d4: int) -> None:
        self.motor = (d1, d2, d3, d4)
        self._log("motor", (d1, d2, d3, d4))

    def set_servo_pwm(self, channel: str, angle: int) -> None:
        self.servo_angles[str(channel)] = int(angle)
        self._log("servo", (channel, angle))

    def get_distance(self) -> float | None:
        if self._script:
            value = self._script[self._script_pos % len(self._script)]
            self._script_pos += 1
            return value
        return round(random.uniform(45.0, SONIC_CLEAR_CM), 1)

    def battery_voltage(self) -> float:
        return 7.9

    def buzz(self, on: bool) -> None:
        self.buzzing = bool(on)
        self._log("buzz", (on,))

    def close(self) -> None:
        self.set_motor_model(0, 0, 0, 0)
        self.closed = True


class NoMotorCar:
    """Wraps a car but swallows motor commands, for bench testing on blocks.

    Servo, ultrasonic, buzzer and battery stay live, so the full state machine
    and the whole sensor path are exercised with the wheels inert.
    """

    def __init__(self, inner: CarInterface) -> None:
        self.inner = inner
        self.last_requested = (0, 0, 0, 0)

    def set_motor_model(self, d1: int, d2: int, d3: int, d4: int) -> None:
        self.last_requested = (d1, d2, d3, d4)  # recorded, never sent

    def set_servo_pwm(self, channel: str, angle: int) -> None:
        self.inner.set_servo_pwm(channel, angle)

    def get_distance(self) -> float | None:
        return self.inner.get_distance()

    def battery_voltage(self) -> float:
        return self.inner.battery_voltage()

    def buzz(self, on: bool) -> None:
        self.inner.buzz(on)

    def close(self) -> None:
        self.inner.close()


def open_car(simulate: bool = False, no_motors: bool = False) -> CarInterface:
    if simulate:
        print("[patrol] using a simulated car (no hardware)")
        return SimulatedCar()
    try:
        car: CarInterface = RealCar()
    except Exception as exc:
        print(f"[patrol] hardware unavailable ({exc}); falling back to simulation")
        return SimulatedCar()
    if no_motors:
        print("[patrol] --no-motors: wheels disabled, sensors live")
        return NoMotorCar(car)
    return car


# --------------------------------------------------------------------------
# drive state machine
# --------------------------------------------------------------------------


class PatrolDrive(threading.Thread):
    """Non-blocking tick-based drive loop.

    Uses its own timer, never `Car.car_record_time` (car.py:16) -- that field
    is shared across every upstream mode, which is precisely why they cannot
    be interleaved.
    """

    def __init__(self, car: CarInterface, shared: SharedState, cfg) -> None:
        super().__init__(name="ppe-drive", daemon=True)
        self.car = car
        self.shared = shared
        self.cfg = cfg
        self.p = cfg.patrol
        self._stop_event = threading.Event()

        self.state = WANDER
        self._state_since = time.monotonic()
        self._phase = 0
        self._phase_since = time.monotonic()
        self._ignore_holds_until = 0.0
        self._stuck_count = 0
        self._last_distance = SONIC_CLEAR_CM
        self._scan_readings: dict[str, float] = {}
        self._last_battery_check = 0.0

    # ------------------------------------------------------------------- loop

    def run(self) -> None:
        print("[drive] patrol started")
        try:
            self._set_pan(self.p.pan_center)
            self.car.set_servo_pwm("1", self.p.tilt)
            while not self._stop_event.is_set():
                self._tick()
                self._stop_event.wait(self.p.tick_s)
        finally:
            # Always leave the wheels stopped, whatever happened.
            self._brake()
            print(f"[drive] stopped in state {self.state}")

    def _tick(self) -> None:
        """One non-blocking step. Never raises: a failed tick brakes instead.

        The guard lives here rather than in run(), so that stopping the wheels
        on an error is a property of the state machine itself and holds for
        every caller, including tests and any future scheduler.
        """
        try:
            now = time.monotonic()
            self._check_battery(now)

            handler = {
                WANDER: self._tick_wander,
                BLOCKED_SCAN: self._tick_blocked_scan,
                HOLD: self._tick_hold,
                ALERT: self._tick_alert,
                RECOVER: self._tick_recover,
                LOW_BATTERY: self._tick_low_battery,
            }.get(self.state)
            if handler is not None:
                handler(now)
        except Exception as exc:
            print(f"[drive] tick failed: {type(exc).__name__}: {exc}")
            try:
                self._brake()
            except Exception:
                pass

    # ----------------------------------------------------------------- states

    def _tick_wander(self, now: float) -> None:
        """Cruise forward, camera fixed ahead and steady for inference."""
        with self.shared.lock:
            wants_hold = self.shared.request_hold
            alert = self.shared.request_alert

        if alert:
            self._enter(ALERT)
            return
        if wants_hold and now >= self._ignore_holds_until:
            self._enter(HOLD)
            return

        distance = self._read_distance()
        if distance < self.p.stop_distance_cm:
            self._enter(BLOCKED_SCAN)
            return

        self.car.set_motor_model(*([self.p.cruise_duty] * 4))

    def _tick_blocked_scan(self, now: float) -> None:
        """Stopped: sweep the bracket for bearings, then turn to the freer side.

        The camera swings with the sensor, so camera_steady is false
        throughout and the vision thread drops every frame taken here.
        """
        elapsed = now - self._phase_since

        if self._phase == 0:  # brake and back off
            self._brake()
            self._set_steady(False)
            self.car.set_motor_model(*([self.p.reverse_duty] * 4))
            if elapsed > 0.4:
                self._brake()
                self._next_phase()

        elif self._phase == 1:  # look left
            self._set_pan(self.p.pan_left)
            if elapsed > self.p.pan_settle_s:
                self._scan_readings["left"] = self._read_distance(update_shared=False)
                self._next_phase()

        elif self._phase == 2:  # look right
            self._set_pan(self.p.pan_right)
            if elapsed > self.p.pan_settle_s:
                self._scan_readings["right"] = self._read_distance(update_shared=False)
                self._next_phase()

        elif self._phase == 3:  # recentre before turning
            self._set_pan(self.p.pan_center)
            if elapsed > self.p.pan_settle_s:
                self._scan_readings["center"] = self._read_distance()
                self._next_phase()

        elif self._phase == 4:  # pivot toward open space
            left = self._scan_readings.get("left", SONIC_CLEAR_CM)
            right = self._scan_readings.get("right", SONIC_CLEAR_CM)
            duty = self.p.pivot_duty
            if left >= right:
                self.car.set_motor_model(-duty, -duty, duty, duty)
            else:
                self.car.set_motor_model(duty, duty, -duty, -duty)
            if elapsed > 0.6:
                self._brake()
                boxed_in = (
                    min(left, right, self._scan_readings.get("center", SONIC_CLEAR_CM))
                    < self.p.stuck_distance_cm
                )
                self._stuck_count = self._stuck_count + 1 if boxed_in else 0
                self._next_phase()

        elif self._phase == 5:  # let the camera settle, then resume
            if elapsed > self.p.pan_settle_s:
                self._set_steady(True)
                if self._stuck_count >= 2:
                    self._enter(RECOVER)
                else:
                    self._enter(WANDER)

    def _tick_hold(self, now: float) -> None:
        """Stopped and steady, so the vision thread can confirm or clear."""
        elapsed = now - self._state_since
        self._brake()

        if self._phase == 0:
            if elapsed > 0.30:  # mechanical settle after braking
                self._set_pan(self.p.pan_center)
                self._set_steady(True)
                with self.shared.lock:
                    self.shared.hold_granted = True
                self._next_phase()
            return

        with self.shared.lock:
            alert = self.shared.request_alert
            still_wanted = self.shared.request_hold

        if alert:
            self._enter(ALERT)
            return

        # Release when vision is done, or when the hold has run too long --
        # a robot that stares at a wall forever is worse than one that moves on.
        if not still_wanted or elapsed > self.p.hold_max_s:
            if elapsed > self.p.hold_max_s:
                print(f"[drive] hold timed out after {self.p.hold_max_s:.0f}s")
            self._release_hold()
            self._ignore_holds_until = now + self.p.post_hold_ignore_s
            self._enter(WANDER)

    def _tick_alert(self, now: float) -> None:
        """Signal a confirmed violation. Stays stopped and quiet about it."""
        elapsed = now - self._state_since
        self._brake()
        if self._phase == 0:
            if self.p.buzz_on_violation:
                self.car.buzz(True)
            self._next_phase()
        elif self._phase == 1 and elapsed > self.p.buzzer_on_violation_s:
            self.car.buzz(False)
            self._next_phase()
        elif self._phase == 2 and elapsed > self.p.buzzer_on_violation_s + 0.6:
            with self.shared.lock:
                self.shared.request_alert = False
            self._release_hold()
            self._ignore_holds_until = now + self.p.post_hold_ignore_s
            self._enter(WANDER)

    def _tick_recover(self, now: float) -> None:
        """Boxed in: back out properly and turn a long way."""
        elapsed = now - self._state_since
        self._set_steady(False)
        if self._phase == 0:
            self.car.set_motor_model(*([self.p.reverse_duty] * 4))
            if elapsed > 1.0:
                self._next_phase()
        elif self._phase == 1:
            duty = self.p.pivot_duty
            self.car.set_motor_model(-duty, -duty, duty, duty)
            if elapsed > 2.0:
                self._brake()
                self._next_phase()
        else:
            self._stuck_count = 0
            self._set_steady(True)
            self._enter(WANDER)

    def _tick_low_battery(self, now: float) -> None:
        """Terminal. Stay stopped; the uploader keeps draining the queue."""
        self._brake()
        if self._phase == 0:
            print(
                f"[drive] battery {self.shared.battery_v:.2f}V below "
                f"{self.p.min_battery_v}V -- parked. Snapshots continue uploading."
            )
            self.car.buzz(True)
            self._next_phase()
        elif self._phase == 1 and now - self._state_since > 1.0:
            self.car.buzz(False)
            self._next_phase()

    # ---------------------------------------------------------------- helpers

    # States in which the robot is meant to be stationary. Braking happens on
    # transition rather than on the next tick, so deciding to stop and actually
    # stopping are the same event -- otherwise the robot keeps cruising for a
    # full tick (0.2s at cruise duty) after spotting someone.
    _STATIONARY = frozenset({HOLD, ALERT, LOW_BATTERY, STOPPED, BLOCKED_SCAN})

    def _enter(self, state: str) -> None:
        if state == self.state:
            return
        self.state = state
        self._state_since = time.monotonic()
        self._phase = 0
        self._phase_since = self._state_since
        self._scan_readings.clear()
        with self.shared.lock:
            self.shared.drive_state = state
        if state in self._STATIONARY:
            self._brake()

    def _next_phase(self) -> None:
        self._phase += 1
        self._phase_since = time.monotonic()

    def _brake(self) -> None:
        # duty 0 is an ACTIVE brake (motor.py writes 4095 to both H-bridge
        # channels), not coast, so it is set on entry rather than every tick.
        self.car.set_motor_model(0, 0, 0, 0)

    def _set_pan(self, angle: int) -> None:
        with self.shared.lock:
            changed = self.shared.pan_angle != angle
            self.shared.pan_angle = angle
        if changed:
            self.car.set_servo_pwm("0", angle)

    def _set_steady(self, steady: bool) -> None:
        with self.shared.lock:
            self.shared.camera_steady = steady

    def _release_hold(self) -> None:
        with self.shared.lock:
            self.shared.hold_granted = False
            self.shared.request_hold = False

    def _read_distance(self, update_shared: bool = True) -> float:
        """Read the sonar, coping with its two quirks.

        `get_distance()` can return None (ultrasonic.py:21 swallows a
        RuntimeWarning), and gpiozero reports the 300cm ceiling when nothing
        echoes. A None is treated as "unchanged" rather than "clear", because
        assuming clear on a failed read drives into things.
        """
        try:
            value = self.car.get_distance()
        except Exception as exc:
            print(f"[drive] sonar read failed: {exc}")
            value = None
        if value is None or value <= 0:
            value = self._last_distance
        self._last_distance = float(value)
        if update_shared:
            with self.shared.lock:
                self.shared.sonic_center_cm = self._last_distance
        return self._last_distance

    def _check_battery(self, now: float) -> None:
        if now - self._last_battery_check < 10.0:
            return
        self._last_battery_check = now
        try:
            volts = self.car.battery_voltage()
        except Exception:
            return
        with self.shared.lock:
            self.shared.battery_v = volts
        if 0.5 < volts < self.p.min_battery_v and self.state != LOW_BATTERY:
            self._enter(LOW_BATTERY)

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout)
        try:
            self.car.set_motor_model(0, 0, 0, 0)
            self.car.buzz(False)
        except Exception:
            pass


# --------------------------------------------------------------------------
# vision thread
# --------------------------------------------------------------------------


class PatrolVision(threading.Thread):
    """Capture -> detect -> rules -> confirm -> snapshot, plus the live view."""

    def __init__(
        self,
        camera,
        shared: SharedState,
        cfg,
        store=None,
        frame_buffer=None,
        verbose: bool = False,
    ) -> None:
        super().__init__(name="ppe-vision", daemon=True)
        self.camera = camera
        self.shared = shared
        self.cfg = cfg
        self.store = store
        self.frame_buffer = frame_buffer
        self.verbose = verbose
        self._stop_event = threading.Event()
        self.detector = None
        self.engine = None
        self.vision_state = None
        self.error: str | None = None

    def _build(self) -> None:
        """Construct the ORT session here, inside the thread.

        Not in __init__: main.py starts a multiprocessing.Process for the LEDs
        (main.py:333), and a live onnxruntime/OpenMP thread pool at fork time
        is a classic silent hang.
        """
        from ppe_detector import PpeDetector
        from ppe_rules import RuleEngine
        from ppe_tracker import PatrolVisionState

        self.detector = PpeDetector.from_config(self.cfg, which="confirm")
        self.detector.warmup(2)
        self.engine = RuleEngine(self.cfg.rules, self.cfg.model)
        self.vision_state = PatrolVisionState(self.cfg.temporal, self.cfg.cooldown)
        print(
            f"[vision] detector {self.detector.onnx_path.name} "
            f"input {self.detector.input_size}, requiring "
            f"{self.cfg.rules.required_ppe}"
        )

    def run(self) -> None:
        try:
            self._build()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[vision] FAILED to start: {self.error}")
            return

        from ppe_detector import annotate
        from ppe_mjpeg import encode_jpeg

        # Minimum seconds between inference passes. Deliberately capped: a
        # patrolling robot looking for a condition that persists for seconds
        # gains nothing from running flat out, and the cap halves both the
        # heat and the battery drain.
        min_interval = max(0.0, float(self.cfg.patrol.scan_interval_s))

        while not self._stop_event.is_set():
            loop_start = time.perf_counter()
            frame = self.camera.capture_bgr()
            if frame is None:
                self._stop_event.wait(0.1)
                continue

            height, width = frame.shape[:2]
            t0 = time.perf_counter()
            try:
                detections = self.detector.detect(frame)
            except Exception as exc:
                print(f"[vision] inference failed: {exc}")
                self._stop_event.wait(0.5)
                continue
            infer_ms = (time.perf_counter() - t0) * 1000.0

            observations = self.engine.evaluate_frame(detections, (width, height))
            self._update_person_telemetry(observations, height, infer_ms)

            # Only frames captured while stopped and settled count toward
            # N-of-M. Everything else maintains identity but adds no evidence.
            recording = self.shared.may_record()
            events = self.vision_state.process(
                observations, now=time.monotonic(),
                frame_bgr=frame, record=recording,
            )
            with self.shared.lock:
                self.shared.frames_total += 1
                if recording:
                    self.shared.frames_recorded += 1

            for event in events:
                self._handle_violation(event, frame, (width, height))

            if self.frame_buffer is not None and self.frame_buffer.has_viewers:
                self._publish(frame, detections, observations, annotate, encode_jpeg)

            if self.verbose:
                self._log_tick(observations, infer_ms, recording)

            # Cap the rate deliberately. A patrolling robot looking for a
            # condition that persists for seconds does not need more, and the
            # cap halves both heat and battery drain.
            spare = min_interval - (time.perf_counter() - loop_start)
            if spare > 0:
                self._stop_event.wait(spare)

        print("[vision] stopped")

    # ---------------------------------------------------------------- helpers

    def _update_person_telemetry(self, observations, frame_h: int, infer_ms: float) -> None:
        trigger = self.cfg.patrol.person_trigger_conf
        candidates = [o for o in observations if o.person.conf >= trigger]
        largest = max((o.height_frac for o in candidates), default=0.0)
        self.shared.set_person(bool(candidates), largest)
        with self.shared.lock:
            self.shared.infer_ms = infer_ms

        # Ask for a hold whenever there is someone worth inspecting. The drive
        # thread decides whether and when to grant it.
        self.shared.ask_hold(bool(candidates))

    def _handle_violation(self, event, frame, frame_shape) -> None:
        from ppe_snapshot import build_payload

        # Snapshot the clearest confirming frame the tracker cached, not
        # whichever frame happened to tip the counter over.
        track = self.vision_state.tracker.tracks.get(event.track_id)
        best = getattr(track, "best_frame", None) if track else None
        evidence_frame = best if best is not None else frame

        state = self.shared.snapshot()
        payload = build_payload(
            event,
            self.cfg,
            frame_shape,
            battery_v=state["battery_v"] or None,
            patrol_state=state["drive_state"],
            pan_angle=state["pan_angle"],
            sonic_cm=state["sonic_cm"],
        )

        items = ", ".join(event.items)
        agree = ", ".join(f"{k} {v}" for k, v in event.frames_agree.items())
        print(f"[vision] VIOLATION track {event.track_id}: no {items} ({agree})")

        if self.store is not None:
            try:
                event_id = self.store.write_event(evidence_frame, payload)
                print(f"[vision] queued snapshot {event_id[:12]}")
            except Exception as exc:
                print(f"[vision] could not write snapshot: {exc}")

        with self.shared.lock:
            self.shared.violations_total += 1
            self.shared.last_violation_at = time.time()
        self.shared.ask_alert()

    def _publish(self, frame, detections, observations, annotate, encode_jpeg) -> None:
        try:
            drawn = annotate(frame, detections, observations)
            state = self.shared.snapshot()
            self._draw_hud(drawn, state)
            jpeg = encode_jpeg(drawn, self.cfg.camera.mjpeg_quality)
            if jpeg:
                self.frame_buffer.publish(jpeg)
                self.frame_buffer.status_line = (
                    f"{state['drive_state']} &middot; "
                    f"{state['violations']} violations &middot; "
                    f"{state['infer_ms']:.0f} ms/frame"
                )
        except Exception as exc:
            print(f"[vision] live view failed: {exc}")

    def _draw_hud(self, frame, state: dict) -> None:
        import cv2

        steady = "STEADY" if state["camera_steady"] else "MOVING"
        line1 = (
            f"{state['drive_state']}  {steady}  "
            f"{state['infer_ms']:.0f}ms  {state['sonic_cm']:.0f}cm"
        )
        line2 = (
            f"violations {state['violations']}  "
            f"battery {state['battery_v']:.2f}V  "
            f"frames {state['recorded']}/{state['frames']}"
        )
        h = frame.shape[0]
        cv2.rectangle(frame, (0, h - 38), (frame.shape[1], h), (20, 20, 20), -1)
        for i, text in enumerate((line1, line2)):
            cv2.putText(
                frame, text, (8, h - 22 + i * 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA,
            )
        if not state["camera_steady"]:
            cv2.putText(
                frame, "frames discarded (camera moving)",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 180, 250), 1, cv2.LINE_AA,
            )

    def _log_tick(self, observations, infer_ms: float, recording: bool) -> None:
        from ppe_rules import summarize

        state = self.shared.snapshot()
        flag = "REC" if recording else "   "
        print(
            f"  {flag} {state['drive_state']:<13} pan={state['pan_angle']:>3} "
            f"sonic={state['sonic_cm']:>5.0f} {infer_ms:>5.0f}ms "
            f"people={len(observations)} {self.vision_state.describe()}"
        )
        if observations:
            print(summarize(observations))

    def stop(self, timeout: float = 5.0) -> None:
        # Flag and join. Never Thread.py:stop_thread -- that injects a ctypes
        # async exception, which would abandon the ORT session and the camera.
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout)
        if self.detector is not None:
            self.detector.close()
