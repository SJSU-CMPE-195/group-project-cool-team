# PPE Patrol — a Roomba for hardhat compliance

An autonomous patrol built on the Freenove 4WD Smart Car (Raspberry Pi 5, ordinary wheels).
The robot wanders an area, stops when it sees a person, decides whether they're wearing a
hardhat, and uploads a snapshot of any violation to an admin portal on your PC.

Everything runs **on the Pi** — camera, YOLO inference, driving, and the violation queue. The
only thing on the PC is the portal, and the robot works fine when the PC is switched off.

---

## 1. What it actually detects, and why that shapes everything

Your `yolo8n.pt` is a real PPE model — a YOLOv8-nano fine-tuned on the **SH17 safety dataset**
(run `safehuman_nano_v0_14`, ultralytics 8.2.38, trained at imgsz 640). It has **17 classes**:

```
0 person   4 face-guard  8  glasses  12 head
1 ear      5 face-mask   9  gloves   13 medical-suit
2 ear-mufs 6 foot        10 helmet   14 shoes
3 face     7 tool        11 hands    15 safety-suit
                                     16 safety-vest
```

(The other file, `yolo11s.pt`, is the stock COCO model — person, bicycle, car… no PPE classes
at all. It's gitignored.)

**There is no `no-helmet` class.** That single fact drives the whole design. What the dataset
*does* give you is paired classes: a bare `head` and a `helmet` are labelled separately, as are
bare `hands` and `gloves`. So a violation is never a detection — it's an inference over the
geometry of a frame:

> attach each `head`/`helmet` box to a person by containment → a person with a bare `head` box
> and no `helmet` covering it is a violation.

That's **positive evidence**: we can see the uncovered head. It's trustworthy, and it's why the
default configuration enforces helmet only.

Compare `safety-vest`, which has no counterpart class. The only possible reasoning is "no vest
box appeared, so there's no vest" — **absence evidence**, which is much weaker. No box might
mean no vest, or a person with their back turned, or someone half out of frame. The rule engine
treats the two completely differently:

| | helmet / gloves / shoes / face-mask | safety-vest |
|---|---|---|
| Evidence | positive (uncovered body part seen) | absence only |
| Frames needed | 5 of 10 | **8 of 10** |
| Requires a clear view | no | **yes** (≥35% of frame height, not clipped by an edge) |

Three further guards keep it honest:

- **Presence veto.** One confident `helmet` detection clears the person outright. Positive
  evidence of compliance always beats inference.
- **Two thresholds per class.** `class_conf` to be a detection at all; the higher
  `evidence_conf` to be used as *proof of a violation*. Seeing a `head` at 0.35 is fine;
  accusing someone on the strength of it is not.
- **Anatomy gates.** A "helmet" detected at knee height is a misdetection, not PPE, and is
  rejected. Equally, a helmet detected in the frame but *not overlapping the bare head we can
  see* reads as **carried, not worn** — still a violation.
- **Ambiguity refusal.** With two overlapping people and one helmet, attributing it to either
  invents a violation for the other, so the verdict is `UNKNOWN` for both. Refusing to answer is
  the correct behaviour in a crowd.

Critically, when neither a `helmet` nor a `head` is visible, the answer is `UNKNOWN` — **never a
violation**. Reporting "missing" from no evidence is the single fastest way to fill the portal
with noise until nobody reads it.

---

## 2. Why the robot stops to look

The ultrasonic sensor and the camera are bolted to the **same pan/tilt bracket**. They
physically cannot look in different directions. Upstream's obstacle mode sweeps the pan servo
across three bearings every 0.2 s (`car.py:69-88`), which would swing the camera continuously —
and inference on a camera panning across a room is worthless.

So the two activities take turns, and the robot stops before deciding anything:

```
WANDER        cruising, camera fixed forward and steady, inference running
  │ person detected
  ▼
HOLD          brake → 0.3 s settle → camera_steady = true, hold_granted = true
              only now do frames count toward 5-of-10 agreement
  │ violation confirmed
  ▼
ALERT         buzzer chirp, snapshot queued to disk
  │
  ▼
WANDER        (with a 5 s hold-ignore so the same person doesn't re-trigger)

BLOCKED_SCAN  obstacle under 35 cm: stop, reverse, sweep pan 60°/90°/120°,
              turn toward open space. camera_steady = FALSE throughout —
              every frame taken during a sweep is discarded.
```

This resolves three problems with one mechanism:

1. The servo conflict disappears — sweeping only happens while stopped.
2. Temporal confirmation becomes meaningful. N-of-M agreement across frames taken *while
   driving* compares different scenes and means nothing.
3. Motion blur, the dominant cause of both missed detections and false positives on a robot
   driving over a hard floor, is eliminated at the moment of decision.

Note the pan sweep uses **60°/90°/120°**, not upstream's 30°/150°. The factory servo test only
covers pan 50–110 (`test.py:91-96`), so upstream's own autonomous mode drives the bracket past
its stated safe range. `servo.py` does no angle clamping, and a stalled servo draws current and
gets hot.

---

## 3. Setup

### 3a. On the PC: export the model (once)

The Pi never needs torch or ultralytics. Export to ONNX here, commit the result, and the Pi
runs onnxruntime alone — a single 15 MB wheel instead of a 2 GB install.

```powershell
py -3.12 -m venv $env:TEMP\yolo-export
& $env:TEMP\yolo-export\Scripts\pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cpu
& $env:TEMP\yolo-export\Scripts\pip install "ultralytics==8.2.38" onnx onnxsim onnxruntime

& $env:TEMP\yolo-export\Scripts\python Tools\export_onnx.py --imgsz 640
& $env:TEMP\yolo-export\Scripts\python Tools\export_onnx.py --imgsz 448
```

Both pins matter. **ultralytics 8.2.38** is the version that produced your checkpoint; later
versions have renamed internals and may fail to unpickle it. **torch 2.4.x** because torch 2.6
flipped `torch.load(weights_only=True)` to the default, which breaks checkpoints of this era.

Then run the verification, which is **the most valuable single test in this project**:

```powershell
& $env:TEMP\yolo-export\Scripts\python Tools\export_onnx.py --imgsz 640 --verify-only --verify yourphoto.jpg
```

Use a photo of a person, ideally one with a hardhat and one without. It runs ultralytics'
`predict()` *and* our onnxruntime path on the same image and compares every box. `ppe_detector.py`
reimplements letterboxing, YOLOv8 head decoding and NMS by hand; a subtle error there doesn't
crash, it quietly degrades every detection forever. Once the boxes agree, delete the export venv
and never think about torch again.

### 3b. On the PC: the portal

```powershell
pip install -r Portal\requirements.txt
$env:PPE_API_KEY = "pick-a-shared-secret"
$env:PPE_ADMIN_PASSWORD = "pick-a-password"
$env:PPE_ROBOT_MJPEG = "http://192.168.129.51:8080/stream.mjpg"   # your Pi's address
python Portal\app.py
```

Then `http://127.0.0.1:5050`. To see it populated before the robot exists:

```powershell
python Portal\seed_demo.py          # 24 synthetic violations
python Portal\seed_demo.py --clear  # remove them
```

**Windows Firewall is the single most common failure here.** Inbound TCP 5050 must be allowed on
the *Private* profile, or the Pi's uploads are silently dropped and it looks like a robot fault —
Flask's dev server logs nothing when packets never arrive. Test it from another machine before
involving the Pi:

```powershell
Invoke-WebRequest http://<your-pc-ip>:5050/api/v1/health
```

### 3c. On the Pi

```bash
sudo apt update
sudo apt install -y python3-opencv python3-requests python3-numpy

python3 -m venv --system-site-packages ~/ppe-venv
~/ppe-venv/bin/pip install --upgrade pip
~/ppe-venv/bin/pip install onnxruntime

# THE IMPORTANT CHECK -- see the warning below
~/ppe-venv/bin/pip list --local
~/ppe-venv/bin/python -c "import picamera2, cv2, numpy, gpiozero, onnxruntime as o; print(numpy.__version__, cv2.__version__, o.__version__)"
```

> ### ⚠ The numpy ABI trap
>
> `--system-site-packages` is **mandatory** — without it picamera2, gpiozero and PyQt5 vanish,
> since Bookworm ships them as system packages.
>
> More subtly: Bookworm's picamera2 stack is compiled against **numpy 1.x**. If pip ever pulls
> `numpy>=2` or `opencv-python` into this venv, camera capture breaks with ABI errors that look
> exactly like a camera hardware fault, and you will spend an evening on it. `pip list --local`
> is the canary — it should show onnxruntime and its own small dependencies, and **neither numpy
> nor opencv-python**. If it does, `pip uninstall numpy opencv-python` and reinstall onnxruntime
> with `--no-deps`.
>
> This is exactly why the Pi doesn't get ultralytics: it depends on `opencv-python` and
> `numpy>=2`, so installing it would break the camera.

Set your portal address and key in a gitignored local config:

```bash
cd ~/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi/Code/Server
cat > ppe_config.local.json <<'EOF'
{
  "portal": { "url": "http://192.168.1.50:5050", "api_key": "pick-a-shared-secret" },
  "zone_label": "workshop"
}
EOF
```

**Create `params.json` before running headless.** `parameter.py` prompts on stdin via `input()`
when it's missing (`parameter.py:96-114`), and both `Led()` and `ADC()` construct a
`ParameterManager` — so a first run under systemd or a detached shell hangs forever on an
invisible prompt. Run stock `python main.py` once interactively from `Code/Server` to create it.
`ppe_main.py` pre-checks this and refuses to start with a clear message rather than hanging.

---

## 4. Running

```bash
cd ~/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi/Code/Server

~/ppe-venv/bin/python ppe_main.py                  # full patrol
~/ppe-venv/bin/python ppe_main.py --no-motors -v    # bench test on blocks, verbose trace
~/ppe-venv/bin/python ppe_main.py --simulate -v     # no hardware at all
~/ppe-venv/bin/python ppe_main.py --duration 120    # stop after 2 minutes
```

Live view: `http://<pi>:8080/` directly, or through the portal at `/live` so one address gives
you both the live feed and the violation history. The live view shows **annotated** frames —
boxes and PPE verdicts drawn on the Pi — so it's what the detector actually sees, which makes it
a real debugging tool. It also shows a `frames discarded (camera moving)` banner whenever the
camera isn't steady, so the handshake is visible at a glance.

**Picamera2 is exclusive.** Run stock `main.py` (manual driving, PyQt client) *or* `ppe_main.py`
(patrol), never both. Nothing in upstream's files was modified, so stock behaviour is untouched.

On a PC, you can develop the whole vision stack against a webcam before the robot is involved:

```powershell
python Code\Server\ppe_main.py --simulate --source webcam -v
```

---

## 5. Verification, in stages

Each stage is independently checkable, and the first five need no robot.

| Stage | Command | Gate |
|---|---|---|
| 0 | `Tools/export_onnx.py --verify photo.jpg` | **Our boxes match ultralytics'** — proves the hand-written decoder |
| 1 | `python test_ppe_rules.py` | 22 tests: violation logic |
| 1 | `python test_ppe_detector.py` | 25 tests: letterbox round-trip, class-wise NMS |
| 1 | `python test_ppe_tracker.py` | 23 tests: N-of-M, cooldown, rate limits |
| 1 | `python test_ppe_patrol.py` | 26 tests: state machine, hold handshake |
| 2 | `python Portal/test_portal.py` | 34 tests: ingest, idempotency, auth, gallery |
| 3 | `ppe_detector.py --bench` on the Pi | **measure real FPS** — see below |
| 4 | `ppe_camera.py --source picamera` | colours correct, not blue-tinted |
| 5 | `ppe_main.py --no-motors -v` on blocks | confirmations fire on a real violation |
| 6 | `ppe_main.py` | 20-minute autonomous run |

### Stage 3 is the one that replaces guesswork

```bash
cd Code/Server
~/ppe-venv/bin/python ppe_detector.py --bench --runs 50 --threads 3
~/ppe-venv/bin/python ppe_detector.py --bench --onnx models/ppe_yolo8n_448.onnx --runs 50
```

It prints p50/p90 latency, FPS, CPU temperature and `vcgencmd get_throttled`. My estimate for a
Pi 5 with the active cooler is roughly **4–7 FPS at 640** and **9–13 FPS at 448**, but treat
those as a hypothesis to test, not a fact — measure yours and set `scan_interval_s` from the
result. Re-run it with the live view open, and expect 20–40% worse: the Pi 5 has no hardware
video encoder, so JPEG encoding competes for the same cores.

You need far less FPS than instinct suggests. A person standing without a hardhat is a condition
that persists for *seconds*. At 5 FPS, ten frames of agreement takes two seconds — comfortably
inside the 8-second hold. `scan_interval_s` deliberately caps the rate: running flat out buys
nothing and doubles your heat and battery cost.

### Stage 5: prove the offline path

With the robot queueing snapshots, **stop the portal**. The patrol must keep running and the
queue must grow. Restart the portal, and everything should backfill with no duplicates:

```bash
~/ppe-venv/bin/python ppe_uploader.py --status    # queue depth
~/ppe-venv/bin/python ppe_uploader.py --health    # can we reach the portal?
~/ppe-venv/bin/python ppe_uploader.py --once      # drain it now
```

---

## 6. Tuning

**Get a precision number before changing any threshold.** The portal's `/stats` page computes it
from your review flags (confirmed vs false positive). Tuning before you have that number is
guessing.

`Code/Server/ppe_config.json` holds every tunable; `ppe_config.local.json` overrides it and is
gitignored. The ones worth touching:

| Setting | Default | Effect |
|---|---|---|
| `rules.required_ppe` | `["helmet"]` | **Add `safety-vest` only after measuring precision** — it's absence-only and will produce more false positives |
| `temporal.positive_n` | 5 of 10 | Raise for fewer, more certain violations |
| `cooldown.per_person_s` | 90 | How often a persisting violation is re-reported |
| `model.evidence_conf.head` | 0.60 | Bar for a bare head to count as proof. Raise if you get false helmets |
| `patrol.scan_interval_s` | 0.15 | Set from your Stage 3 benchmark |
| `patrol.cruise_duty` | 700 | Keep ≤800; faster means more blur and bigger current spikes |
| `rules.min_person_height_frac` | 0.35 | How close someone must be before absence evidence counts |
| `dry_run` | false | Snapshot locally, never upload. **Use this for your first session** |

Run your first real session with `"dry_run": true` and look at what lands in
`Code/Server/snapshots/pending/` before letting it fill the portal.

---

## 7. Things that will bite you

**SH17 is an industrial dataset, and your living room is not a construction site.** A person at
home genuinely isn't wearing a hardhat, so the model will be *correct* and the portal will fill
with "violations". That's the system working. Use `dry_run` to calibrate, and judge it on a real
test: someone putting a hardhat on and taking it off.

**Thermals.** Sustained multi-core inference will push a Pi 5 to 80–85 °C and it throttles at 85.
The official Active Cooler is effectively required; without it your FPS silently degrades after a
few minutes, which is a miserable bug to chase on a moving robot. `intra_op_threads: 3` leaves a
core for the camera and drive loop. CPU temperature is recorded in every snapshot, so throttling
shows up in your data rather than as a mystery.

**Power, and why "software crashes" often aren't.** Two 18650s feed a Pi 5 (8–10 W under
inference) *plus* four motors *plus* a servo. Expect well under an hour. Worse, a motor stall
current spike coinciding with an inference spike can brown out the regulator and reboot the Pi
mid-patrol — which looks exactly like a software crash. Every snapshot records battery voltage
and `vcgencmd get_throttled` (bit 0 = undervoltage detected), so check those first. The
`LOW_BATTERY` state parks the robot at 6.6 V but keeps the uploader draining.

**A nano model trained at 640, run smaller.** Small-object recall collapses first, and a helmet
on a person 150 px tall is only ~25 px. Hence: the camera runs at 640×480 so the confirm pass
letterboxes into 640×640 with *padding only and zero downscaling*; the robot stops before
deciding; and `detect_crop()` can re-run on a padded person box to put small PPE back near its
training scale.

**AGPL-3.0.** The weights carry ultralytics' license metadata, and ONNX export doesn't change
that. Fine for private use; relevant if you ever publish or demo this as a product.

**Privacy.** This stores identifiable photographs of people. Tell anyone in the patrol area, set
a retention policy, and keep it off the internet. The portal's auth — one shared API key and one
password, over plain HTTP — is adequate for a trusted home LAN and **nothing more**: the key and
session cookie cross the network in cleartext. Do not port-forward it. If it ever needs to leave
the LAN, use a WireGuard/Tailscale tunnel so there's no public listener at all.

---

## 8. File map

```
Code/Server/
  ppe_classes.py      17-class vocabulary + the positive/absence item table
  ppe_types.py        Detection + box geometry (zero dependencies)
  ppe_config.py       frozen dataclasses, validated at load
  ppe_config.json     committed defaults
  ppe_detector.py     ONNX inference: letterbox, YOLOv8 decode, class-wise NMS
  ppe_rules.py        person association -> per-item PPE verdicts
  ppe_tracker.py      IoU tracker, N-of-M confirmation, 3-layer de-duplication
  ppe_snapshot.py     atomic on-disk event queue + metadata builder
  ppe_uploader.py     background upload with backoff and idempotent retry
  ppe_camera.py       picamera2 / webcam / file backends
  ppe_mjpeg.py        MJPEG server for the live view
  ppe_patrol.py       drive state machine + vision thread + the handshake
  ppe_main.py         entry point: preflight, threads, signal handling
  test_ppe_*.py       96 tests, no hardware needed
  models/             the exported .onnx files

Portal/               Flask + SQLite admin portal (runs on the PC)
  app.py db.py        routes / schema
  templates/ static/  gallery, detail, live, stats
  test_portal.py      34 tests
  seed_demo.py        synthetic data, to see it working

Tools/export_onnx.py  one-time ONNX export + ultralytics parity check
```

Nothing in upstream's `Code/Server/*.py` or `Code/Client/*.py` was modified.

---

## 9. Known gaps

**There is no map.** This chassis has no wheel encoders, no IMU and no SLAM, so coverage is
statistical random-bounce wander, not systematic room coverage — it's a Roomba in spirit, but an
early one. "Where" in a snapshot is the configured `zone_label` plus a wall-clock time, and the
metadata deliberately avoids implying localisation the robot doesn't have.

Also not built: multi-robot coordination, cloud hosting, per-viewer accounts, and any retraining
of the model.
