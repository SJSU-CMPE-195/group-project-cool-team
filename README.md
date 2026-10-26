# S.A.F.E.R (Smart Autonomous Facility Enforcement Robot)

## Team
- Ricky Hernandez (Jason.a.hernandez@sjsu.edu)
- David Zhang (David.zhang@sjsu.edu)
- Ricky Liang (ricky.liang@sjsu.edu)
- Viet-Joshua Le (viet-joshua.le@sjsu.edu)

**Advisor:** Eric Vanuska (eric.vanuska@sjsu.edu)

## Grading Summary

```yaml
impl_claim: >
  A Raspberry Pi 5 on a Freenove 4WD chassis wanders an area on its own and
  steers away from obstacles closer than 35 cm, using an ultrasonic sensor.
  When the camera sees a person, the robot stops and runs a YOLOv8n PPE model
  (17 SH17 classes, ONNX through onnxruntime) on each frame. It flags a hard
  hat violation when a bare "head" box with no "helmet" box over it is seen in
  at least 5 of 10 consecutive frames. When a violation is confirmed, the
  robot sounds its buzzer, saves the frame as a JPEG plus JSON metadata (the
  detection boxes and the reason for the call) to an on-disk queue, and
  uploads it to a Flask/SQLite admin portal, which draws the boxes over the
  image. The portal
  lists every violation, and a reviewer marks each one confirmed or false
  positive. Those review flags are used to compute precision. If no head or
  helmet is visible, the robot makes no call.
mock_mode: >
  "make mock" runs the real vision, rules, tracker and patrol state machine
  code, with three stand-ins: a simulated car (motors, servos, sonar, buzzer),
  a still image looped as the camera, and snapshots kept on disk instead of
  uploaded. The bundled image shows a worker wearing a hard hat, so the
  expected result is "helmet=PRESENT" on every frame and "violations found 0".
  To see a violation instead, run "make mock MOCK_IMAGE=path/to/photo.jpg" with
  a photo of someone without a hard hat; the snapshot lands in
  Code/Server/snapshots/pending/. "make portal" starts the admin portal and
  seeds it with 24 synthetic violations.
notes: >
  Robot hardware (Raspberry Pi 5, Pi camera, PCA9685 motor/servo board,
  ultrasonic sensor, buzzer) is required for the physical patrol and
  is shown in the demo videos. This repo has no microcontroller firmware: the
  Pi drives the motors and servos directly through the Freenove kit's PCA9685
  board, so there is nothing to flash. "make compile" byte-compiles every
  Python source. The vendored C library Code/Libs/rpi-ws281x-python (LED
  driver) builds only on a Raspberry Pi and is not part of grading. "make
  models" re-exports the ONNX files; this step is optional, because the
  exports are already committed in Code/Server/models/. We checked our ONNX
  decoder against ultralytics' own predict() on an 800x800 photo, with the
  same resize interpolation on both sides. All 13 detections matched (IoU
  1.000, identical confidences). Without that change, the detector shrinks
  images with INTER_AREA where ultralytics uses INTER_LINEAR, so confidences
  can differ by a few hundredths. The Pi's 640x480 frames are never shrunk,
  so this does not affect the robot.
```

## Project Description
S.A.F.E.R is an autonomous robot designed to improve workplace safety in industrial and manufacturing environments by scanning workers for required personal protective equipment. The robot patrols a designated area, verifies safety-wear compliance (hard hats by default; safety vests and other SH17 classes can be enabled), and reports violations with photographic evidence to an admin portal for human review.

## Prototype Scope
- **In scope:** autonomous wander with ultrasonic obstacle avoidance; on-device PPE detection on the Pi 5; multi-frame confirmation of hard-hat violations; local snapshot queue that survives the portal being offline; admin portal with gallery, review flags, precision stats, CSV export and live MJPEG view.
- **Enabled by default:** hard hat (`helmet`). Safety vest is implemented but off by default because it can only be inferred from *absence* of a vest box, which is weaker evidence (see `docs/PPE_PATROL.md` §1).
- **Out of scope:** mapping/SLAM (the chassis has no encoders or IMU, so coverage is random-bounce), multi-robot coordination, model retraining, cloud hosting.

## System Architecture
The Pi camera captures 640×480 frames for the PPE detection system. The Raspberry Pi 5 runs everything on board: it runs the trained YOLO model (exported to ONNX), applies the rule engine that turns detections into per-person PPE verdicts, and confirms violations across frames. It also drives the robot: it reads the ultrasonic sensor and commands the motors and pan/tilt servos through the Freenove kit's PCA9685 PWM board, so the robot can move through obstacles in its area. The camera and ultrasonic sensor share one pan bracket, so the robot **stops before judging**. Frames taken while it moves or sweeps are discarded. Confirmed violations go to an on-disk queue. A background uploader posts them to the admin portal running on a PC.

```
Pi camera ─► ppe_detector (ONNX) ─► ppe_rules ─► ppe_tracker (5-of-10) ─► ppe_snapshot queue ─► ppe_uploader ─► Portal (Flask + SQLite)
Ultrasonic ─► ppe_patrol state machine (WANDER / HOLD / ALERT / BLOCKED_SCAN) ─► PCA9685 ─► motors + servos
```

Diagrams: `Diagrams/*.drawio`. Full design write-up: [`docs/PPE_PATROL.md`](docs/PPE_PATROL.md).

## Prerequisites
**To grade (no hardware):** Python 3.10+ and GNU `make`. On Windows, run it from Git Bash. All other dependencies are installed into `.venv` by the Makefile.

**To run the physical robot:** Raspberry Pi 5 (with Active Cooler), Pi camera module, Freenove 4WD Smart Car kit (chassis, 4 DC motors, PCA9685 motor/servo board, ultrasonic sensor, pan/tilt servos, buzzer), 2× 18650 batteries, and Raspberry Pi OS Bookworm. Software includes Python, Flask, OpenCV, onnxruntime and Ultralytics YOLO (needed only on a PC, to export the model). The trained weights are `yolo8n.pt`; the committed ONNX exports are in `Code/Server/models/`.

## Installation

### Grader quick start
```bash
git clone https://github.com/SJSU-CMPE-195/group-project-cool-team.git
cd group-project-cool-team
make            # install + compile + 130 tests + 20 s mock patrol
```

| Command | What it does |
|---|---|
| `make install` | creates `.venv` and installs `requirements-dev.txt` |
| `make compile` | byte-compiles all Python sources |
| `make test` | runs all 130 unit tests (rules, detector, tracker, patrol, portal) |
| `make mock` | full patrol pipeline with a simulated car and a still image as the camera |
| `make portal` | seeds demo data and serves the portal at http://127.0.0.1:5050 |
| `make models` | (optional) re-exports the ONNX models from `yolo8n.pt` |
| `make clean` | removes `.venv`, caches and mock output |

### On the robot (Raspberry Pi 5)
1. Clone the repository on the Pi: `git clone https://github.com/SJSU-CMPE-195/group-project-cool-team.git`
2. Install system packages and create the venv. It **must** use `--system-site-packages` so picamera2 keeps its numpy 1.x:
   ```bash
   sudo apt install -y python3-opencv python3-requests python3-numpy
   python3 -m venv --system-site-packages ~/ppe-venv
   ~/ppe-venv/bin/pip install -r Code/Server/requirements-pi.txt
   ~/ppe-venv/bin/pip list --local     # must NOT list numpy or opencv-python
   ```
3. Configure the AI model. The ONNX exports are already in `Code/Server/models/`. To rebuild them, run `make models` on a PC.
4. Configure the camera. Connect the camera module to the Pi 5 and check that capture works before running detection: `python ppe_camera.py --source picamera`.
5. Configure the robot. Run the stock `python main.py` once from `Code/Server`, interactively, to create `params.json`. Then create `Code/Server/ppe_config.local.json` with the portal URL and API key (see `docs/PPE_PATROL.md` §3c).
6. Connect the hardware. Assemble the Freenove kit following `Tutorial(ordinary_wheels).pdf`, and check that the motors, servos and ultrasonic sensor respond before you start a patrol.
7. On the PC, start the portal (`pip install -r Portal/requirements.txt`, then `python Portal/app.py`). Allow inbound TCP 5050 through the firewall.

## Running the Prototype
```bash
cd Code/Server
~/ppe-venv/bin/python ppe_main.py                   # full autonomous patrol
~/ppe-venv/bin/python ppe_main.py --no-motors -v    # bench test, wheels off
~/ppe-venv/bin/python ppe_main.py --simulate -v     # no hardware at all
```
Live annotated view: `http://<pi>:8080/`, or through the portal at `/live`. Violations appear at `http://<pc>:5050/`.

## Testing
```bash
make test
```
| Suite | Tests | Covers |
|---|---|---|
| `Code/Server/test_ppe_rules.py` | 22 | head/helmet association, carried-not-worn, ambiguity refusal, vest absence rules |
| `Code/Server/test_ppe_detector.py` | 25 | letterbox round-trip, YOLOv8 decode, class-wise NMS |
| `Code/Server/test_ppe_tracker.py` | 23 | N-of-M confirmation, cooldowns, de-duplication |
| `Code/Server/test_ppe_patrol.py` | 26 | drive state machine, stop-and-look handshake, fault handling |
| `Portal/test_portal.py` | 34 | ingest, idempotency, auth, gallery, stats |

None of these tests need the robot. Some print messages like `sensor on fire`; those come from intentional fault-injection tests.

## Demo
**Autonomous navigation and obstacle avoidance:** the built robot navigating with object detection running, stopping just before it hits obstacles.
- https://drive.google.com/file/d/1LniFDaX6oen-bgjoeET2wxvGK0oPmdRs/view?usp=drive_link

**Servo tracking and live PPE detection:** the built robot with its servos working. It looks around and locks onto people while running PPE (safety gear) detection live. The annotated camera feed is streamed locally over HTML, so we can watch what the detector sees.
- https://drive.google.com/file/d/1PDEvJdE0BhoLxZ3X0HmEzh4dVs2YntvA/view?usp=sharing

## Technical Stack
**Hardware**
- Raspberry Pi 5 (8 GB) with Active Cooler
- Pi camera module
- Freenove 4WD Smart Car kit: PCA9685 motor/servo driver, 4 DC gear motors, pan/tilt servos
- Ultrasonic sensor
- Buzzer and WS2812 LEDs
- 2× 18650 batteries and the kit's regulator

**Software**
- Python
- Flask + SQLite (admin portal)
- Ultralytics YOLOv8 (training/export), onnxruntime (on-robot inference)
- Raspberry Pi OS Bookworm, picamera2
- OpenCV, NumPy
- Kaggle / SH17 dataset

## Repository Layout
```
Code/Server/ppe_*.py    patrol: detector, rules, tracker, snapshot queue, uploader, state machine
Code/Server/test_*.py   unit tests (no hardware)
Code/Server/models/     exported ONNX models
Code/                   Freenove kit vendor code (unmodified)
Portal/                 Flask + SQLite admin portal
Tools/export_onnx.py    ONNX export + parity check against ultralytics
docs/PPE_PATROL.md      full setup, tuning and troubleshooting guide
Diagrams/               system block, detection pipeline, decision flowchart
modeldraft/             earlier Flask model-demo prototype (best.pt)
Makefile                grader entry points
```

## What's Next
During CMPE 195B, our team will focus on completing the physical integration of S.A.F.E.R. and validating its functionality. Phase 1 will integrate the AI model into the Raspberry Pi 5, Phase 2 will assemble the robot and connect its hardware components, and Phase 3 will integrate AI detection with autonomous movement and obstacle avoidance. Phase 4 will focus on testing detection accuracy, movement reliability, and system communication before preparing the final demonstration. By the end of the project, we aim to demonstrate a functional robot capable of recognizing selected PPE categories, navigating a predefined route, and responding to obstacles in a controlled environment.

## Licensing
Chassis code derives from the [Freenove 4WD Smart Car Kit](https://github.com/Freenove/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi) (CC BY-NC-SA 3.0, see `LICENSE.txt`). The model weights carry Ultralytics' AGPL-3.0 metadata.
