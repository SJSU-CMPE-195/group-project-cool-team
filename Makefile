# S.A.F.E.R. -- grader entry points. Nothing here needs the robot.
#
#   make            same as `make all`: install, compile check, tests, mock run
#   make install    create .venv and install desktop dependencies
#   make compile    byte-compile every Python source (syntax/import-time check)
#   make test       run all 130 unit tests (rules, detector, tracker, patrol, portal)
#   make mock       run the full patrol pipeline with a simulated car and a still image
#   make portal     start the admin portal at http://127.0.0.1:5050 (seeded demo data)
#   make models     re-export the ONNX models from yolo8n.pt (optional, ~2 GB download)
#   make clean      remove .venv, caches and mock-run output
#
# Works with GNU make on Linux, macOS and Windows (Git Bash / MSYS shell).

ifeq ($(OS),Windows_NT)
  SYS_PY ?= python
  VENV_PY := .venv/Scripts/python.exe
else
  SYS_PY ?= python3
  VENV_PY := .venv/bin/python
endif

PY := $(abspath $(VENV_PY))
SERVER := Code/Server
MOCK_IMAGE ?= modeldraft/uploads/16296adea9a9.jpg
MOCK_SECONDS ?= 20

.PHONY: all help install compile test mock portal models clean

all: install compile test mock

help:
	@sed -n '3,12p' Makefile

$(VENV_PY): requirements-dev.txt
	$(SYS_PY) -m venv .venv
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements-dev.txt
	@touch $(VENV_PY)

install: $(VENV_PY)

compile: install
	$(PY) -m compileall -q $(SERVER) Portal Tools modeldraft/app.py
	@echo "compile: OK"

test: install
	cd $(SERVER) && $(PY) test_ppe_rules.py
	cd $(SERVER) && $(PY) test_ppe_detector.py
	cd $(SERVER) && $(PY) test_ppe_tracker.py
	cd $(SERVER) && $(PY) test_ppe_patrol.py
	cd Portal && $(PY) test_portal.py

# mock_mode: --simulate swaps the Freenove motor/servo/sonar/buzzer drivers for
# a simulated car, --source file replays a still image instead of the Pi
# camera, and --no-upload keeps snapshots on disk instead of posting them.
mock: install
	cd $(SERVER) && $(PY) ppe_main.py --simulate --source file \
		--path $(abspath $(MOCK_IMAGE)) --no-upload --no-live \
		--duration $(MOCK_SECONDS) -v
	@echo "Snapshots written to $(SERVER)/snapshots/pending/"

# Starts the portal, waits for /api/v1/health, seeds 24 synthetic violations
# through the real ingest endpoint, then keeps serving until Ctrl-C.
portal: install
	$(PY) Portal/app.py & pid=$$!; \
	for i in $$(seq 1 30); do \
		$(PY) -c "import urllib.request as u; u.urlopen('http://127.0.0.1:5050/api/v1/health')" 2>/dev/null && break; \
		sleep 1; \
	done; \
	$(PY) Portal/seed_demo.py; \
	echo "Portal running at http://127.0.0.1:5050 -- Ctrl-C to stop"; \
	wait $$pid

models:
	@echo "Needs torch 2.4.1 + ultralytics 8.2.38 in a separate venv; see docs/PPE_PATROL.md section 3a."
	$(SYS_PY) -m venv .venv-export
	.venv-export/$(if $(filter Windows_NT,$(OS)),Scripts/python.exe,bin/python) -m pip install "ultralytics==8.2.38" "numpy<2" onnx onnxsim onnxruntime
	# ultralytics pulls the newest torch, which cannot load this checkpoint; pin it back.
	.venv-export/$(if $(filter Windows_NT,$(OS)),Scripts/python.exe,bin/python) -m pip install "torch==2.4.1" "torchvision==0.19.1" --index-url https://download.pytorch.org/whl/cpu
	.venv-export/$(if $(filter Windows_NT,$(OS)),Scripts/python.exe,bin/python) Tools/export_onnx.py --imgsz 640
	.venv-export/$(if $(filter Windows_NT,$(OS)),Scripts/python.exe,bin/python) Tools/export_onnx.py --imgsz 448

clean:
	rm -rf .venv .venv-export $(SERVER)/snapshots
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
