# PPE Detector — Local Web App

A local Flask website that classifies PPE (helmet, vest, mask, etc.) in uploaded images using the YOLOv8 model from the Kaggle notebook by **hinepo**.

Runs entirely on your machine — no hosting needed.

---

## Project structure

```
ppe-detector/
├── app.py                  # Flask server + YOLOv8 inference
├── best.pt                 # <-- you add this (trained model weights)
├── requirements.txt
├── templates/
│   └── index.html          # Upload UI
├── static/
│   └── results/            # Annotated output images
└── uploads/                # Raw uploads
```

---

## Step 1 — Get the trained model (`best.pt`)

You have two options:

### Option A (easiest): Download the Kaggle notebook's output
1. Go to https://www.kaggle.com/code/hinepo/yolov8-finetuning-for-ppe-detection
2. Click **Copy & Edit** (top right) — this requires a free Kaggle account.
3. Run all cells. Training on the notebook's GPU takes roughly 15–30 minutes.
4. After the run finishes, open the **Output** tab of your copy of the notebook.
5. Navigate into the `runs/detect/train/weights/` folder and download **`best.pt`**.
6. Place `best.pt` next to `app.py` in this project.

### Option B: Train locally
Fork the notebook, copy the training code into a local script, download the Roboflow PPE dataset referenced in the notebook, and run `model.train(...)` on your own GPU. You'll end up with the same `best.pt` file.

> If `best.pt` is missing, the app still runs — it falls back to generic YOLOv8n (which detects people/cars/etc., not PPE). You'll see a warning in the console. Drop in `best.pt` and restart to get PPE classes.

---

## Step 2 — Install dependencies

Requires **Python 3.9+**. A virtual environment is strongly recommended.

```bash
cd ppe-detector

# Create & activate a venv
python -m venv venv
# macOS / Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install
pip install -r requirements.txt
```

First install takes a few minutes because it pulls PyTorch.

---

## Step 3 — Run the app

```bash
python app.py
```

You'll see:
```
Loading model...
Loaded custom PPE model from .../best.pt
 * Running on http://127.0.0.1:5000
```

Open **http://127.0.0.1:5000** in your browser.

---

## How to use

1. Drag an image onto the drop zone (or click to browse).
2. Optionally tune the confidence threshold slider (default 0.25).
3. Click **Detect PPE**.
4. You'll see the image with bounding boxes, a summary of counts per class, and a detail table.

---

## Notes

- The app listens only on `127.0.0.1`, so it's **not reachable from other devices on your network** — exactly what you want for "local, no need to host."
- To share on your LAN, change `host="127.0.0.1"` to `host="0.0.0.0"` in `app.py`.
- On first run without a GPU, inference is a bit slow (1–3 s per image on CPU). With a CUDA-capable GPU, Ultralytics uses it automatically.
- Class names come from your trained model — they'll match whatever the Kaggle notebook's dataset used (typically `helmet`, `vest`, `no-helmet`, `no-vest`, `person`, etc.).

---

## Troubleshooting

**`ModuleNotFoundError: ultralytics`** — activate your venv and rerun `pip install -r requirements.txt`.

**Upload fails with `413`** — your image is over 16 MB. Bump `MAX_CONTENT_MB` in `app.py`.

**`best.pt` detected no objects** — try lowering the confidence threshold slider, or the model may not generalize to your image.

**OpenCV error on Linux** — install system libs: `sudo apt install libgl1 libglib2.0-0`.
