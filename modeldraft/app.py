"""
PPE Detection Web App - Local Flask Server
Runs a fine-tuned YOLOv8 model for Personal Protective Equipment detection.
"""

import os
import uuid
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2

# ---------- Config ----------
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_CONTENT_MB = 16

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

# Path to your trained model. Replace with the .pt file exported from the Kaggle
# notebook (usually named `best.pt`). If not present, we fall back to a generic
# pretrained YOLOv8n model so the app still runs.
MODEL_PATH = BASE_DIR / "best.pt"
FALLBACK_MODEL = "yolov8n.pt"  # auto-downloaded by ultralytics on first run

print("Loading model...")
if MODEL_PATH.exists():
    model = YOLO(str(MODEL_PATH))
    print(f"Loaded custom PPE model from {MODEL_PATH}")
else:
    model = YOLO(FALLBACK_MODEL)
    print(f"WARNING: {MODEL_PATH} not found. Using fallback {FALLBACK_MODEL}.")
    print("         Download best.pt from the Kaggle notebook output and place it next to app.py.")

# ---------- Flask app ----------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image part in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    # Save the upload with a unique name
    ext = file.filename.rsplit(".", 1)[1].lower()
    uid = uuid.uuid4().hex[:12]
    saved_name = f"{uid}.{ext}"
    saved_path = UPLOAD_FOLDER / saved_name
    file.save(saved_path)

    # Confidence threshold from form, default 0.25
    try:
        conf = float(request.form.get("conf", 0.25))
    except ValueError:
        conf = 0.25
    conf = max(0.05, min(conf, 0.95))

    # Run inference
    results = model.predict(source=str(saved_path), conf=conf, verbose=False)
    r = results[0]

    # Draw boxes on the image and save to static/results
    annotated = r.plot()  # BGR numpy array with boxes/labels drawn
    result_name = f"{uid}_annotated.jpg"
    result_path = RESULT_FOLDER / result_name
    cv2.imwrite(str(result_path), annotated)

    # Build detection summary
    detections = []
    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls.item())
            label = model.names.get(cls_id, str(cls_id))
            detections.append(
                {
                    "label": label,
                    "confidence": round(float(box.conf.item()), 3),
                    "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
                }
            )

    # Count per class for the summary card
    counts = {}
    for d in detections:
        counts[d["label"]] = counts.get(d["label"], 0) + 1

    return jsonify(
        {
            "result_image_url": f"/static/results/{result_name}",
            "detections": detections,
            "counts": counts,
            "total": len(detections),
        }
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    # host=127.0.0.1 means local-only, which is what you asked for
    app.run(host="127.0.0.1", port=5000, debug=True)
