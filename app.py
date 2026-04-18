import csv
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
PLATES_DIR = BASE_DIR / "plates"
LOG_TXT = BASE_DIR / "car_plate_data.txt"
LOG_CSV = BASE_DIR / "car_plate_data.csv"
MODEL_PATH = BASE_DIR / "best.pt"
CLASS_FILE = BASE_DIR / "coco1.txt"
DATASET_SAMPLE_DIR = BASE_DIR / "Dataset" / "images" / "validation"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PLATE_PATTERN = re.compile(r"^[A-Z0-9]{4,10}$")


os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / "Ultralytics"))

if os.name == "nt":
    windows_tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if windows_tesseract.exists():
        pytesseract.pytesseract.tesseract_cmd = str(windows_tesseract)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "alpr-demo-secret")


_model = None


def ensure_directories() -> None:
    for directory in (UPLOAD_DIR, RESULT_DIR, PLATES_DIR, Path(os.environ["YOLO_CONFIG_DIR"])):
        directory.mkdir(parents=True, exist_ok=True)


def load_model():
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO(str(MODEL_PATH))
    return _model


def load_class_names() -> list[str]:
    if not CLASS_FILE.exists():
        return ["numberplate"]
    return [line.strip() for line in CLASS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


CLASS_NAMES = load_class_names()


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def sanitize_plate_text(text: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    return cleaned


def append_plate_log(plate_text: str) -> None:
    timestamp = datetime.now()
    LOG_TXT.parent.mkdir(parents=True, exist_ok=True)

    if not LOG_TXT.exists():
        LOG_TXT.write_text("NumberPlate\tDate\tTime\n", encoding="utf-8")

    write_csv_header = not LOG_CSV.exists() or LOG_CSV.stat().st_size == 0

    with LOG_TXT.open("a", encoding="utf-8") as txt_file:
        txt_file.write(f"{plate_text}\t{timestamp:%Y-%m-%d}\t{timestamp:%H:%M:%S}\n")

    with LOG_CSV.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if write_csv_header:
            writer.writerow(["NumberPlate", "Date", "Time"])
        writer.writerow([plate_text, f"{timestamp:%Y-%m-%d}", f"{timestamp:%H:%M:%S}"])


def run_ocr(crop: np.ndarray) -> str:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 10, 20, 20)
    text = pytesseract.image_to_string(gray, config="--psm 7")
    return sanitize_plate_text(text)


def detect_on_frame(frame: np.ndarray, source_stem: str, logged_plates: set[str]) -> tuple[np.ndarray, list[dict]]:
    results = load_model().predict(frame, verbose=False)
    detections: list[dict] = []

    if not results or results[0].boxes is None or results[0].boxes.data is None:
        return frame, detections

    for index, box in enumerate(results[0].boxes.data.tolist(), start=1):
        x1, y1, x2, y2, confidence, class_id = box[:6]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        plate_text = run_ocr(crop)
        is_valid_plate = bool(PLATE_PATTERN.match(plate_text))
        class_name = CLASS_NAMES[int(class_id)] if int(class_id) < len(CLASS_NAMES) else "numberplate"
        label_text = f"{class_name} {confidence:.2f}"

        if is_valid_plate:
            label_text = f"{plate_text} {confidence:.2f}"
            if plate_text not in logged_plates:
                logged_plates.add(plate_text)
                append_plate_log(plate_text)
                crop_name = f"{source_stem}_{plate_text}_{index}.jpg"
                cv2.imwrite(str(PLATES_DIR / crop_name), crop)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        detections.append(
            {
                "plate_text": plate_text or "Unreadable",
                "confidence": round(float(confidence), 3),
                "class_name": class_name,
                "valid_plate": is_valid_plate,
            }
        )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame, detections


def process_image(file_path: Path) -> dict:
    image = cv2.imread(str(file_path))
    if image is None:
        raise ValueError("Could not read the uploaded image.")

    annotated, detections = detect_on_frame(image, file_path.stem, set())
    output_name = f"{file_path.stem}_result.jpg"
    output_path = RESULT_DIR / output_name
    cv2.imwrite(str(output_path), annotated)

    return {
        "media_type": "image",
        "output_name": output_name,
        "detections": detections,
    }


def process_video(file_path: Path) -> dict:
    capture = cv2.VideoCapture(str(file_path))
    if not capture.isOpened():
        raise ValueError("Could not open the uploaded video.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    output_name = f"{file_path.stem}_result.mp4"
    output_path = RESULT_DIR / output_name
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_index = 0
    detection_summary: list[dict] = []
    logged_plates: set[str] = set()

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_index += 1
        annotated = frame
        if frame_index % 5 == 0:
            annotated, detections = detect_on_frame(frame.copy(), file_path.stem, logged_plates)
            detection_summary.extend(detections)
        writer.write(annotated)

    capture.release()
    writer.release()

    return {
        "media_type": "video",
        "output_name": output_name,
        "detections": detection_summary,
    }


def create_synthetic_demo() -> dict:
    canvas = np.full((540, 960, 3), (236, 231, 223), dtype=np.uint8)
    cv2.rectangle(canvas, (110, 220), (850, 400), (64, 82, 99), -1)
    cv2.rectangle(canvas, (170, 170), (710, 255), (88, 110, 132), -1)
    cv2.circle(canvas, (250, 400), 56, (25, 25, 25), -1)
    cv2.circle(canvas, (710, 400), 56, (25, 25, 25), -1)
    cv2.rectangle(canvas, (380, 300), (610, 360), (242, 233, 191), -1)
    cv2.rectangle(canvas, (380, 300), (610, 360), (30, 41, 59), 2)
    plate_text = "TS09AB1234"
    cv2.putText(canvas, plate_text, (395, 340), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    cv2.rectangle(canvas, (380, 300), (610, 360), (0, 255, 0), 2)
    cv2.putText(canvas, f"{plate_text} 0.99", (380, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(canvas, "Bundled web demo preview", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (15, 118, 110), 2)

    output_name = "demo_result.jpg"
    output_path = RESULT_DIR / output_name
    cv2.imwrite(str(output_path), canvas)

    return {
        "media_type": "image",
        "output_name": output_name,
        "detections": [
            {
                "plate_text": plate_text,
                "confidence": 0.99,
                "class_name": "numberplate",
                "valid_plate": True,
            }
        ],
        "demo_mode": "synthetic",
    }


def process_demo() -> dict:
    ensure_directories()
    if DATASET_SAMPLE_DIR.exists():
        for extension in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            sample_files = sorted(DATASET_SAMPLE_DIR.glob(extension))
            if sample_files:
                result = process_image(sample_files[0])
                result["demo_mode"] = "dataset"
                return result
    return create_synthetic_demo()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    upload = request.files.get("file")
    if upload is None or upload.filename == "":
        flash("Choose an image or video file to analyze.")
        return redirect(url_for("index"))

    if not allowed_file(upload.filename):
        flash("Unsupported file type. Upload an image or video.")
        return redirect(url_for("index"))

    ensure_directories()

    extension = Path(upload.filename).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{extension}"
    upload_path = UPLOAD_DIR / unique_name
    upload.save(upload_path)

    try:
        if extension in IMAGE_EXTENSIONS:
            result = process_image(upload_path)
        else:
            result = process_video(upload_path)
    except Exception as exc:
        flash(f"Processing failed: {exc}")
        return redirect(url_for("index"))

    return render_template(
        "index.html",
        result=result,
        result_url=url_for("download_result", filename=result["output_name"]),
    )


@app.route("/demo", methods=["POST"])
def demo():
    result = process_demo()
    return render_template(
        "index.html",
        result=result,
        result_url=url_for("download_result", filename=result["output_name"]),
    )


@app.route("/results/<path:filename>", methods=["GET"])
def download_result(filename: str):
    return send_from_directory(RESULT_DIR, filename)


if __name__ == "__main__":
    ensure_directories()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
