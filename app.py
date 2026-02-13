from flask import Flask, render_template, request, url_for
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # prevent caching

# Stats
stats = {
    "total_uploads": 0,
    "total_vehicles": 0,
    "high_congestion": 0
}

# Model
model = YOLO("yolov8n.pt")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

vehicle_classes = ["car", "truck", "bus", "motorbike"]


def detect_image(path):
    results = model(path)
    count = 0

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = model.names[cls]

            if conf > 0.5 and name in vehicle_classes:
                count += 1

    return count, results[0].plot()


def congestion_logic(count):
    if count < 5:
        return "LOW", 30
    elif count < 15:
        return "MEDIUM", 50
    else:
        return "HIGH", 80


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            save_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(save_path)

            count, img = detect_image(save_path)
            congestion, time = congestion_logic(count)

            # Save output
            output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")
            cv2.imwrite(output_path, img)

            # IMPORTANT (for HTML)
            image_path = "uploads/output.jpg"

            # Stats update
            stats["total_uploads"] += 1
            stats["total_vehicles"] += count
            if congestion == "HIGH":
                stats["high_congestion"] += 1

            result = {
                "count": count,
                "congestion": congestion,
                "time": time
            }

    return render_template("index.html", result=result, image=image_path, stats=stats)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

