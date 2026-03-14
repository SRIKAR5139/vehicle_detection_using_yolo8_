from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)

# Load trained YOLO model
model = YOLO("./runs/detect/train7/weights/best.pt")

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static"

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result_img = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            # Save uploaded image
            upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(upload_path)

            # Run YOLO detection
            results = model(upload_path, conf=0.25)

            # Save result image
            result_name = f"result_{uuid.uuid4().hex}.jpg"
            result_path = os.path.join(RESULT_FOLDER, result_name)

            results[0].save(filename=result_path)

            result_img = result_name

    return render_template("index.html", result_img=result_img)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)