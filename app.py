from flask import Flask, render_template, request
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Lazy load model (loads only when first detection happens)
model = None

def get_model():
    global model
    if model is None:
        print("Loading YOLO model...")
        model = YOLO("runs/detect/train7/weights/best.pt")
    return model


@app.route("/", methods=["GET", "POST"])
def index():
    result_img = None

    if request.method == "POST":

        if "image" not in request.files:
            return render_template("index.html", result_img=None)

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html", result_img=None)

        filename = secure_filename(file.filename)

        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        # Load model only when needed
        model = get_model()

        results = model(upload_path, conf=0.25)

        result_name = f"result_{uuid.uuid4().hex}.jpg"
        result_path = os.path.join(RESULT_FOLDER, result_name)

        results[0].save(filename=result_path)

        result_img = result_name

    return render_template("index.html", result_img=result_img)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)