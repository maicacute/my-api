from flask import Flask, request, jsonify
import os
from datetime import datetime
import numpy as np
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# === Flask setup ===
app = Flask(__name__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# === Load trained model and class names ===
MODEL_PATH = "egg_model.keras"
CLASS_FILE = "class_names.txt"

print("🔄 Loading model...")

# Check kung meron ang files
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")
if not os.path.exists(CLASS_FILE):
    raise FileNotFoundError(f"❌ Class file not found: {CLASS_FILE}")

model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Load class names
with open(CLASS_FILE, "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]
print("Classes:", CLASS_NAMES)


@app.route("/api/infer", methods=["POST"])
def infer():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + file.filename
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis=0)

    # Predict
    prediction = model.predict(img_array)
    confidence = float(np.max(prediction))
    class_index = int(np.argmax(prediction))
    status = CLASS_NAMES[class_index]

    return jsonify({
        "status": status,
        "confidence": confidence,
        "filename": filename
    })


if __name__ == "__main__":
    print("🔍 Python version:", sys.version)
    app.run(host="0.0.0.0", port=5000, debug=True)
