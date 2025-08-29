import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import shutil
import os
import json
import numpy as np
import tensorflow as tf

app = FastAPI()

# Ensure uploads directory exists
os.makedirs("uploads", exist_ok=True)

# Load TFLite model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.tflite")
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TFLite model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load TFLite model: {e}")

# Load class labels from labels.txt
labels_path = os.path.join(BASE_DIR, "label.txt")
try:
    with open(labels_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"✅ Loaded {len(class_names)} class labels.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load labels.txt: {e}")

# Load remedies (optional)
remedies_path = os.path.join(BASE_DIR, "remedies.json")
if os.path.exists(remedies_path):
    with open(remedies_path, "r") as f:
        remedies = json.load(f)
    print("✅ Remedies loaded.")
else:
    remedies = {}
    print("⚠️ remedies.json not found. Proceeding without remedies.")

# Simple preprocessing function
def preprocess_image(img_path, target_size=(224, 224)):
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # normalize
    return np.expand_dims(img_array, axis=0).astype(np.float32)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_path = f"uploads/{file.filename}"
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        img_array = preprocess_image(img_path)
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        class_index = int(np.argmax(predictions))
        class_name = class_names[class_index]
        confidence = float(predictions[class_index])
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Prediction failed: {str(e)}"})

    return {
        "class": class_name,
        "confidence": round(confidence, 3),
        "remedy": remedies.get(class_name, "No remedy found.")
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render provides PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
