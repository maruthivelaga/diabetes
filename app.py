from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import uvicorn

app = FastAPI()

# Load model
model = torch.load("model.pkl", map_location="cpu")
model.eval()

# Define preprocessing (adjust to your model’s training setup)
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # match training image size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

@app.get("/")
def home():
    return {"message": "Retina Model API is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Preprocess
        img_tensor = transform(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)

        return JSONResponse({"prediction": int(predicted.item())})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render uses $PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
