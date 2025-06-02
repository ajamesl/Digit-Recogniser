# model_service/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import torch.nn.functional as F
from model import MNIST_CNN
from PIL import Image
import io
import base64
import numpy as np

app = FastAPI(title="MNIST Model API")

# Define Pydantic schema for the input JSON
class PredictRequest(BaseModel):
    # The image will be sent as a base64‐encoded PNG (28×28, grayscale)
    image_b64: str

class PredictResponse(BaseModel):
    predicted: int
    confidence: float
    probabilities: List[float]

# Load model at startup
@app.on_event("startup")
def load_model():
    global model
    # Instantiate the network
    model = MNIST_CNN()
    # Load saved weights
    try:
        state_dict = torch.load("mnist_cnn.pth", map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Could not load model weights: {e}")

# Define the /predict endpoint
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Decode the base64 image
    try:
        image_bytes = base64.b64decode(req.image_b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("L")  # convert to single‐channel
        # Ensure it’s 28x28 (if not, resize)
        img = img.resize((28, 28))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    # Preprocess to a tensor of shape [1,1,28,28], normalized exactly as during training
    img_arr = np.array(img).astype(np.float32) / 255.0
    # MNIST normalization: mean=0.1307, std=0.3081
    img_arr = (img_arr - 0.1307) / 0.3081
    tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0)  # [1,1,28,28]

    with torch.no_grad():
        logits = model(tensor)                      # [1,10]
        probs = F.softmax(logits, dim=1).numpy().flatten().tolist()
        pred_index = int(np.argmax(probs))
        pred_conf = float(probs[pred_index])

    return PredictResponse(
        predicted=pred_index,
        confidence=pred_conf,
        probabilities=probs
    )