"""
Wildfire Detection API
-----------------------
FastAPI backend that wraps the trained Keras wildfire detection model.

Endpoints:
    GET  /health    -> simple health check
    POST /predict   -> upload an image, get wildfire / no-wildfire prediction

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

import io
import os

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = os.getenv("WILDFIRE_MODEL_PATH", "model/wildfire_model.h5")
IMG_SIZE = (224, 224)  # change to match the input size the model was trained on
CLASS_NAMES = ["no_wildfire", "wildfire"]  # adjust order to match training labels

app = FastAPI(
    title="Wildfire Detection API",
    description="Serves predictions from the ISRO wildfire detection CNN model",
    version="1.0.0",
)

# Allow a frontend/dashboard (built separately) to call this API during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None  # lazy-loaded so the API can start even before the model file exists


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"Model file not found at '{MODEL_PATH}'. "
                f"Set WILDFIRE_MODEL_PATH or place the .h5 file there.",
            )
        _model = load_model(MODEL_PATH)
    return _model


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    raw_score: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Load image bytes, resize, normalize, and add a batch dimension."""
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    image = image.resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape: (1, H, W, 3)
    return arr


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", model_loaded=_model is not None)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are supported.")

    model = get_model()
    file_bytes = await file.read()
    input_tensor = preprocess_image(file_bytes)

    prediction = model.predict(input_tensor)
    score = float(prediction[0][0])  # assumes sigmoid output, single unit

    label_index = 1 if score >= 0.5 else 0
    confidence = score if label_index == 1 else 1 - score

    return PredictionResponse(
        label=CLASS_NAMES[label_index],
        confidence=round(confidence, 4),
        raw_score=round(score, 4),
    )
