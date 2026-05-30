"""FastAPI service for Spectra-AASIST3 deepfake detection."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.audio_preprocess import load_audio_from_bytes
from src.inference import initialize_inference, predict_one_from_prepared
from src.model_loader import get_loaded_model

_model = None
_device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _device
    _model, _device = initialize_inference("auto")
    yield


app = FastAPI(
    title="Spectra-AASIST3 Deepfake Detector",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    try:
        model, device = get_loaded_model()
        return {
            "status": "ok",
            "model_loaded": model is not None,
            "device": str(device),
        }
    except RuntimeError:
        return {
            "status": "loading",
            "model_loaded": False,
            "device": None,
        }


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    if _model is None or _device is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    import time

    t0 = time.perf_counter()
    try:
        prepared = load_audio_from_bytes(data, mode="deterministic")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {exc}") from exc
    preprocess_ms = (time.perf_counter() - t0) * 1000.0

    pred, inference_ms = predict_one_from_prepared(_model, _device, prepared)
    total_ms = preprocess_ms + inference_ms

    result = asdict(pred)
    result.update(
        {
            "preprocess_ms": round(preprocess_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "total_ms": round(total_ms, 2),
            "filename": file.filename,
        }
    )
    return result
