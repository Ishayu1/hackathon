"""FastAPI service for Spectra-AASIST3 deepfake detection."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import initialize_inference_context, predict_one_from_bytes

_ctx = None
_backend = os.getenv("MODEL_BACKEND", "spectra").strip().lower()
_fast_model_path = os.getenv("FAST_MODEL_PATH", "results/fast_baseline_mfcc_logistic_regression.joblib")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ctx
    _ctx = initialize_inference_context(
        backend="fast" if _backend == "fast" else "spectra",
        device="auto",
        fast_model_path=_fast_model_path,
    )
    yield


app = FastAPI(
    title="Spectra-AASIST3 Deepfake Detector",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok" if _ctx is not None else "loading",
        "model_loaded": _ctx is not None,
        "device": str(_ctx.device) if _ctx is not None else None,
        "backend": _backend,
    }


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    if _ctx is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        pred = predict_one_from_bytes(_ctx, data, mode="deterministic")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}") from exc

    result = asdict(pred)
    result.update(
        {
            "preprocess_ms": round(pred.preprocess_ms, 2),
            "feature_ms": round(pred.feature_ms, 2),
            "inference_ms": round(pred.inference_ms, 2),
            "total_ms": round(pred.total_ms, 2),
            "filename": file.filename,
            "backend": _backend,
        }
    )
    return result
