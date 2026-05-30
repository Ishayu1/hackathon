"""FastAPI service for Spectra-AASIST3 deepfake detection."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_DEMO_FAST_MODEL, DEFAULT_FAST_MODEL, DEFAULT_SPECTRA_DECISION
from src.inference import initialize_inference_context, predict_one_from_bytes

_ctx = None
_backend = os.getenv("MODEL_BACKEND", "fast").strip().lower()
_fast_model_path = os.getenv(
    "FAST_MODEL_PATH",
    str(DEFAULT_DEMO_FAST_MODEL if DEFAULT_DEMO_FAST_MODEL.exists() else DEFAULT_FAST_MODEL),
)
_spectra_decision = os.getenv("SPECTRA_DECISION", DEFAULT_SPECTRA_DECISION).strip().lower()


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

_cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok" if _ctx is not None else "loading",
        "model_loaded": _ctx is not None,
        "device": str(_ctx.device) if _ctx is not None else None,
        "backend": _backend,
        "spectra_decision": _spectra_decision if _backend == "spectra" else None,
        "fast_model_path": _fast_model_path if _backend == "fast" else None,
    }


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    if _ctx is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        pred = predict_one_from_bytes(
            _ctx,
            data,
            mode="deterministic",
            spectra_decision=_spectra_decision,  # type: ignore[arg-type]
        )
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
            "spectra_decision": _spectra_decision if _backend == "spectra" else None,
        }
    )
    return result
