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

from src.audio_preprocess import decode_audio_bytes
from src.config import DEFAULT_FAST_MODEL, DEFAULT_FAST_PROFILE_PATH
from src.fast_baseline import prepare_waveform_fast
from src.fast_explain import FastFeatureProfiles, explain_fast_features
from src.inference import initialize_inference_context, predict_one_from_bytes

_ctx = None
_profiles = None
_backend = os.getenv("MODEL_BACKEND", "fast").strip().lower()
_fast_model_path = os.getenv("FAST_MODEL_PATH", str(DEFAULT_FAST_MODEL))
_fast_profile_path = os.getenv("FAST_PROFILE_PATH", str(DEFAULT_FAST_PROFILE_PATH))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ctx, _profiles
    _ctx = initialize_inference_context(
        backend="fast" if _backend == "fast" else "spectra",
        device="auto",
        fast_model_path=_fast_model_path,
    )
    if _ctx.backend == "fast" and Path(_fast_profile_path).exists():
        _profiles = FastFeatureProfiles.load(_fast_profile_path)
    yield


app = FastAPI(
    title="Spectra-AASIST3 Deepfake Detector",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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
        "xai_available": bool(_ctx is not None and _ctx.backend == "fast" and _profiles is not None),
        "xai_scope": "fast MFCC/LFCC class-profile comparison only",
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
    if _ctx.backend == "fast":
        result["explanation"] = _build_fast_explanation(data, pred.label)
    else:
        result["explanation"] = None
    return result


def _build_fast_explanation(data: bytes, label: str):
    if _profiles is None:
        return {
            "method": "unavailable",
            "note": "Interpretability available for fast MFCC model only when feature profiles are present.",
        }

    waveform, sample_rate = decode_audio_bytes(data)
    prepared = prepare_waveform_fast(waveform, sample_rate)
    features = _ctx.model.extract_features(prepared)  # type: ignore[union-attr]
    return explain_fast_features(features, label, _profiles)
