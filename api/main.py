"""FastAPI service for Spectra-AASIST3 deepfake detection."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.audio_preprocess import decode_audio_bytes
from src.config import (
    DEFAULT_DEMO_FAST_MODEL,
    DEFAULT_FAST_MODEL,
    DEFAULT_FAST_PROFILE_PATH,
    DEFAULT_SPECTRA_DECISION,
    SAMPLE_RATE,
)
from src.fast_baseline import prepare_waveform_fast
from src.fast_explain import FastFeatureProfiles, explain_fast_features
from src.inference import initialize_inference_context, predict_one_from_bytes
from transcriber import FastMilitaryTranscriber

_ctx = None
_profiles = None
_transcriber = None
_transcriber_error = None
_transcriber_lock = threading.Lock()
_backend = os.getenv("MODEL_BACKEND", "fast").strip().lower()
_fast_model_path = os.getenv(
    "FAST_MODEL_PATH",
    str(DEFAULT_DEMO_FAST_MODEL if DEFAULT_DEMO_FAST_MODEL.exists() else DEFAULT_FAST_MODEL),
)
_fast_profile_path = os.getenv("FAST_PROFILE_PATH", str(DEFAULT_FAST_PROFILE_PATH))
_spectra_decision = os.getenv("SPECTRA_DECISION", DEFAULT_SPECTRA_DECISION).strip().lower()
_transcriber_model_size = os.getenv("TRANSCRIBER_MODEL_SIZE", "tiny.en")
_transcriber_device = os.getenv("TRANSCRIBER_DEVICE", "cpu")
_transcriber_compute_type = os.getenv("TRANSCRIBER_COMPUTE_TYPE", "int8")
_transcriber_vad_filter = os.getenv("TRANSCRIBER_VAD_FILTER", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
_cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")


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
    threading.Thread(target=_warm_models, daemon=True).start()
    yield


app = FastAPI(
    title="Spectra-AASIST3 Deepfake Detector",
    version="1.0.0",
    lifespan=lifespan,
)
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
        "xai_available": bool(_ctx is not None and _ctx.backend == "fast" and _profiles is not None),
        "xai_scope": "fast MFCC/LFCC class-profile comparison only",
        "transcriber_available": _transcriber_error is None,
        "transcriber_loaded": _transcriber is not None,
        "transcriber_model_size": _transcriber_model_size,
        "transcriber_vad_filter": _transcriber_vad_filter,
        "transcriber_error": str(_transcriber_error) if _transcriber_error else None,
    }


@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
):
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
    if _ctx.backend == "fast":
        result["explanation"] = _build_fast_explanation(data, pred.label)
    else:
        result["explanation"] = None
    return result


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    custom_keywords: str = Form(""),
    deepfake_probability: float = Form(0.0),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    return _build_transcription_result(
        data,
        file.filename,
        custom_keywords,
        deepfake_probability=deepfake_probability,
    )


def _get_transcriber() -> FastMilitaryTranscriber:
    global _transcriber, _transcriber_error
    if _transcriber is not None:
        return _transcriber
    with _transcriber_lock:
        if _transcriber is not None:
            return _transcriber
        try:
            _transcriber = FastMilitaryTranscriber(
                model_size=_transcriber_model_size,
                device=_transcriber_device,
                compute_type=_transcriber_compute_type,
                vad_filter=_transcriber_vad_filter,
            )
            _transcriber_error = None
            return _transcriber
        except Exception as exc:
            _transcriber_error = exc
            raise


def _warm_models() -> None:
    if _ctx is not None and _ctx.backend == "fast":
        try:
            prepared = np.zeros(SAMPLE_RATE * 4, dtype=np.float32)
            _ctx.model.predict_one_from_prepared(prepared)  # type: ignore[union-attr]
            if _profiles is not None:
                features = _ctx.model.extract_features(prepared)  # type: ignore[union-attr]
                explain_fast_features(features, "bonafide", _profiles)
        except Exception:
            pass
    try:
        _get_transcriber()
    except Exception:
        pass


def _build_transcription_result(
    data: bytes,
    filename: str | None,
    custom_keywords: str,
    pred=None,
    deepfake_probability: float | None = None,
):
    suffix = Path(filename or "audio.wav").suffix or ".wav"
    if deepfake_probability is None and pred is not None:
        deepfake_probability = float(pred.score_spoof if pred.is_spoof else 1.0 - pred.confidence)
    spoof_probability = min(1.0, max(0.0, float(deepfake_probability or 0.0)))
    external_signals = {
        "deepfake_probability": spoof_probability,
    }

    try:
        transcriber = _get_transcriber()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            transcription = transcriber.transcribe(
                tmp_path,
                external_signals=external_signals,
                custom_keywords=custom_keywords,
            )
        finally:
            tmp_path.unlink(missing_ok=True)
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "custom_keywords": custom_keywords,
        }

    payload = asdict(transcription)
    payload.update(
        {
            "available": True,
            "processing_ms": round(transcription.processing_seconds * 1000.0, 2),
            "real_time_factor": (
                round(transcription.real_time_factor, 3)
                if transcription.real_time_factor is not None
                else None
            ),
            "custom_keywords": custom_keywords,
        }
    )
    return payload


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
