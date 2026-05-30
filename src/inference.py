"""Inference helpers for Spectra-AASIST3 and fast classical baseline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from src.audio_preprocess import PreprocessMode, decode_audio_bytes, prepare_batch, prepare_waveform
from src.config import (
    CLIP_LEN,
    DEFAULT_FAST_MODEL,
    DEFAULT_SPECTRA_DECISION,
    DEFAULT_THRESHOLD,
)
from src.fast_baseline import FastAudioClassifier, prepare_waveform_fast
from src.model_loader import load_model, sync_device, warmup
from vendor.spectra_aasist3.model import SpectraAASIST3

BackendType = Literal["spectra", "fast"]
SpectraDecisionMode = Literal["threshold", "argmax"]


@dataclass
class Prediction:
    label: Literal["bonafide", "spoof"]
    is_spoof: bool
    score_spoof: float
    score_bonafide: float
    confidence: float
    threshold: float


@dataclass
class TimedPrediction(Prediction):
    preprocess_ms: float = 0.0
    feature_ms: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class InferenceContext:
    backend: BackendType
    model: Any
    device: torch.device | str


def logits_to_prediction(
    logits: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
    *,
    decision: SpectraDecisionMode = DEFAULT_SPECTRA_DECISION,
) -> Prediction:
    """
    Convert model logits to prediction.

    logits shape: (batch, 2) where index 0=spoof, 1=bonafide.
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    score_spoof = float(logits[0, 0].item())
    score_bonafide = float(logits[0, 1].item())
    probs = F.softmax(logits[0], dim=0)

    if decision == "argmax":
        is_bonafide = score_bonafide >= score_spoof
    else:
        is_bonafide = score_bonafide > threshold

    return Prediction(
        label="bonafide" if is_bonafide else "spoof",
        is_spoof=not is_bonafide,
        score_spoof=score_spoof,
        score_bonafide=score_bonafide,
        confidence=float(probs[1 if is_bonafide else 0].item()),
        threshold=threshold,
    )


def predict_batch(
    model: SpectraAASIST3,
    device: torch.device,
    waveforms: list[torch.Tensor],
    sample_rates: list[int],
    *,
    mode: PreprocessMode = "deterministic",
    threshold: float = DEFAULT_THRESHOLD,
    decision: SpectraDecisionMode = DEFAULT_SPECTRA_DECISION,
) -> tuple[list[Prediction], list[float]]:
    """
    Run batch inference.

    Returns predictions and per-sample inference latencies in ms.
    """
    batch = prepare_batch(waveforms, sample_rates, mode=mode).to(device)

    latencies_ms: list[float] = []
    with torch.inference_mode():
        sync_device(device)
        start = time.perf_counter()
        logits = model(batch)
        sync_device(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_sample_ms = elapsed_ms / max(len(waveforms), 1)

    predictions = [
        logits_to_prediction(logits[i : i + 1], threshold, decision=decision)
        for i in range(logits.size(0))
    ]
    latencies_ms = [per_sample_ms] * len(predictions)
    return predictions, latencies_ms


def predict_one_from_prepared(
    model: SpectraAASIST3,
    device: torch.device,
    prepared: torch.Tensor,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    decision: SpectraDecisionMode = DEFAULT_SPECTRA_DECISION,
) -> tuple[Prediction, float]:
    """Run inference on an already-prepared 1D waveform."""
    batch = prepared.unsqueeze(0).to(device)
    with torch.inference_mode():
        sync_device(device)
        start = time.perf_counter()
        logits = model(batch)
        sync_device(device)
        inference_ms = (time.perf_counter() - start) * 1000.0
    return logits_to_prediction(logits, threshold, decision=decision), inference_ms


def predict_one(
    model: SpectraAASIST3,
    device: torch.device,
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    mode: PreprocessMode = "deterministic",
    threshold: float = DEFAULT_THRESHOLD,
) -> TimedPrediction:
    """Full single-sample predict with timing breakdown."""
    t0 = time.perf_counter()
    prepared = prepare_waveform(waveform, sample_rate, mode=mode)
    preprocess_ms = (time.perf_counter() - t0) * 1000.0

    pred, inference_ms = predict_one_from_prepared(
        model, device, prepared, threshold=threshold
    )
    total_ms = preprocess_ms + inference_ms

    return TimedPrediction(
        label=pred.label,
        is_spoof=pred.is_spoof,
        score_spoof=pred.score_spoof,
        score_bonafide=pred.score_bonafide,
        confidence=pred.confidence,
        threshold=pred.threshold,
        preprocess_ms=preprocess_ms,
        inference_ms=inference_ms,
        total_ms=total_ms,
    )


def initialize_inference(device: str = "auto") -> tuple[SpectraAASIST3, torch.device]:
    """Load model and run warmup pass."""
    model, resolved = load_model(device)
    warmup(model, resolved)
    return model, resolved


def initialize_inference_context(
    *,
    backend: BackendType = "spectra",
    device: str = "auto",
    fast_model_path: str | Path | None = None,
) -> InferenceContext:
    """Initialize either Spectra-AASIST3 or fast baseline backend."""
    if backend == "fast":
        model_path = Path(fast_model_path or DEFAULT_FAST_MODEL)
        model = FastAudioClassifier.load(model_path)
        return InferenceContext(backend="fast", model=model, device="cpu")

    model, resolved = initialize_inference(device)
    return InferenceContext(backend="spectra", model=model, device=resolved)


def predict_one_from_bytes(
    ctx: InferenceContext,
    data: bytes,
    *,
    mode: PreprocessMode = "deterministic",
    spectra_threshold: float = DEFAULT_THRESHOLD,
    spectra_decision: SpectraDecisionMode = DEFAULT_SPECTRA_DECISION,
    fast_threshold: float = 0.5,
) -> TimedPrediction:
    """
    Unified single-file inference entrypoint.

    Handles decode + preprocess + backend-specific inference and returns consistent timings.
    """
    t0 = time.perf_counter()
    waveform, sample_rate = decode_audio_bytes(data)
    decode_ms = (time.perf_counter() - t0) * 1000.0

    if ctx.backend == "fast":
        t1 = time.perf_counter()
        prepared = prepare_waveform_fast(waveform, sample_rate)
        preprocess_ms = (time.perf_counter() - t1) * 1000.0
        pred, feature_ms, inference_ms = ctx.model.predict_one_from_prepared(  # type: ignore[union-attr]
            prepared, threshold=fast_threshold
        )
        total_ms = decode_ms + preprocess_ms + feature_ms + inference_ms
        return TimedPrediction(
            label=pred.label,
            is_spoof=pred.is_spoof,
            score_spoof=pred.score_spoof,
            score_bonafide=pred.score_bonafide,
            confidence=pred.confidence,
            threshold=pred.threshold,
            preprocess_ms=decode_ms + preprocess_ms,
            feature_ms=feature_ms,
            inference_ms=inference_ms,
            total_ms=total_ms,
        )

    t2 = time.perf_counter()
    prepared = prepare_waveform(waveform, sample_rate, mode=mode)
    preprocess_only_ms = (time.perf_counter() - t2) * 1000.0
    pred, inference_ms = predict_one_from_prepared(
        ctx.model,  # type: ignore[arg-type]
        ctx.device,  # type: ignore[arg-type]
        prepared,
        threshold=spectra_threshold,
        decision=spectra_decision,
    )
    total_preprocess_ms = decode_ms + preprocess_only_ms
    return TimedPrediction(
        label=pred.label,
        is_spoof=pred.is_spoof,
        score_spoof=pred.score_spoof,
        score_bonafide=pred.score_bonafide,
        confidence=pred.confidence,
        threshold=pred.threshold,
        preprocess_ms=total_preprocess_ms,
        feature_ms=0.0,
        inference_ms=inference_ms,
        total_ms=total_preprocess_ms + inference_ms,
    )


def bonafide_logit_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Detection score for EER: bonafide logit (index 1)."""
    return logits[:, 1]
