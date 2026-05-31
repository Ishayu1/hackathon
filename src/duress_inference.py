"""Wav2Vec2 + BiLSTM acoustic duress detection."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

from src.audio_preprocess import decode_audio_bytes, resample_if_needed, to_mono
from src.config import DEFAULT_DURESS_MODEL, DEFAULT_DURESS_THRESHOLD, SAMPLE_RATE

_model = None
_model_error: Exception | None = None
_model_lock = threading.Lock()

# Matches temporal_bilstm_duress.pth: BiLSTM hidden=128, single-logit classifier head.
DEFAULT_DURESS_HIDDEN_DIM = 128


class Wav2Vec2BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        wav2vec2_model_name: str = "facebook/wav2vec2-base",
        hidden_dim: int = DEFAULT_DURESS_HIDDEN_DIM,
    ):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(wav2vec2_model_name)
        self.lstm = nn.LSTM(
            input_size=self.wav2vec2.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.wav2vec2(waveforms).last_hidden_state
        lstm_out, _ = self.lstm(embeddings)
        pooled_out = torch.mean(lstm_out, dim=1)
        return self.classifier(pooled_out)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class DuressPrediction:
    available: bool
    label: str
    is_duress: bool
    probability: float
    probability_percent: float
    score_normal: float
    score_duress: float
    threshold: float
    inference_ms: float
    error: str | None = None


def get_duress_model_status() -> tuple[bool, str | None]:
    """Return (loaded, error_message)."""
    if _model is not None:
        return True, None
    return False, str(_model_error) if _model_error else None


def _load_model(weights_path: Path, device: torch.device) -> Wav2Vec2BiLSTMClassifier:
    if not weights_path.exists():
        raise FileNotFoundError(f"Duress model weights not found: {weights_path}")

    model = Wav2Vec2BiLSTMClassifier().to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    # Checkpoint stores LSTM + classifier only; Wav2Vec2 stays from HuggingFace.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    expected_prefixes = ("lstm.", "classifier.")
    missing_trainable = [key for key in missing if key.startswith(expected_prefixes)]
    if missing_trainable:
        raise RuntimeError(f"Duress checkpoint missing expected layers: {missing_trainable[:5]}")
    if unexpected:
        raise RuntimeError(f"Duress checkpoint has unexpected layers: {unexpected[:5]}")
    model.eval()
    return model


def get_duress_model(weights_path: Path | str | None = None) -> Wav2Vec2BiLSTMClassifier:
    global _model, _model_error

    if _model is not None:
        return _model

    path = Path(weights_path) if weights_path else DEFAULT_DURESS_MODEL
    with _model_lock:
        if _model is not None:
            return _model
        try:
            device = resolve_device()
            _model = _load_model(path, device)
            _model_error = None
            return _model
        except Exception as exc:
            _model_error = exc
            raise


def predict_duress_from_bytes(
    data: bytes,
    *,
    weights_path: Path | str | None = None,
    threshold: float = DEFAULT_DURESS_THRESHOLD,
) -> DuressPrediction:
    """Run acoustic duress inference on raw audio bytes."""
    try:
        model = get_duress_model(weights_path)
    except Exception as exc:
        return DuressPrediction(
            available=False,
            label="unavailable",
            is_duress=False,
            probability=0.0,
            probability_percent=0.0,
            score_normal=0.0,
            score_duress=0.0,
            threshold=threshold,
            inference_ms=0.0,
            error=str(exc),
        )

    device = next(model.parameters()).device
    t0 = time.perf_counter()

    try:
        waveform, sample_rate = decode_audio_bytes(data)
        waveform = to_mono(waveform)
        waveform = resample_if_needed(waveform, sample_rate)
        waveform = waveform.unsqueeze(0).to(device)

        with torch.no_grad():
            logit = model(waveform).squeeze()
            score_duress = float(torch.sigmoid(logit).item())

        score_normal = 1.0 - score_duress
        is_duress = score_duress >= threshold
        inference_ms = (time.perf_counter() - t0) * 1000.0

        return DuressPrediction(
            available=True,
            label="duress" if is_duress else "normal",
            is_duress=is_duress,
            probability=score_duress,
            probability_percent=round(score_duress * 100.0, 2),
            score_normal=score_normal,
            score_duress=score_duress,
            threshold=threshold,
            inference_ms=round(inference_ms, 2),
        )
    except Exception as exc:
        return DuressPrediction(
            available=False,
            label="unavailable",
            is_duress=False,
            probability=0.0,
            probability_percent=0.0,
            score_normal=0.0,
            score_duress=0.0,
            threshold=threshold,
            inference_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            error=str(exc),
        )


def analyze_duress_probability(
    audio_path: str | Path,
    model_weights_path: str | Path | None = None,
    target_sample_rate: int = SAMPLE_RATE,
) -> float:
    """Legacy file-path API: returns duress probability as a percentage."""
    path = Path(audio_path)
    data = path.read_bytes()
    result = predict_duress_from_bytes(data, weights_path=model_weights_path)
    if not result.available:
        raise RuntimeError(result.error or "Duress inference unavailable")
    _ = target_sample_rate
    return result.probability_percent
