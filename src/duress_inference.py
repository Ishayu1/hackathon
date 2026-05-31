"""Wav2Vec2 + TemporalBiLSTM acoustic duress detection.

Architecture matches CNN_Classification.py exactly:
  - Wav2Vec2Processor + Wav2Vec2Model ("facebook/wav2vec2-base-960h")
    run separately to produce (1, 149, 768) embeddings
  - TemporalBiLSTM: LSTM(768->128 hidden, 2 layers, dropout=0.4, bidirectional)
    + global max pooling + Linear(256, 1)
  - BCEWithLogitsLoss at train time -> torch.sigmoid at inference
  - Audio normalised to 16 kHz mono, truncated/zero-padded to exactly 3 s
    (48 000 samples) before encoding — same as process_audio() in training

The .pth file contains ONLY TemporalBiLSTM weights (lstm.* and classifier.*).
Wav2Vec2 weights are always loaded fresh from HuggingFace.
"""

from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from src.config import DEFAULT_DURESS_MODEL, DEFAULT_DURESS_THRESHOLD, SAMPLE_RATE

# ---------------------------------------------------------------------------
# Constants — must match process_audio() in CNN_Classification.py exactly
# ---------------------------------------------------------------------------
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base-960h"
TARGET_SR = 16000           # 16 kHz
TARGET_SAMPLES = TARGET_SR * 3  # 48 000 — exactly 3 seconds

# ---------------------------------------------------------------------------
# Module-level singletons (lazy-loaded, thread-safe)
# ---------------------------------------------------------------------------
_bilstm: "TemporalBiLSTM | None" = None
_wav2vec2: "Wav2Vec2Model | None" = None
_processor: "Wav2Vec2Processor | None" = None
_model_error: Exception | None = None
_load_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 1. Architecture — verbatim copy of TemporalBiLSTM from CNN_Classification.py
# ---------------------------------------------------------------------------
class TemporalBiLSTM(nn.Module):
    def __init__(self):
        super(TemporalBiLSTM, self).__init__()
        # Input shape from Wav2Vec2: (Batch, Sequence=149, Features=768)
        # 128 hidden units * 2 directions = 256 output features per time step
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.4,
            bidirectional=True,
        )
        # Classification head mapping the 256 features to a binary logit
        self.classifier = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, 149, 768)
        lstm_out, _ = self.lstm(x)                    # (Batch, 149, 256)
        features, _ = torch.max(lstm_out, dim=1)      # (Batch, 256) — global max pool
        return self.classifier(features)              # (Batch, 1)


# ---------------------------------------------------------------------------
# 2. Device resolution
# ---------------------------------------------------------------------------
def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 3. Model loading
# ---------------------------------------------------------------------------
def _load_all_models(weights_path: Path, device: torch.device):
    """Load Wav2Vec2 (HuggingFace) + TemporalBiLSTM (.pth) onto device."""
    global _bilstm, _wav2vec2, _processor

    if not weights_path.exists():
        raise FileNotFoundError(f"Duress model weights not found: {weights_path}")

    # --- Wav2Vec2 (feature extractor) ---
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL_NAME)
    wav2vec2 = Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL_NAME).to(device)
    wav2vec2.eval()

    # --- TemporalBiLSTM (classifier) ---
    bilstm = TemporalBiLSTM().to(device)

    # The .pth was saved with torch.save(model.state_dict(), path) where
    # model is TemporalBiLSTM, so keys are exactly "lstm.*" and "classifier.*".
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    # Sanity-check: every key must belong to lstm or classifier
    unexpected = [k for k in state_dict if not (k.startswith("lstm.") or k.startswith("classifier."))]
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys in checkpoint (expected only lstm.* / classifier.*): {unexpected[:5]}"
        )

    missing, unexpected_load = bilstm.load_state_dict(state_dict, strict=True)
    if missing:
        raise RuntimeError(f"Checkpoint is missing required keys: {missing[:5]}")

    bilstm.eval()

    _processor = processor
    _wav2vec2 = wav2vec2
    _bilstm = bilstm


def _ensure_models_loaded(weights_path: Path) -> None:
    global _model_error
    if _bilstm is not None:
        return
    with _load_lock:
        if _bilstm is not None:
            return
        try:
            device = _resolve_device()
            _load_all_models(weights_path, device)
            _model_error = None
        except Exception as exc:
            _model_error = exc
            raise


def get_duress_model_status() -> tuple[bool, str | None]:
    """Return (loaded, error_message)."""
    if _bilstm is not None:
        return True, None
    return False, str(_model_error) if _model_error else None


# ---------------------------------------------------------------------------
# 4. Audio preprocessing — matches process_audio() in CNN_Classification.py
# ---------------------------------------------------------------------------
def _preprocess_waveform(data: bytes) -> torch.Tensor:
    """
    Decode bytes -> mono -> 16 kHz -> truncate/pad to exactly 3 s.
    Returns a 1-D float32 tensor of length TARGET_SAMPLES (48 000).
    """
    audio_data, file_sr = sf.read(io.BytesIO(data), dtype="float32")
    waveform = torch.tensor(audio_data, dtype=torch.float32)

    # Mono — matches: if waveform.ndim > 1: waveform = torch.mean(waveform, dim=1)
    if waveform.ndim > 1:
        waveform = torch.mean(waveform, dim=1)

    # Resample if needed
    if file_sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=file_sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    # Static truncation / zero-padding to exactly 3 seconds
    # Matches: waveform = waveform[:max_len]  /  F.pad(waveform, (0, pad_amount))
    if waveform.shape[0] > TARGET_SAMPLES:
        waveform = waveform[:TARGET_SAMPLES]
    else:
        pad_amount = TARGET_SAMPLES - waveform.shape[0]
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

    return waveform  # shape: (48000,)


# ---------------------------------------------------------------------------
# 5. Prediction dataclass + main inference function
# ---------------------------------------------------------------------------
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
def get_duress_model(weights_path: "Path | str | None" = None) -> None:
    """Warmup entrypoint called by api/main.py on startup.
    Triggers lazy loading of Wav2Vec2 + TemporalBiLSTM so the first
    real request doesn't pay the load cost."""
    path = Path(weights_path) if weights_path else DEFAULT_DURESS_MODEL
    _ensure_models_loaded(path)


def predict_duress_from_bytes(
    data: bytes,
    *,
    weights_path: "Path | str | None" = None,
    threshold: float = DEFAULT_DURESS_THRESHOLD,
) -> DuressPrediction:
    """Run acoustic duress inference on raw audio bytes."""
    path = Path(weights_path) if weights_path else DEFAULT_DURESS_MODEL

    try:
        _ensure_models_loaded(path)
    except Exception as exc:
        return DuressPrediction(
            available=False, label="unavailable", is_duress=False,
            probability=0.0, probability_percent=0.0,
            score_normal=0.0, score_duress=0.0,
            threshold=threshold, inference_ms=0.0, error=str(exc),
        )

    device = next(_bilstm.parameters()).device  # type: ignore[union-attr]
    t0 = time.perf_counter()

    try:
        # Step 1: preprocess to fixed-length waveform (matches process_audio)
        waveform = _preprocess_waveform(data)  # (48000,)

        # Step 2: Wav2Vec2 feature extraction (matches training exactly)
        with torch.no_grad():
            inputs = _processor(          # type: ignore[union-attr]
                waveform,
                return_tensors="pt",
                sampling_rate=TARGET_SR,
            ).input_values.to(device)
            embeddings = _wav2vec2(inputs).last_hidden_state  # type: ignore[union-attr]
            # embeddings shape: (1, 149, 768)

        # Step 3: TemporalBiLSTM classification
        with torch.no_grad():
            logit = _bilstm(embeddings).squeeze()  # type: ignore[union-attr]
            # logit is a scalar; sigmoid gives P(duress)
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
            available=False, label="unavailable", is_duress=False,
            probability=0.0, probability_percent=0.0,
            score_normal=0.0, score_duress=0.0,
            threshold=threshold,
            inference_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# 6. Legacy file-path API (keeps api/main.py happy)
# ---------------------------------------------------------------------------
def analyze_duress_probability(
    audio_path: "str | Path",
    model_weights_path: "str | Path | None" = None,
    target_sample_rate: int = SAMPLE_RATE,
) -> float:
    """Legacy file-path API: returns duress probability as a percentage."""
    data = Path(audio_path).read_bytes()
    result = predict_duress_from_bytes(data, weights_path=model_weights_path)
    if not result.available:
        raise RuntimeError(result.error or "Duress inference unavailable")
    _ = target_sample_rate
    return result.probability_percent