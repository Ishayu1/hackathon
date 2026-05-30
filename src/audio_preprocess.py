"""Audio loading and preprocessing for Spectra-AASIST3."""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.config import CLIP_LEN, SAMPLE_RATE

PreprocessMode = Literal["deterministic", "random"]


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert (channels, samples) or (samples,) to 1D mono tensor."""
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim == 2:
        if waveform.shape[0] <= waveform.shape[1]:
            return waveform.mean(dim=0)
        return waveform.mean(dim=1)
    raise ValueError(f"Unexpected waveform shape: {tuple(waveform.shape)}")


def resample_if_needed(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if sample_rate == SAMPLE_RATE:
        return waveform
    return torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)


def pad_random(x: torch.Tensor, max_len: int = CLIP_LEN) -> torch.Tensor:
    """Match Spectra-AASIST3 README pad_random."""
    if x.ndim > 1:
        x = x.squeeze()
    x_len = x.shape[0]
    if x_len >= max_len:
        start = random.randint(0, x_len - max_len)
        return x[start : start + max_len]
    num_repeats = int(max_len / x_len) + 1
    return x.repeat(num_repeats)[:max_len]


def pad_deterministic(x: torch.Tensor, max_len: int = CLIP_LEN) -> torch.Tensor:
    """Center-crop or zero-pad to fixed length."""
    if x.ndim > 1:
        x = x.squeeze()
    x_len = x.shape[0]
    if x_len >= max_len:
        start = (x_len - max_len) // 2
        return x[start : start + max_len]
    return torch.nn.functional.pad(x, (0, max_len - x_len))


def apply_preemphasis(waveform: torch.Tensor) -> torch.Tensor:
    """Preemphasis on 1D waveform; returns 1D tensor."""
    return torchaudio.functional.preemphasis(waveform.unsqueeze(0)).squeeze(0)


def prepare_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    mode: PreprocessMode = "deterministic",
) -> torch.Tensor:
    """
    Full preprocessing pipeline: mono -> resample -> preemphasis -> pad/crop.

    Returns 1D float tensor of length CLIP_LEN.
    """
    waveform = to_mono(waveform.float())
    waveform = resample_if_needed(waveform, sample_rate)
    waveform = apply_preemphasis(waveform)
    if mode == "random":
        return pad_random(waveform, CLIP_LEN)
    return pad_deterministic(waveform, CLIP_LEN)


def prepare_batch(
    waveforms: list[torch.Tensor],
    sample_rates: list[int],
    *,
    mode: PreprocessMode = "deterministic",
) -> torch.Tensor:
    """Prepare a batch of waveforms; returns (batch, CLIP_LEN)."""
    processed = [
        prepare_waveform(w, sr, mode=mode) for w, sr in zip(waveforms, sample_rates)
    ]
    return torch.stack(processed)


def load_audio_from_path(path: str | Path, *, mode: PreprocessMode = "deterministic") -> torch.Tensor:
    """Load audio file and return prepared 1D tensor."""
    audio, sr = sf.read(str(path), dtype="float32")
    waveform = torch.from_numpy(audio)
    return prepare_waveform(waveform, sr, mode=mode)


def load_audio_from_bytes(data: bytes, *, mode: PreprocessMode = "deterministic") -> torch.Tensor:
    """Load audio from bytes (API uploads)."""
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    waveform = torch.from_numpy(audio)
    return prepare_waveform(waveform, sr, mode=mode)


def waveform_from_hf_audio(
    audio_dict: dict,
    *,
    mode: PreprocessMode = "deterministic",
) -> torch.Tensor:
    """Convert HuggingFace datasets Audio feature dict to prepared tensor."""
    array = audio_dict["array"]
    sr = int(audio_dict["sampling_rate"])
    waveform = torch.from_numpy(np.asarray(array, dtype=np.float32))
    return prepare_waveform(waveform, sr, mode=mode)
