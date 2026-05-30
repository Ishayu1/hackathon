"""Model loading and warmup for Spectra-AASIST3."""

from __future__ import annotations

import torch

from src.config import CLIP_LEN, MODEL_ID
from vendor.spectra_aasist3.model import SpectraAASIST3

_model: SpectraAASIST3 | None = None
_device: torch.device | None = None


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model(device: str = "auto") -> tuple[SpectraAASIST3, torch.device]:
    """Load model once and cache globally."""
    global _model, _device

    resolved = resolve_device(device)
    if _model is not None and _device == resolved:
        return _model, _device

    model = SpectraAASIST3.from_pretrained(MODEL_ID)
    model.eval()
    model.to(resolved)

    _model = model
    _device = resolved
    return model, resolved


def warmup(model: SpectraAASIST3, device: torch.device) -> None:
    """Run a dummy forward pass to avoid cold-start latency skew."""
    dummy = torch.zeros(1, CLIP_LEN, device=device)
    with torch.inference_mode():
        model(dummy)


def get_loaded_model() -> tuple[SpectraAASIST3, torch.device]:
    if _model is None or _device is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model, _device
