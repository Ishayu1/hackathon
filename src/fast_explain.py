"""Feature-profile explanations for the fast MFCC/LFCC baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.fast_baseline import FastLabel, fast_feature_label

Direction = Literal["toward_spoof", "toward_bonafide"]

DISCLAIMER = "Signals reflect similarity to training distributions, not proof of synthesis method."
_EPS = 1e-6


@dataclass(frozen=True)
class FastFeatureProfiles:
    feature_names: list[str]
    bonafide_mean: np.ndarray
    bonafide_std: np.ndarray
    spoof_mean: np.ndarray
    spoof_std: np.ndarray
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "FastFeatureProfiles":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_names=list(data["feature_names"]),
            bonafide_mean=np.asarray(data["bonafide_mean"], dtype=np.float64),
            bonafide_std=np.asarray(data["bonafide_std"], dtype=np.float64),
            spoof_mean=np.asarray(data["spoof_mean"], dtype=np.float64),
            spoof_std=np.asarray(data["spoof_std"], dtype=np.float64),
            metadata=dict(data.get("metadata", {})),
        )


def _round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _trend_phrase(value: float, reference_mean: float) -> str:
    if value > reference_mean * 1.001:
        return "above"
    if value < reference_mean * 0.999:
        return "below"
    return "near"


def _format_signal_phrase(signal: dict[str, Any], prediction: FastLabel) -> str:
    reference = signal["bonafide_reference"] if prediction == "bonafide" else signal["spoof_reference"]
    class_word = "authentic" if prediction == "bonafide" else "synthetic"
    trend = _trend_phrase(float(signal["value"]), float(reference["mean"]))
    return (
        f"{signal['label']} ({signal['value']}, {trend} the {class_word} training average "
        f"of {reference['mean']})"
    )


def build_explanation_summary(prediction: FastLabel, top_signals: list[dict[str, Any]]) -> str:
    """Turn ranked feature signals into a short multi-sentence rationale."""
    if not top_signals:
        return ""

    class_word = "authentic" if prediction == "bonafide" else "synthetic"
    contra_word = "synthetic" if prediction == "bonafide" else "authentic"
    phrases = [_format_signal_phrase(signal, prediction) for signal in top_signals[:3]]

    intro = (
        f"The classifier labeled this clip as {prediction} because its acoustic profile "
        f"matches {class_word} training examples more closely than {contra_word} ones."
    )

    if len(phrases) == 1:
        detail = f"The main supporting measurement is {phrases[0]}."
    elif len(phrases) == 2:
        detail = f"Supporting measurements include {phrases[0]} and {phrases[1]}."
    else:
        detail = f"Supporting measurements include {phrases[0]}, {phrases[1]}, and {phrases[2]}."

    return f"{intro} {detail}"


def explain_fast_features(
    features: np.ndarray,
    prediction: FastLabel,
    profiles: FastFeatureProfiles,
    *,
    top_k: int = 3,
) -> dict[str, Any]:
    """Rank features by closeness to the predicted class profile vs the other class."""
    values = np.asarray(features, dtype=np.float64).reshape(-1)
    if values.shape[0] != len(profiles.feature_names):
        raise ValueError(
            f"Feature length mismatch: got {values.shape[0]}, expected {len(profiles.feature_names)}"
        )

    bonafide_z = np.abs(values - profiles.bonafide_mean) / np.maximum(profiles.bonafide_std, _EPS)
    spoof_z = np.abs(values - profiles.spoof_mean) / np.maximum(profiles.spoof_std, _EPS)

    if prediction == "spoof":
        predicted_z = spoof_z
        other_z = bonafide_z
        direction: Direction = "toward_spoof"
    else:
        predicted_z = bonafide_z
        other_z = spoof_z
        direction = "toward_bonafide"

    closeness_margin = other_z - predicted_z
    profile_gap = np.abs(profiles.bonafide_mean - profiles.spoof_mean) / np.maximum(
        (profiles.bonafide_std + profiles.spoof_std) / 2.0,
        _EPS,
    )
    rank_score = closeness_margin + (0.15 * profile_gap)

    candidate_idx = np.where(closeness_margin > 0)[0]
    if candidate_idx.size == 0:
        candidate_idx = np.arange(values.shape[0])
    order = candidate_idx[np.argsort(rank_score[candidate_idx])[::-1]][:top_k]

    top_signals: list[dict[str, Any]] = []
    for idx in order:
        name = profiles.feature_names[int(idx)]
        value = float(values[int(idx)])
        top_signals.append(
            {
                "name": name,
                "label": fast_feature_label(name),
                "value": _round_float(value),
                "bonafide_reference": {
                    "mean": _round_float(profiles.bonafide_mean[int(idx)]),
                    "std": _round_float(profiles.bonafide_std[int(idx)]),
                },
                "spoof_reference": {
                    "mean": _round_float(profiles.spoof_mean[int(idx)]),
                    "std": _round_float(profiles.spoof_std[int(idx)]),
                },
                "direction": direction,
                "closeness_margin": _round_float(closeness_margin[int(idx)]),
            }
        )

    summary = build_explanation_summary(prediction, top_signals)

    return {
        "method": "class_profile_comparison",
        "prediction": prediction,
        "summary": summary,
        "top_signals": top_signals,
        "disclaimer": DISCLAIMER,
    }
