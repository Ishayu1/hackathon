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


def _relative_phrase(value: float, predicted_mean: float, predicted_label: FastLabel) -> str:
    if value > predicted_mean:
        return f"higher than the {predicted_label} training average"
    if value < predicted_mean:
        return f"lower than the {predicted_label} training average"
    return f"near the {predicted_label} training average"


def explain_fast_features(
    features: np.ndarray,
    prediction: FastLabel,
    profiles: FastFeatureProfiles,
    *,
    top_k: int = 5,
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
        predicted_mean = profiles.spoof_mean
    else:
        predicted_z = bonafide_z
        other_z = spoof_z
        direction = "toward_bonafide"
        predicted_mean = profiles.bonafide_mean

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
        text = (
            f"{fast_feature_label(name)} = {_round_float(value)}; "
            f"this is {_relative_phrase(value, float(predicted_mean[int(idx)]), prediction)} "
            f"and is closer to training examples labeled {prediction}."
        )
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
                "plain_text": text,
            }
        )

    return {
        "method": "class_profile_comparison",
        "prediction": prediction,
        "top_signals": top_signals,
        "disclaimer": DISCLAIMER,
    }
