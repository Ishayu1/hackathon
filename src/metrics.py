"""Evaluation metrics for anti-spoofing."""

from __future__ import annotations

import numpy as np


def compute_det_curve(
    target_scores: np.ndarray, nontarget_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute DET curve following ASVspoof official eval package.

    target_scores: bonafide scores (higher = more bonafide)
    nontarget_scores: spoof scores
    """
    target_scores = np.sort(target_scores)
    nontarget_scores = np.sort(nontarget_scores)

    n_target = target_scores.size
    n_nontarget = nontarget_scores.size

    frr = np.zeros(n_target + n_nontarget + 1)
    far = np.zeros(n_target + n_nontarget + 1)
    thresholds = np.zeros(n_target + n_nontarget + 1)

    idx_target = 0
    idx_nontarget = 0

    for i in range(n_target + n_nontarget):
        if idx_target == n_target:
            threshold = nontarget_scores[idx_nontarget]
            idx_nontarget += 1
        elif idx_nontarget == n_nontarget:
            threshold = target_scores[idx_target]
            idx_target += 1
        elif target_scores[idx_target] <= nontarget_scores[idx_nontarget]:
            threshold = target_scores[idx_target]
            idx_target += 1
        else:
            threshold = nontarget_scores[idx_nontarget]
            idx_nontarget += 1

        frr[i] = idx_target / n_target
        far[i] = (n_nontarget - idx_nontarget) / n_nontarget
        thresholds[i] = threshold

    frr[-1] = 1.0
    far[-1] = 0.0
    thresholds[-1] = target_scores[-1] if n_target else 0.0
    return frr, far, thresholds


def compute_eer(
    bonafide_scores: np.ndarray, spoof_scores: np.ndarray
) -> tuple[float, float]:
    """
    Returns (eer, threshold_at_eer).
    EER is a fraction in [0, 1]; multiply by 100 for percentage.
    """
    frr, far, thresholds = compute_det_curve(bonafide_scores, spoof_scores)
    abs_diffs = np.abs(frr - far)
    min_index = int(np.argmin(abs_diffs))
    eer = float(np.mean((frr[min_index], far[min_index])))
    return eer, float(thresholds[min_index])


def accuracy_at_threshold(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
    threshold: float,
) -> float:
    """Fraction of correct classifications at a fixed bonafide-logit threshold."""
    bonafide_correct = np.sum(bonafide_scores > threshold)
    spoof_correct = np.sum(spoof_scores <= threshold)
    total = bonafide_scores.size + spoof_scores.size
    return float((bonafide_correct + spoof_correct) / total)


def latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }
