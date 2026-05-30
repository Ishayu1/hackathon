"""Fast classical baseline: MFCC/LFCC features + sklearn classifier."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import librosa
import numpy as np
import torch
import torchaudio
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from src.audio_preprocess import resample_if_needed, to_mono
from src.config import SAMPLE_RATE

FastFeatureType = Literal["mfcc", "lfcc"]
FastModelType = Literal["logistic_regression", "random_forest", "linear_svc", "rbf_svc"]
DEFAULT_FAST_MODEL_TYPE: FastModelType = "rbf_svc"
RBF_MAX_TRAIN = 5000
FastLabel = Literal["bonafide", "spoof"]
FAST_CLIP_SECONDS = 4
FAST_CLIP_LEN = SAMPLE_RATE * FAST_CLIP_SECONDS


@dataclass
class FastPrediction:
    label: FastLabel
    is_spoof: bool
    score_spoof: float
    score_bonafide: float
    confidence: float
    threshold: float


@dataclass
class FastTimedPrediction(FastPrediction):
    preprocess_ms: float = 0.0
    feature_ms: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0


def prepare_waveform_fast(waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
    """Mono + 16kHz + deterministic center crop/pad to 4s."""
    waveform = to_mono(waveform.float())
    waveform = resample_if_needed(waveform, sample_rate)
    x = waveform.numpy()

    if x.shape[0] >= FAST_CLIP_LEN:
        start = (x.shape[0] - FAST_CLIP_LEN) // 2
        x = x[start : start + FAST_CLIP_LEN]
    else:
        x = np.pad(x, (0, FAST_CLIP_LEN - x.shape[0]))
    return x.astype(np.float32, copy=False)


def fast_feature_names(feature_type: FastFeatureType = "mfcc", n_coeffs: int = 20) -> list[str]:
    """Return stable names matching FastAudioClassifier.extract_features order."""
    prefix = "mfcc" if feature_type == "mfcc" else "lfcc"
    names: list[str] = []
    names.extend(f"{prefix}_{i}_mean" for i in range(n_coeffs))
    names.extend(f"{prefix}_{i}_std" for i in range(n_coeffs))
    names.extend(f"delta_{i}_mean" for i in range(n_coeffs))
    names.extend(f"delta_{i}_std" for i in range(n_coeffs))
    names.extend(
        [
            "spectral_centroid_mean",
            "spectral_centroid_std",
            "spectral_rolloff_mean",
            "spectral_rolloff_std",
            "zero_crossing_rate_mean",
            "zero_crossing_rate_std",
            "rms_mean",
            "rms_std",
        ]
    )
    return names


def fast_feature_label(name: str) -> str:
    labels = {
        "spectral_centroid_mean": "Average spectral centroid",
        "spectral_centroid_std": "Spectral centroid variation",
        "spectral_rolloff_mean": "Average spectral rolloff",
        "spectral_rolloff_std": "Spectral rolloff variation",
        "zero_crossing_rate_mean": "Average zero-crossing rate",
        "zero_crossing_rate_std": "Zero-crossing variation",
        "rms_mean": "Average signal energy",
        "rms_std": "Signal energy variation",
    }
    if name in labels:
        return labels[name]
    parts = name.split("_")
    if len(parts) >= 3 and parts[0] in {"mfcc", "lfcc"}:
        return f"{parts[0].upper()} coefficient {parts[1]} {parts[2]}"
    if len(parts) >= 3 and parts[0] == "delta":
        return f"MFCC delta coefficient {parts[1]} {parts[2]}"
    return name.replace("_", " ")


class FastAudioClassifier:
    """Classical audio anti-spoof baseline for fast CPU inference."""

    def __init__(
        self,
        *,
        feature_type: FastFeatureType = "mfcc",
        model_type: FastModelType = DEFAULT_FAST_MODEL_TYPE,
        n_mfcc: int = 20,
        n_lfcc: int = 20,
    ) -> None:
        self.feature_type = feature_type
        self.model_type = model_type
        self.n_mfcc = n_mfcc
        self.n_lfcc = n_lfcc
        self._clf = self._build_model()

    def _build_model(self):
        if self.model_type == "logistic_regression":
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1500, n_jobs=None)),
                ]
            )
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1,
            )
        if self.model_type == "linear_svc":
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LinearSVC(random_state=42)),
                ]
            )
        if self.model_type == "rbf_svc":
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        SVC(
                            kernel="rbf",
                            C=1.0,
                            gamma="scale",
                            probability=True,
                            random_state=42,
                        ),
                    ),
                ]
            )
        raise ValueError(f"Unsupported model_type: {self.model_type}")

    def _subsample_stratified(self, x: np.ndarray, y: np.ndarray, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if len(y) <= max_samples:
            return x, y
        rng = np.random.default_rng(42)
        idx_bonafide = np.where(y == 1)[0]
        idx_spoof = np.where(y == 0)[0]
        ratio = len(idx_bonafide) / len(y)
        n_bonafide = min(len(idx_bonafide), max(1, int(round(max_samples * ratio))))
        n_spoof = min(len(idx_spoof), max_samples - n_bonafide)
        sel = np.concatenate(
            [
                rng.choice(idx_bonafide, n_bonafide, replace=False),
                rng.choice(idx_spoof, n_spoof, replace=False),
            ]
        )
        rng.shuffle(sel)
        return x[sel], y[sel]

    def extract_features(self, x: np.ndarray) -> np.ndarray:
        if self.feature_type == "mfcc":
            base = librosa.feature.mfcc(
                y=x, sr=SAMPLE_RATE, n_mfcc=self.n_mfcc, n_fft=512, hop_length=160
            )
        elif self.feature_type == "lfcc":
            t = torch.from_numpy(x).unsqueeze(0)
            lfcc = torchaudio.transforms.LFCC(
                sample_rate=SAMPLE_RATE,
                n_lfcc=self.n_lfcc,
                speckwargs={"n_fft": 512, "hop_length": 160},
            )(t)
            base = lfcc.squeeze(0).numpy()
        else:
            raise ValueError(f"Unsupported feature_type: {self.feature_type}")

        delta = librosa.feature.delta(base)
        centroid = librosa.feature.spectral_centroid(y=x, sr=SAMPLE_RATE, n_fft=512, hop_length=160)
        rolloff = librosa.feature.spectral_rolloff(y=x, sr=SAMPLE_RATE, n_fft=512, hop_length=160)
        zcr = librosa.feature.zero_crossing_rate(x, frame_length=512, hop_length=160)
        rms = librosa.feature.rms(y=x, frame_length=512, hop_length=160)

        feats = np.concatenate(
            [
                base.mean(axis=1),
                base.std(axis=1),
                delta.mean(axis=1),
                delta.std(axis=1),
                np.array(
                    [
                        centroid.mean(),
                        centroid.std(),
                        rolloff.mean(),
                        rolloff.std(),
                        zcr.mean(),
                        zcr.std(),
                        rms.mean(),
                        rms.std(),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return feats.astype(np.float32, copy=False)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.model_type == "rbf_svc":
            x, y = self._subsample_stratified(x, y, RBF_MAX_TRAIN)
        self._clf.fit(x, y)

    def _uses_proba(self) -> bool:
        return hasattr(self._clf, "predict_proba")

    def decision_scores(self, x: np.ndarray) -> np.ndarray:
        if self._uses_proba():
            probs = self._clf.predict_proba(x)
            return probs[:, 1]
        if hasattr(self._clf, "decision_function"):
            scores = self._clf.decision_function(x)
            return np.asarray(scores, dtype=np.float64)
        raise RuntimeError("Classifier does not expose predict_proba or decision_function")

    def predict_labels(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        scores = self.decision_scores(x)
        return (scores > threshold).astype(np.int64)

    def predict_one_from_prepared(
        self,
        prepared: np.ndarray,
        *,
        threshold: float = 0.5,
    ) -> tuple[FastPrediction, float, float]:
        t0 = time.perf_counter()
        feat = self.extract_features(prepared).reshape(1, -1)
        feature_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        score_bonafide = float(self.decision_scores(feat)[0])
        inference_ms = (time.perf_counter() - t1) * 1000.0
        is_bonafide = score_bonafide > threshold

        if self._uses_proba():
            conf = float(max(score_bonafide, 1.0 - score_bonafide))
            score_spoof = float(1.0 - score_bonafide)
        else:
            conf = min(1.0, abs(score_bonafide - threshold))
            score_spoof = float(-score_bonafide)

        pred = FastPrediction(
            label="bonafide" if is_bonafide else "spoof",
            is_spoof=not is_bonafide,
            score_spoof=score_spoof,
            score_bonafide=score_bonafide,
            confidence=conf,
            threshold=threshold,
        )
        return pred, feature_ms, inference_ms

    def predict_one(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        threshold: float = 0.5,
    ) -> FastTimedPrediction:
        t0 = time.perf_counter()
        prepared = prepare_waveform_fast(waveform, sample_rate)
        preprocess_ms = (time.perf_counter() - t0) * 1000.0
        pred, feature_ms, inference_ms = self.predict_one_from_prepared(prepared, threshold=threshold)
        total_ms = preprocess_ms + feature_ms + inference_ms
        return FastTimedPrediction(
            label=pred.label,
            is_spoof=pred.is_spoof,
            score_spoof=pred.score_spoof,
            score_bonafide=pred.score_bonafide,
            confidence=pred.confidence,
            threshold=pred.threshold,
            preprocess_ms=preprocess_ms,
            feature_ms=feature_ms,
            inference_ms=inference_ms,
            total_ms=total_ms,
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "feature_type": self.feature_type,
            "model_type": self.model_type,
            "n_mfcc": self.n_mfcc,
            "n_lfcc": self.n_lfcc,
            "clf": self._clf,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "FastAudioClassifier":
        payload = joblib.load(path)
        obj = cls(
            feature_type=payload["feature_type"],
            model_type=payload["model_type"],
            n_mfcc=payload["n_mfcc"],
            n_lfcc=payload["n_lfcc"],
        )
        obj._clf = payload["clf"]
        return obj
