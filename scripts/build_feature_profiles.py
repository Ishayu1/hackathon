#!/usr/bin/env python3
"""Build fast-baseline class feature profiles from the demo train split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR
from src.fast_baseline import FastAudioClassifier, fast_feature_names, prepare_waveform_fast

DEFAULT_MANIFEST = ROOT / "data" / "demo" / "deepfake-audio-detection" / "manifest.json"
DEFAULT_CACHE = RESULTS_DIR / "feature_cache_demo_mfcc.npz"
DEFAULT_OUTPUT = RESULTS_DIR / "fast_demo_feature_profiles.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build class-profile explanations for fast demo model")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-type", default="mfcc", choices=["mfcc", "lfcc"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    return parser.parse_args()


def label_to_int(label: str) -> int:
    return 1 if label.strip().lower() in {"real", "bonafide"} else 0


def stratified_split(entries: list[dict], train_frac: float = 0.8, seed: int = 42) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    real = [e for e in entries if e["label"] == "real"]
    fake = [e for e in entries if e["label"] == "fake"]
    rng.shuffle(real)
    rng.shuffle(fake)
    n_real_train = int(len(real) * train_frac)
    n_fake_train = int(len(fake) * train_frac)
    train = real[:n_real_train] + fake[:n_fake_train]
    test = real[n_real_train:] + fake[n_fake_train:]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def load_features_from_cache(cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    cached = np.load(cache_path, allow_pickle=True)
    return np.asarray(cached["x"], dtype=np.float32), np.asarray(cached["y"], dtype=np.int64)


def build_feature_cache(entries: list[dict], cache_path: Path, clf: FastAudioClassifier) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    for idx, entry in enumerate(entries):
        audio_path = ROOT / entry["path"]
        audio, sr = sf.read(str(audio_path), dtype="float32")
        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        prepared = prepare_waveform_fast(waveform, int(sr))
        x_rows.append(clf.extract_features(prepared))
        y_rows.append(label_to_int(entry["label"]))
        if (idx + 1) % 200 == 0 or idx + 1 == len(entries):
            print(f"  features {idx + 1}/{len(entries)}")
    x = np.asarray(x_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=x, y=y)
    print(f"Wrote feature cache {cache_path}")
    return x, y


def compute_profile(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return values.mean(axis=0), np.maximum(values.std(axis=0), 1e-6)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest["files"]
    train_entries, _ = stratified_split(entries, train_frac=args.train_frac, seed=args.seed)

    clf = FastAudioClassifier(feature_type=args.feature_type)
    if args.feature_cache.exists():
        x_all, y_all = load_features_from_cache(args.feature_cache)
    else:
        print(f"Feature cache not found at {args.feature_cache}; extracting from manifest audio ...")
        x_all, y_all = build_feature_cache(entries, args.feature_cache, clf)
    path_to_idx = {str(e["path"]): i for i, e in enumerate(entries)}
    train_idx = [path_to_idx[str(e["path"])] for e in train_entries]
    x_train = x_all[train_idx]
    y_train = y_all[train_idx]

    bonafide_values = x_train[y_train == 1]
    spoof_values = x_train[y_train == 0]
    bonafide_mean, bonafide_std = compute_profile(bonafide_values)
    spoof_mean, spoof_std = compute_profile(spoof_values)

    n_coeffs = clf.n_mfcc if args.feature_type == "mfcc" else clf.n_lfcc
    payload = {
        "feature_names": fast_feature_names(args.feature_type, n_coeffs),
        "bonafide_mean": bonafide_mean.astype(float).tolist(),
        "bonafide_std": bonafide_std.astype(float).tolist(),
        "spoof_mean": spoof_mean.astype(float).tolist(),
        "spoof_std": spoof_std.astype(float).tolist(),
        "metadata": {
            "method": "train_split_class_profile",
            "dataset": manifest.get("dataset", "garystafford/deepfake-audio-detection"),
            "seed": args.seed,
            "train_frac": args.train_frac,
            "n_train": int(y_train.size),
            "n_bonafide": int(bonafide_values.shape[0]),
            "n_spoof": int(spoof_values.shape[0]),
            "feature_type": args.feature_type,
        },
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} with {len(payload['feature_names'])} feature profiles")


if __name__ == "__main__":
    main()
