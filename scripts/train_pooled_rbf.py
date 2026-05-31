#!/usr/bin/env python3
"""Train pooled RBF SVM on demo + ASVspoof train features, eval on OOD + demo holdout."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_feature_profiles import stratified_split
from scripts.eval_fast_baseline import build_features, label_is_bonafide
from src.config import DEMO_FEATURE_CACHE, DEMO_MANIFEST, PROJECT_ROOT, RESULTS_DIR
from src.fast_baseline import FastAudioClassifier, fast_feature_names
from src.metrics import accuracy_at_threshold, compute_eer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train pooled fast RBF on demo + ASVspoof")
    p.add_argument("--max-asvspoof-train", type=int, default=5000, help="Stratified cap from ASVspoof train")
    p.add_argument("--max-per-source", type=int, default=2500, help="Cap clips per source after pooling")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--asvspoof-cache", type=Path, default=RESULTS_DIR / "feature_cache_asvspoof_train_mfcc.npz")
    return p.parse_args()


def load_demo_train() -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads(DEMO_MANIFEST.read_text(encoding="utf-8"))["files"]
    train_entries, test_entries = stratified_split(manifest, seed=42)
    if not DEMO_FEATURE_CACHE.exists():
        raise FileNotFoundError(f"Missing {DEMO_FEATURE_CACHE}; run scripts/train_demo_rbf.py first")

    cached = np.load(DEMO_FEATURE_CACHE, allow_pickle=True)
    paths = [e["path"] for e in manifest]
    path_to_idx = {p: i for i, p in enumerate(paths)}
    train_idx = [path_to_idx[e["path"]] for e in train_entries]
    test_idx = [path_to_idx[e["path"]] for e in test_entries]
    x_train = cached["x"][train_idx]
    y_train = cached["y"][train_idx]
    x_test = cached["x"][test_idx]
    y_test = cached["y"][test_idx]
    return x_train, y_train, x_test, y_test, test_entries


def load_or_build_asvspoof_train(max_samples: int, cache_path: Path, clf: FastAudioClassifier):
    if cache_path.exists():
        print(f"Loading ASVspoof feature cache {cache_path.name} ...")
        data = np.load(cache_path, allow_pickle=True)
        return data["x"], data["y"]

    print(f"Extracting ASVspoof train features (max {max_samples}) ...")
    ds = load_dataset("Bisher/ASVspoof_2019_LA", split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    x, y, _ = build_features(ds, max_samples, clf, stratified=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=x, y=y)
    print(f"  saved {cache_path}")
    return x, y


def balanced_cap(x: np.ndarray, y: np.ndarray, max_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx_b = np.where(y == 1)[0]
    idx_s = np.where(y == 0)[0]
    nb = min(len(idx_b), max_per_class)
    ns = min(len(idx_s), max_per_class)
    sel = np.concatenate([rng.choice(idx_b, nb, replace=False), rng.choice(idx_s, ns, replace=False)])
    rng.shuffle(sel)
    return x[sel], y[sel]


def save_profiles(x: np.ndarray, y: np.ndarray, path: Path, feature_type: str = "mfcc") -> None:
    bon = x[y == 1]
    spo = x[y == 0]
    payload = {
        "feature_names": fast_feature_names(feature_type),
        "bonafide_mean": bon.mean(axis=0).astype(float).tolist(),
        "bonafide_std": np.maximum(bon.std(axis=0), 1e-6).astype(float).tolist(),
        "spoof_mean": spo.mean(axis=0).astype(float).tolist(),
        "spoof_std": np.maximum(spo.std(axis=0), 1e-6).astype(float).tolist(),
        "metadata": {
            "method": "pooled_train_class_profile",
            "n_train": int(y.size),
            "n_bonafide": int(bon.shape[0]),
            "n_spoof": int(spo.shape[0]),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def eval_xy(clf: FastAudioClassifier, x: np.ndarray, y: np.ndarray) -> dict:
    scores = np.asarray(clf.decision_scores(x), dtype=np.float64)
    bon = scores[y == 1]
    spo = scores[y == 0]
    eer, _ = compute_eer(bon, spo)
    acc = accuracy_at_threshold(bon, spo, 0.5)
    return {"accuracy": float(acc), "eer_percent": float(eer * 100.0), "n": int(len(y))}


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x_demo, y_demo, x_demo_test, y_demo_test, _ = load_demo_train()
    clf = FastAudioClassifier(model_type="rbf_svc")
    x_asv, y_asv = load_or_build_asvspoof_train(args.max_asvspoof_train, args.asvspoof_cache, clf)

    x_demo_c, y_demo_c = balanced_cap(x_demo, y_demo, args.max_per_source, args.seed)
    x_asv_c, y_asv_c = balanced_cap(x_asv, y_asv, args.max_per_source, args.seed + 1)
    x_pool = np.vstack([x_demo_c, x_asv_c])
    y_pool = np.concatenate([y_demo_c, y_asv_c])

    print(f"Pooled train: {len(y_pool)} (demo {len(y_demo_c)}, asvspoof {len(y_asv_c)})")
    t0 = time.perf_counter()
    clf.fit(x_pool, y_pool)
    train_ms = (time.perf_counter() - t0) * 1000.0

    model_path = RESULTS_DIR / "fast_baseline_mfcc_rbf_svc_pooled.joblib"
    profile_path = RESULTS_DIR / "fast_pooled_feature_profiles.json"
    clf.save(model_path)
    save_profiles(x_pool, y_pool, profile_path)

    demo_holdout = eval_xy(clf, x_demo_test, y_demo_test)
    print(f"Demo holdout: acc={demo_holdout['accuracy']*100:.1f}% EER={demo_holdout['eer_percent']:.1f}%")

    summary = {
        "model_path": str(model_path),
        "profile_path": str(profile_path),
        "train_ms": train_ms,
        "n_pooled": int(len(y_pool)),
        "demo_holdout": demo_holdout,
    }
    out = RESULTS_DIR / "summary_fast_pooled_rbf.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {model_path}")
    print(f"Saved {profile_path}")
    print(f"Run OOD eval: FAST_MODEL_PATH={model_path} python scripts/eval_ood_testsets.py")


if __name__ == "__main__":
    main()
