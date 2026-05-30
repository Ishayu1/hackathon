#!/usr/bin/env python3
"""Train RBF SVM on Gary Stafford demo dataset (80/20 stratified split)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_DEMO_FAST_MODEL, DEMO_FEATURE_CACHE, DEMO_MANIFEST, PROJECT_ROOT, RESULTS_DIR
from src.fast_baseline import FastAudioClassifier, prepare_waveform_fast
from src.metrics import accuracy_at_threshold, compute_eer, latency_stats


def label_to_int(label: str) -> int:
    return 1 if label.strip().lower() in ("real", "bonafide") else 0


def load_manifest() -> list[dict]:
    data = json.loads(DEMO_MANIFEST.read_text())
    return data["files"]


def extract_features(entries: list[dict], clf: FastAudioClassifier) -> tuple[np.ndarray, np.ndarray, list[float]]:
    x_rows, y_rows, ms = [], [], []
    for j, entry in enumerate(entries):
        path = PROJECT_ROOT / entry["path"]
        audio, sr = sf.read(str(path), dtype="float32")
        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        prepared = prepare_waveform_fast(waveform, sr)
        t0 = time.perf_counter()
        feats = clf.extract_features(prepared)
        ms.append((time.perf_counter() - t0) * 1000.0)
        x_rows.append(feats)
        y_rows.append(label_to_int(entry["label"]))
        if (j + 1) % 200 == 0 or j + 1 == len(entries):
            print(f"  features {j + 1}/{len(entries)}")
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int64), ms


def stratified_split(entries: list[dict], train_frac: float = 0.8, seed: int = 42):
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


def eval_clf(clf: FastAudioClassifier, x: np.ndarray, y: np.ndarray) -> dict:
    scores = np.asarray(clf.decision_scores(x), dtype=np.float64)
    bonafide = scores[y == 1]
    spoof = scores[y == 0]
    eer, eer_thr = compute_eer(bonafide, spoof)
    acc = accuracy_at_threshold(bonafide, spoof, 0.5)
    return {
        "eer_percent": float(eer * 100.0),
        "eer_threshold": float(eer_thr),
        "accuracy": float(acc),
        "n_bonafide": int(bonafide.size),
        "n_spoof": int(spoof.size),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    entries = load_manifest()
    train_entries, test_entries = stratified_split(entries)
    print(f"Demo train/test: {len(train_entries)} / {len(test_entries)}")

    fe = FastAudioClassifier(model_type="rbf_svc")

    if DEMO_FEATURE_CACHE.exists():
        print(f"Loading feature cache {DEMO_FEATURE_CACHE.name} ...")
        cached = np.load(DEMO_FEATURE_CACHE, allow_pickle=True)
        paths = [str(e["path"]) for e in entries]
        path_to_idx = {p: i for i, p in enumerate(paths)}
        x_all, y_all = cached["x"], cached["y"]
        train_idx = [path_to_idx[e["path"]] for e in train_entries]
        test_idx = [path_to_idx[e["path"]] for e in test_entries]
        x_train, y_train = x_all[train_idx], y_all[train_idx]
        x_test, y_test = x_all[test_idx], y_all[test_idx]
    else:
        print("Extracting features for full demo set (cached for reuse) ...")
        x_all, y_all, _ = extract_features(entries, fe)
        np.savez_compressed(DEMO_FEATURE_CACHE, x=x_all, y=y_all)
        print(f"  saved {DEMO_FEATURE_CACHE}")
        train_idx = [entries.index(e) for e in train_entries]
        test_idx = [entries.index(e) for e in test_entries]
        x_train, y_train = x_all[train_idx], y_all[train_idx]
        x_test, y_test = x_all[test_idx], y_all[test_idx]

    print("Training RBF SVM on demo train split ...")
    t0 = time.perf_counter()
    fe.fit(x_train, y_train)
    train_ms = (time.perf_counter() - t0) * 1000.0

    fe.save(DEFAULT_DEMO_FAST_MODEL)
    print(f"Saved {DEFAULT_DEMO_FAST_MODEL}")

    train_metrics = eval_clf(fe, x_train, y_train)
    test_metrics = eval_clf(fe, x_test, y_test)

    summary = {
        "model": "mfcc_rbf_svc_demo",
        "model_path": str(DEFAULT_DEMO_FAST_MODEL),
        "train_ms": train_ms,
        "n_train": len(train_entries),
        "n_test": len(test_entries),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    out = RESULTS_DIR / "summary_fast_demo_rbf.json"
    out.write_text(json.dumps(summary, indent=2))

    print("\n=== Demo-tuned RBF SVM ===")
    print(f"  Train acc: {train_metrics['accuracy']*100:.2f}%  EER: {train_metrics['eer_percent']:.2f}%")
    print(f"  Test  acc: {test_metrics['accuracy']*100:.2f}%  EER: {test_metrics['eer_percent']:.2f}%")
    print(f"  Summary: {out}")


if __name__ == "__main__":
    main()
