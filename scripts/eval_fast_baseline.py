#!/usr/bin/env python3
"""Train + evaluate fast classical baseline on ASVspoof 2019 LA."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR
from src.fast_baseline import FastAudioClassifier, prepare_waveform_fast
from src.metrics import accuracy_at_threshold, compute_eer, latency_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fast MFCC/LFCC + sklearn baseline")
    parser.add_argument("--train-split", default="train", choices=["train", "validation"])
    parser.add_argument("--eval-split", default="validation", choices=["validation", "test", "train"])
    parser.add_argument("--max-train-samples", type=int, default=8000)
    parser.add_argument("--max-eval-samples", type=int, default=3000)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use entire train/eval splits (ignores max-*-samples)",
    )
    parser.add_argument("--feature-type", default="mfcc", choices=["mfcc", "lfcc"])
    parser.add_argument(
        "--model-type",
        default="rbf_svc",
        choices=["logistic_regression", "random_forest", "linear_svc", "rbf_svc"],
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="bonafide threshold for metrics (for linear_svc tune around 0.0)",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def label_is_bonafide(key) -> int:
    """Return 1 for bonafide, 0 for spoof. HF dataset uses 0=bonafide, 1=spoof."""
    if isinstance(key, str):
        k = key.strip().lower()
        if k in ("bonafide", "real", "0"):
            return 1
        if k in ("spoof", "fake", "1"):
            return 0
        raise ValueError(f"Unknown label string: {key!r}")
    if isinstance(key, (int, np.integer)):
        return 1 if int(key) == 0 else 0
    raise ValueError(f"Unsupported label format: {type(key)}")


def row_to_prepared(row: dict) -> np.ndarray:
    audio = row["audio"]
    if "array" in audio and "sampling_rate" in audio:
        waveform = torch.from_numpy(np.asarray(audio["array"], dtype=np.float32))
        sr = int(audio["sampling_rate"])
    elif "bytes" in audio and audio["bytes"]:
        arr, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        waveform = torch.from_numpy(np.asarray(arr, dtype=np.float32))
    elif "path" in audio and audio["path"]:
        arr, sr = sf.read(audio["path"], dtype="float32")
        waveform = torch.from_numpy(np.asarray(arr, dtype=np.float32))
    else:
        raise ValueError("Unsupported audio row format; expected decoded array or file path")
    return prepare_waveform_fast(waveform, sr)


def stratified_indices(labels: np.ndarray, max_samples: int, seed: int = 42) -> np.ndarray:
    """Pick a balanced-ish subset covering both bonafide (0) and spoof (1)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    idx = np.arange(labels.shape[0])
    bonafide_idx = idx[labels == 0]
    spoof_idx = idx[labels == 1]
    if bonafide_idx.size == 0 or spoof_idx.size == 0:
        return idx[:max_samples]

    ratio_bonafide = bonafide_idx.size / labels.size
    n_bonafide = min(bonafide_idx.size, max(1, int(round(max_samples * ratio_bonafide))))
    n_spoof = min(spoof_idx.size, max_samples - n_bonafide)
    if n_bonafide + n_spoof < max_samples:
        n_bonafide = min(bonafide_idx.size, max_samples - n_spoof)

    chosen = np.concatenate(
        [
            rng.choice(bonafide_idx, n_bonafide, replace=False),
            rng.choice(spoof_idx, n_spoof, replace=False),
        ]
    )
    rng.shuffle(chosen)
    return chosen


def build_features(dataset, max_samples: int | None, clf: FastAudioClassifier, *, stratified: bool = True):
    n_ds = len(dataset)
    if max_samples is None or max_samples >= n_ds:
        indices = np.arange(n_ds)
    elif stratified:
        all_labels = np.asarray([label_is_bonafide(dataset[i]["key"]) for i in range(n_ds)])
        # label_is_bonafide returns 1=bonafide; map back to 0/1 key space for stratify
        key_labels = np.where(all_labels == 1, 0, 1)
        indices = stratified_indices(key_labels, max_samples)
    else:
        indices = np.arange(min(len(dataset), max_samples))

    n_total = len(indices)
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    feature_ms: list[float] = []
    t_start = time.perf_counter()
    for j, idx in enumerate(indices):
        row = dataset[int(idx)]
        prepared = row_to_prepared(row)
        t0 = time.perf_counter()
        feats = clf.extract_features(prepared)
        feature_ms.append((time.perf_counter() - t0) * 1000.0)
        x_rows.append(feats)
        y_rows.append(label_is_bonafide(row["key"]))
        if (j + 1) % 500 == 0 or j + 1 == n_total:
            elapsed = time.perf_counter() - t_start
            rate = (j + 1) / elapsed if elapsed > 0 else 0.0
            print(f"  features {j + 1}/{n_total} ({rate:.1f} samples/s)")
    x = np.asarray(x_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int64)
    return x, y, feature_ms


def main() -> None:
    args = parse_args()
    if args.full:
        args.max_train_samples = None
        args.max_eval_samples = None

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading train split={args.train_split} ...")
    train_ds = load_dataset("Bisher/ASVspoof_2019_LA", split=args.train_split)
    print(f"Loading eval split={args.eval_split} ...")
    eval_ds = load_dataset("Bisher/ASVspoof_2019_LA", split=args.eval_split)
    train_ds = train_ds.cast_column("audio", Audio(decode=False))
    eval_ds = eval_ds.cast_column("audio", Audio(decode=False))

    clf = FastAudioClassifier(feature_type=args.feature_type, model_type=args.model_type)

    print("Extracting training features ...")
    x_train, y_train, train_feature_ms = build_features(train_ds, args.max_train_samples, clf)

    print("Training model ...")
    t_train = time.perf_counter()
    clf.fit(x_train, y_train)
    train_ms = (time.perf_counter() - t_train) * 1000.0

    print("Extracting eval features ...")
    x_eval, y_eval, eval_feature_ms = build_features(eval_ds, args.max_eval_samples, clf)

    print("Running eval inference ...")
    t_inf = time.perf_counter()
    bonafide_scores = clf.decision_scores(x_eval)
    inference_total_ms = (time.perf_counter() - t_inf) * 1000.0
    per_sample_infer_ms = inference_total_ms / max(len(x_eval), 1)
    inference_latencies = [per_sample_infer_ms] * len(x_eval)

    bonafide_scores = np.asarray(bonafide_scores, dtype=np.float64)
    labels = np.asarray(y_eval, dtype=np.int64)
    bonafide = bonafide_scores[labels == 1]
    spoof = bonafide_scores[labels == 0]

    eer, eer_threshold = compute_eer(bonafide, spoof)
    acc = accuracy_at_threshold(bonafide, spoof, args.threshold)

    total_eval_s = float(np.sum(eval_feature_ms) / 1000.0 + inference_total_ms / 1000.0)
    n_eval = int(labels.shape[0])
    summary = {
        "pipeline": "fast_baseline",
        "feature_type": args.feature_type,
        "model_type": args.model_type,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "n_train": int(y_train.shape[0]),
        "n_eval": n_eval,
        "eer": eer,
        "eer_percent": eer * 100.0,
        "eer_threshold": eer_threshold,
        "accuracy_at_threshold": acc,
        "classification_threshold": args.threshold,
        "training_ms": train_ms,
        "eval_samples_per_sec": n_eval / total_eval_s if total_eval_s > 0 else 0.0,
        "feature_latency": latency_stats(eval_feature_ms),
        "inference_latency": latency_stats(inference_latencies),
        "train_feature_latency": latency_stats(train_feature_ms),
    }

    model_path = output_dir / f"fast_baseline_{args.feature_type}_{args.model_type}.joblib"
    clf.save(model_path)
    summary["model_path"] = str(model_path)

    summary_path = output_dir / f"summary_fast_{args.feature_type}_{args.model_type}_{args.eval_split}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    scores_path = output_dir / f"scores_fast_{args.feature_type}_{args.model_type}_{args.eval_split}.tsv"
    with scores_path.open("w", encoding="utf-8") as f:
        f.write("row_index\tbonafide_score\tlabel\n")
        for i, (score, label) in enumerate(zip(bonafide_scores, labels)):
            f.write(f"{i}\t{float(score):.8f}\t{'bonafide' if label == 1 else 'spoof'}\n")

    print("\n=== Fast Baseline Results ===")
    print(f"Model: {args.feature_type} + {args.model_type}")
    print(f"EER: {summary['eer_percent']:.4f}% (threshold @ EER: {eer_threshold:.6f})")
    print(f"Accuracy @ threshold {args.threshold}: {acc * 100:.2f}%")
    print(
        f"Feature latency p50/p95: {summary['feature_latency']['p50_ms']:.2f} / "
        f"{summary['feature_latency']['p95_ms']:.2f} ms"
    )
    print(
        f"Inference latency p50/p95: {summary['inference_latency']['p50_ms']:.4f} / "
        f"{summary['inference_latency']['p95_ms']:.4f} ms"
    )
    print(f"Eval throughput: {summary['eval_samples_per_sec']:.2f} samples/s")
    print(f"Model saved: {model_path}")
    print(f"Summary: {summary_path}")
    print(f"Scores: {scores_path}")


if __name__ == "__main__":
    main()
