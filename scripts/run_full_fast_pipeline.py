#!/usr/bin/env python3
"""Train fast baseline on full ASVspoof train; eval dev + test; compare to Spectra published EER."""

from __future__ import annotations

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

from scripts.eval_fast_baseline import build_features, label_is_bonafide  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.fast_baseline import FastAudioClassifier  # noqa: E402
from src.metrics import accuracy_at_threshold, compute_eer, latency_stats  # noqa: E402

SPECTRA_PUBLISHED = {
    "validation": {"split_name": "ASVspoof19_LA_dev", "eer_percent": None},
    "test": {"split_name": "ASVspoof19_LA_eval", "eer_percent": 0.723},
}


def eval_split(clf: FastAudioClassifier, split: str) -> dict:
    print(f"\n--- Evaluating split={split} (full) ---")
    ds = load_dataset("Bisher/ASVspoof_2019_LA", split=split)
    ds = ds.cast_column("audio", Audio(decode=False))
    x_eval, y_eval, eval_feature_ms = build_features(ds, None, clf, stratified=False)

    t_inf = time.perf_counter()
    scores = np.asarray(clf.decision_scores(x_eval), dtype=np.float64)
    inference_total_ms = (time.perf_counter() - t_inf) * 1000.0

    bonafide = scores[y_eval == 1]
    spoof = scores[y_eval == 0]
    eer, eer_threshold = compute_eer(bonafide, spoof)
    acc = accuracy_at_threshold(bonafide, spoof, 0.5)

    n_eval = int(y_eval.shape[0])
    total_eval_s = float(np.sum(eval_feature_ms) / 1000.0 + inference_total_ms / 1000.0)
    per_sample_infer = inference_total_ms / max(n_eval, 1)

    return {
        "eval_split": split,
        "n_eval": n_eval,
        "n_bonafide": int(bonafide.size),
        "n_spoof": int(spoof.size),
        "eer": eer,
        "eer_percent": eer * 100.0,
        "eer_threshold": eer_threshold,
        "accuracy_at_threshold": acc,
        "eval_samples_per_sec": n_eval / total_eval_s if total_eval_s > 0 else 0.0,
        "feature_latency": latency_stats(eval_feature_ms),
        "inference_latency": latency_stats([per_sample_infer] * n_eval),
        "scores": scores,
        "labels": y_eval,
    }


def main() -> None:
    feature_type, model_type = "mfcc", "rbf_svc"
    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Full fast baseline pipeline")
    print("  Train: ASVspoof 2019 LA train (full)")
    print("  Eval:  validation (dev) + test (official eval)")
    print("=" * 60)

    print("\nLoading train split ...")
    train_ds = load_dataset("Bisher/ASVspoof_2019_LA", split="train")
    train_ds = train_ds.cast_column("audio", Audio(decode=False))

    clf = FastAudioClassifier(feature_type=feature_type, model_type=model_type)

    print("Extracting training features (full train split) ...")
    t0 = time.perf_counter()
    x_train, y_train, train_feature_ms = build_features(train_ds, None, clf, stratified=False)
    feature_train_s = time.perf_counter() - t0
    print(f"  train features done in {feature_train_s:.1f}s ({len(y_train)} samples)")

    print("Training classifier ...")
    t_train = time.perf_counter()
    clf.fit(x_train, y_train)
    train_ms = (time.perf_counter() - t_train) * 1000.0

    model_path = output_dir / f"fast_baseline_{feature_type}_{model_type}_full.joblib"
    clf.save(model_path)
    print(f"Model saved: {model_path}")

    results = {}
    for split in ("validation", "test"):
        r = eval_split(clf, split)
        scores_path = output_dir / f"scores_fast_{feature_type}_{model_type}_{split}_full.tsv"
        with scores_path.open("w", encoding="utf-8") as f:
            f.write("row_index\tbonafide_score\tlabel\n")
            for i, (score, label) in enumerate(zip(r["scores"], r["labels"])):
                f.write(f"{i}\t{float(score):.8f}\t{'bonafide' if label == 1 else 'spoof'}\n")

        summary = {k: v for k, v in r.items() if k not in ("scores", "labels")}
        summary.update(
            {
                "pipeline": "fast_baseline",
                "feature_type": feature_type,
                "model_type": model_type,
                "train_split": "train",
                "n_train": int(y_train.shape[0]),
                "training_ms": train_ms,
                "train_feature_latency": latency_stats(train_feature_ms),
                "model_path": str(model_path),
                "full_split": True,
            }
        )
        summary_path = output_dir / f"summary_fast_{feature_type}_{model_type}_{split}_full.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        results[split] = summary
        print(f"  EER: {summary['eer_percent']:.4f}%  acc@0.5: {summary['accuracy_at_threshold']*100:.2f}%")
        print(f"  saved: {summary_path}")

    comparison = {
        "fast_baseline": {
            "validation_dev_eer_percent": results["validation"]["eer_percent"],
            "test_eval_eer_percent": results["test"]["eer_percent"],
            "n_train": results["test"]["n_train"],
            "n_eval_test": results["test"]["n_eval"],
            "feature_latency_p95_ms": results["test"]["feature_latency"]["p95_ms"],
        },
        "spectra_aasist3_published": {
            "ASVspoof19_LA_eval_eer_percent": SPECTRA_PUBLISHED["test"]["eer_percent"],
            "source": "https://huggingface.co/lab260/Spectra-AASIST3",
        },
        "delta_test_eer_percent": results["test"]["eer_percent"] - SPECTRA_PUBLISHED["test"]["eer_percent"],
        "speed_ratio_vs_spectra_mps": "~190ms / ~6ms ≈ 32x faster (fast baseline on CPU)",
    }

    comparison_path = output_dir / "comparison_fast_vs_spectra.json"
    comparison_path.write_text(json.dumps(comparison, indent=2))

    print("\n" + "=" * 60)
    print("COMPARISON: Fast baseline (full) vs Spectra-AASIST3 (published)")
    print("=" * 60)
    print(f"  Fast baseline  — dev EER:  {results['validation']['eer_percent']:.4f}%")
    print(f"  Fast baseline  — test EER: {results['test']['eer_percent']:.4f}%  (n={results['test']['n_eval']})")
    print(f"  Spectra (pub.) — test EER: {SPECTRA_PUBLISHED['test']['eer_percent']:.4f}%")
    print(f"  Gap on test:               +{comparison['delta_test_eer_percent']:.4f} pp (fast worse)")
    print(f"  Fast latency p95:          {results['test']['feature_latency']['p95_ms']:.2f} ms")
    print(f"  Comparison saved:          {comparison_path}")


if __name__ == "__main__":
    main()
