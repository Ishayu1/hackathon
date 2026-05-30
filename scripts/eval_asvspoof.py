#!/usr/bin/env python3
"""Batch evaluation on ASVspoof 2019 LA via Hugging Face datasets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.audio_preprocess import PreprocessMode, waveform_from_hf_audio
from src.config import DEFAULT_THRESHOLD, PREPROCESS_MODES, RESULTS_DIR
from src.inference import bonafide_logit_from_logits, initialize_inference
from src.metrics import accuracy_at_threshold, compute_eer, latency_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Spectra-AASIST3 on ASVspoof 2019 LA")
    parser.add_argument(
        "--split",
        choices=["validation", "test", "train"],
        default="validation",
        help="HF dataset split (validation=dev, test=eval)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--preprocess",
        default="deterministic",
        choices=list(PREPROCESS_MODES),
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def label_is_bonafide(key: str) -> bool:
    return key.strip().lower() == "bonafide"


def run_eval(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model on device={args.device} ...")
    model, device = initialize_inference(args.device)
    print(f"Model ready on {device}")

    print(f"Loading dataset split={args.split} ...")
    dataset = load_dataset("Bisher/ASVspoof_2019_LA", split=args.split)

    required_cols = {"audio", "audio_file_name", "key"}
    missing = required_cols - set(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    n_total = len(dataset)
    if args.max_samples is not None:
        n_total = min(n_total, args.max_samples)

    utterance_ids: list[str] = []
    bonafide_logits: list[float] = []
    labels: list[int] = []
    latencies_ms: list[float] = []

    batch_waveforms: list[torch.Tensor] = []
    batch_ids: list[str] = []
    batch_labels: list[int] = []

    t_start = time.perf_counter()

    def flush_batch() -> None:
        nonlocal batch_waveforms, batch_ids, batch_labels
        if not batch_waveforms:
            return

        prepared = torch.stack(batch_waveforms).to(device)
        with torch.inference_mode():
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(prepared)
            if device.type == "cuda":
                torch.cuda.synchronize()
            batch_ms = (time.perf_counter() - t0) * 1000.0

        scores = bonafide_logit_from_logits(logits).cpu().numpy()
        per_sample_ms = batch_ms / len(batch_ids)
        for uid, score, label in zip(batch_ids, scores, batch_labels):
            utterance_ids.append(uid)
            bonafide_logits.append(float(score))
            labels.append(label)
            latencies_ms.append(per_sample_ms)

        batch_waveforms = []
        batch_ids = []
        batch_labels = []

    mode: PreprocessMode = args.preprocess  # type: ignore[assignment]

    for idx in range(n_total):
        row = dataset[idx]
        prepared = waveform_from_hf_audio(row["audio"], mode=mode)
        batch_waveforms.append(prepared)
        batch_ids.append(row["audio_file_name"])
        batch_labels.append(1 if label_is_bonafide(row["key"]) else 0)

        if len(batch_waveforms) >= args.batch_size:
            flush_batch()

        if (idx + 1) % 500 == 0 or idx + 1 == n_total:
            elapsed = time.perf_counter() - t_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
            print(f"  processed {idx + 1}/{n_total} ({rate:.1f} samples/s)")

    flush_batch()

    scores_arr = np.asarray(bonafide_logits, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)

    bonafide_scores = scores_arr[labels_arr == 1]
    spoof_scores = scores_arr[labels_arr == 0]

    eer, eer_threshold = compute_eer(bonafide_scores, spoof_scores)
    acc = accuracy_at_threshold(bonafide_scores, spoof_scores, args.threshold)

    total_elapsed = time.perf_counter() - t_start
    lat = latency_stats(latencies_ms)

    summary = {
        "split": args.split,
        "n_samples": n_total,
        "eer": eer,
        "eer_percent": eer * 100.0,
        "eer_threshold": eer_threshold,
        "accuracy_at_threshold": acc,
        "classification_threshold": args.threshold,
        "preprocess_mode": args.preprocess,
        "device": str(device),
        "batch_size": args.batch_size,
        "total_elapsed_s": total_elapsed,
        "samples_per_sec": n_total / total_elapsed if total_elapsed > 0 else 0.0,
        "latency": lat,
        "model_card_baseline_eer_percent": 0.723,
        "n_bonafide": int(bonafide_scores.size),
        "n_spoof": int(spoof_scores.size),
    }

    scores_path = output_dir / f"scores_{args.split}.tsv"
    with scores_path.open("w", encoding="utf-8") as f:
        f.write("utterance_id\tbonafide_logit\tlabel\n")
        for uid, score, label in zip(utterance_ids, bonafide_logits, labels):
            label_str = "bonafide" if label == 1 else "spoof"
            f.write(f"{uid}\t{score:.6f}\t{label_str}\n")

    summary_path = output_dir / f"summary_{args.split}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Results ===")
    print(f"EER: {summary['eer_percent']:.4f}% (threshold @ EER: {eer_threshold:.6f})")
    print(f"Accuracy @ threshold {args.threshold}: {acc * 100:.2f}%")
    print(f"Latency p50/p95: {lat['p50_ms']:.1f} / {lat['p95_ms']:.1f} ms")
    print(f"Throughput: {summary['samples_per_sec']:.2f} samples/s")
    print(f"Model card baseline: 0.723% EER")
    print(f"Scores: {scores_path}")
    print(f"Summary: {summary_path}")

    return summary


def main() -> None:
    args = parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
