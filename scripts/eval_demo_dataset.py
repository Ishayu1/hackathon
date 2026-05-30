#!/usr/bin/env python3
"""Batch-eval Spectra + fast models on Gary Stafford demo FLAC files."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    DEFAULT_DEMO_FAST_MODEL,
    DEFAULT_FAST_MODEL,
    DEFAULT_SPECTRA_DECISION,
    DEFAULT_THRESHOLD,
    DEMO_MANIFEST,
    PROJECT_ROOT,
    RESULTS_DIR,
)
from src.fast_baseline import FastAudioClassifier, prepare_waveform_fast
from src.inference import (
    initialize_inference,
    initialize_inference_context,
    logits_to_prediction,
    predict_one_from_prepared,
)
from src.metrics import compute_eer, latency_stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate models on demo FLAC dataset")
    p.add_argument("--max-samples", type=int, default=None, help="Limit files for quick run")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-spectra", action="store_true")
    return p.parse_args()


def load_entries(max_samples: int | None) -> list[dict]:
    data = json.loads(DEMO_MANIFEST.read_text())
    entries = data["files"]
    if max_samples is not None:
        entries = entries[:max_samples]
    return entries


def ground_truth_is_bonafide(label: str) -> bool:
    return label.strip().lower() in ("real", "bonafide")


def summarize(name: str, y_true: np.ndarray, y_pred: np.ndarray, latencies: list[float]) -> dict:
    bonafide_true = y_true == 1
    acc = float(np.mean(y_true == y_pred))
    # EER on bonafide scores: use predicted bonafide prob proxy — for classification use 1/0 scores
    scores = y_pred.astype(np.float64)
    bonafide_scores = scores[bonafide_true]
    spoof_scores = scores[~bonafide_true]
    # For EER we need continuous scores; store acc primarily
    lat = latency_stats(latencies)
    return {
        "model": name,
        "accuracy": acc,
        "n": int(len(y_true)),
        "n_correct": int(np.sum(y_true == y_pred)),
        "latency_ms": lat,
    }


def eval_spectra(entries: list[dict], device: str, batch_size: int) -> dict:
    model, dev = initialize_inference(device)
    modes = ["threshold", "argmax"]
    y_true = np.asarray([1 if ground_truth_is_bonafide(e["label"]) else 0 for e in entries])

    results = {}
    for mode in modes:
        y_pred = []
        latencies = []
        batch_paths = []
        batch_waves = []
        batch_srs = []

        def flush():
            nonlocal batch_waves, batch_srs, batch_paths
            if not batch_waves:
                return
            from src.audio_preprocess import prepare_batch

            prepared = prepare_batch(batch_waves, batch_srs, mode="deterministic").to(dev)
            t0 = time.perf_counter()
            with torch.inference_mode():
                from src.model_loader import sync_device

                sync_device(dev)
                logits = model(prepared)
                sync_device(dev)
            ms = (time.perf_counter() - t0) * 1000.0
            per = ms / len(batch_waves)
            for i in range(logits.size(0)):
                pred = logits_to_prediction(
                    logits[i : i + 1], DEFAULT_THRESHOLD, decision=mode  # type: ignore[arg-type]
                )
                y_pred.append(1 if pred.label == "bonafide" else 0)
                latencies.append(per)
            batch_waves, batch_srs, batch_paths = [], [], []

        for entry in entries:
            path = PROJECT_ROOT / entry["path"]
            audio, sr = sf.read(str(path), dtype="float32")
            batch_waves.append(torch.from_numpy(np.asarray(audio, dtype=np.float32)))
            batch_srs.append(sr)
            batch_paths.append(path)
            if len(batch_waves) >= batch_size:
                flush()
        flush()

        y_pred_arr = np.asarray(y_pred, dtype=np.int64)
        acc = float(np.mean(y_true == y_pred_arr))
        results[f"spectra_{mode}"] = {
            "accuracy": acc,
            "n": len(y_true),
            "latency_ms": latency_stats(latencies),
        }
        print(f"  spectra ({mode}): acc={acc*100:.2f}%  p50={results[f'spectra_{mode}']['latency_ms']['p50_ms']:.0f}ms")

    return results


def eval_fast(name: str, model_path: Path, entries: list[dict]) -> dict:
    if not model_path.exists():
        print(f"  skip {name}: missing {model_path}")
        return {"skipped": True, "path": str(model_path)}

    clf = FastAudioClassifier.load(model_path)
    y_true, y_pred, latencies = [], [], []
    for entry in entries:
        path = PROJECT_ROOT / entry["path"]
        audio, sr = sf.read(str(path), dtype="float32")
        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        t0 = time.perf_counter()
        pred = clf.predict_one(waveform, sr)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        y_true.append(1 if ground_truth_is_bonafide(entry["label"]) else 0)
        y_pred.append(1 if pred.label == "bonafide" else 0)

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    acc = float(np.mean(y_true_arr == y_pred_arr))
    row = {"accuracy": acc, "n": len(y_true_arr), "latency_ms": latency_stats(latencies)}
    print(f"  {name}: acc={acc*100:.2f}%  p50={row['latency_ms']['p50_ms']:.1f}ms")
    return row


def main() -> None:
    args = parse_args()
    entries = load_entries(args.max_samples)
    print(f"Evaluating {len(entries)} demo clips ...\n")

    out: dict = {"n_samples": len(entries), "models": {}}

    if not args.skip_spectra:
        print("Spectra-AASIST3:")
        out["models"].update(eval_spectra(entries, args.device, args.batch_size))

    print("\nFast baseline:")
    out["models"]["fast_asvspoof_rbf"] = eval_fast("fast_asvspoof_rbf", DEFAULT_FAST_MODEL, entries)
    out["models"]["fast_demo_rbf"] = eval_fast("fast_demo_rbf", DEFAULT_DEMO_FAST_MODEL, entries)

    out["recommended"] = {
        "spectra_decision": DEFAULT_SPECTRA_DECISION,
        "demo_api_backend": "fast",
        "demo_api_model": str(DEFAULT_DEMO_FAST_MODEL),
    }

    path = RESULTS_DIR / "summary_demo_eval.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
