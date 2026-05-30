#!/usr/bin/env python3
"""Quick MPS/CPU inference smoke test — no dataset download required."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CLIP_LEN
from src.inference import initialize_inference, predict_one_from_prepared
from src.metrics import latency_stats
from src.model_loader import sync_device


def main() -> None:
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== Spectra-AASIST3 smoke test ({device_name}) ===\n")

    t0 = time.perf_counter()
    model, device = initialize_inference(device_name)
    load_s = time.perf_counter() - t0
    print(f"Model load + warmup: {load_s:.1f}s  (one-time cost at API startup)\n")

    n_runs = 20
    infer_ms: list[float] = []

    # Use varied synthetic clips (simulates different uploads)
    for i in range(n_runs):
        clip = torch.randn(CLIP_LEN) * (0.05 + 0.01 * (i % 5))
        t1 = time.perf_counter()
        pred, ms = predict_one_from_prepared(model, device, clip)
        infer_ms.append(ms)
        if i == 0:
            print(f"Sample prediction: {pred.label} (bonafide={pred.score_bonafide:.3f}, spoof={pred.score_spoof:.3f})")

    stats = latency_stats(infer_ms)
    batch = torch.randn(4, CLIP_LEN, device=device)
    sync_device(device)
    t2 = time.perf_counter()
    with torch.inference_mode():
        model(batch)
    sync_device(device)
    batch_ms = (time.perf_counter() - t2) * 1000.0

    print(f"\n--- Single-file inference ({n_runs} runs, batch=1) ---")
    print(f"  mean:  {stats['mean_ms']:.0f} ms")
    print(f"  p50:   {stats['p50_ms']:.0f} ms")
    print(f"  p95:   {stats['p95_ms']:.0f} ms")
    print(f"  min/max: {stats['min_ms']:.0f} / {stats['max_ms']:.0f} ms")

    print(f"\n--- Batched inference (batch=4) ---")
    print(f"  total: {batch_ms:.0f} ms  (~{batch_ms / 4:.0f} ms/clip)")

    total_api = stats["p95_ms"] + 20  # ~20ms preprocess estimate
    print(f"\n--- Hackathon API estimate (p95 infer + ~20ms preprocess) ---")
    print(f"  ~{total_api:.0f} ms per upload")

    print("\n--- Verdict ---")
    if stats["p95_ms"] < 500:
        print("  FEASIBLE for live demo on this hardware.")
    elif stats["p95_ms"] < 1000:
        print("  BORDERLINE — works but users will notice ~0.5–1s wait.")
    else:
        print("  TOO SLOW for live demo — consider a lighter model.")

    if device_name == "cpu" and torch.backends.mps.is_available():
        print("  NOTE: MPS is available but not used. Re-run with MPS for ~3-4x speedup.")


if __name__ == "__main__":
    main()
