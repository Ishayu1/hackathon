#!/usr/bin/env python3
"""Compare Spectra-AASIST3 vs fast baseline on speed and accuracy."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PYTHON = sys.executable


def run(cmd: list[str]) -> int:
    print(f"\n>>> {' '.join(cmd)}\n")
    return subprocess.call(cmd, cwd=ROOT)


def load_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def main() -> None:
    train_n = 5000
    eval_n = 2000
    split = "validation"

    print("=" * 60)
    print("Hackathon backend comparison")
    print(f"  Train samples: {train_n}  |  Eval samples: {eval_n}  |  Split: {split}")
    print("=" * 60)

    # Fast baseline: train + eval
    rc_fast = run(
        [
            PYTHON,
            "scripts/eval_fast_baseline.py",
            "--train-split",
            "train",
            "--eval-split",
            split,
            "--max-train-samples",
            str(train_n),
            "--max-eval-samples",
            str(eval_n),
            "--feature-type",
            "mfcc",
            "--model-type",
            "logistic_regression",
        ]
    )

    # Spectra: inference-only on same eval split (cached dataset, no streaming)
    rc_spectra = run(
        [
            PYTHON,
            "scripts/eval_asvspoof.py",
            "--split",
            split,
            "--max-samples",
            str(eval_n),
            "--batch-size",
            "8",
            "--device",
            "mps",
        ]
    )

    # Spectra inference-only speed (synthetic, no dataset)
    rc_speed = run([PYTHON, "scripts/smoke_test_mps.py"])

    fast_summary = load_json(
        RESULTS / "summary_fast_mfcc_logistic_regression_validation.json"
    )
    spectra_summary = load_json(RESULTS / f"summary_{split}.json")

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    if fast_summary:
        fl = fast_summary.get("feature_latency", {})
        il = fast_summary.get("inference_latency", {})
        print("\n[Fast baseline] MFCC + Logistic Regression")
        print(f"  EER:              {fast_summary['eer_percent']:.4f}%")
        print(f"  Accuracy @ 0.5:   {fast_summary['accuracy_at_threshold'] * 100:.2f}%")
        print(f"  Feature p95:      {fl.get('p95_ms', 0):.2f} ms")
        print(f"  Classifier p95:   {il.get('p95_ms', 0):.4f} ms")
        print(f"  End-to-end ~:     {fl.get('p95_ms', 0) + il.get('p95_ms', 0):.2f} ms/clip")
        print(f"  Eval throughput:  {fast_summary.get('eval_samples_per_sec', 0):.2f} samples/s")
        print(f"  Needs training:   yes ({fast_summary.get('n_train', 0)} samples)")
    else:
        print("\n[Fast baseline] — no results (eval failed)")

    if spectra_summary:
        lat = spectra_summary.get("latency", {})
        print("\n[Spectra-AASIST3] pretrained, MPS")
        print(f"  EER:              {spectra_summary['eer_percent']:.4f}%")
        print(f"  Accuracy @ thr:   {spectra_summary['accuracy_at_threshold'] * 100:.2f}%")
        print(f"  Inference p95:    {lat.get('p95_ms', 0):.1f} ms")
        print(f"  Throughput:       {spectra_summary.get('samples_per_sec', 0):.2f} samples/s")
        print(f"  Model card EER:   {spectra_summary.get('model_card_baseline_eer_percent', 0.723)}%")
        print(f"  Needs training:   no")
    else:
        print("\n[Spectra-AASIST3] — no results (eval failed)")

    print("\n--- Hackathon recommendation ---")
    if fast_summary and spectra_summary:
        fast_eer = fast_summary["eer_percent"]
        spec_eer = spectra_summary["eer_percent"]
        fast_e2e = fast_summary["feature_latency"]["p95_ms"] + fast_summary["inference_latency"]["p95_ms"]
        spec_p95 = spectra_summary["latency"]["p95_ms"]

        if spec_eer < fast_eer * 0.5 and spec_p95 < 500:
            print("  Spectra wins on accuracy; speed is OK on MPS → use Spectra for demo.")
        elif fast_e2e < spec_p95 * 0.3 and fast_eer < 5.0:
            print("  Fast baseline wins on speed with decent EER → use fast for live demo.")
        elif spec_eer < fast_eer and spec_p95 < 1000:
            print("  Spectra: better accuracy, acceptable latency → prefer Spectra.")
        else:
            print("  Fast baseline: much faster; Spectra: better accuracy if EER lower.")
            print(f"    EER gap: Spectra {spec_eer:.2f}% vs Fast {fast_eer:.2f}%")
            print(f"    Speed gap: ~{spec_p95:.0f} ms vs ~{fast_e2e:.0f} ms per clip")

    if rc_fast != 0 or rc_spectra != 0 or rc_speed != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
