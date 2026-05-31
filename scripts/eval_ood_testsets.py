#!/usr/bin/env python3
"""Evaluate demo fast model on out-of-domain test manifests."""

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

from src.config import DEFAULT_DEMO_FAST_MODEL, PROJECT_ROOT, RESULTS_DIR
from src.fast_baseline import FastAudioClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval fast demo model on OOD test sets")
    p.add_argument("--model-path", type=Path, default=DEFAULT_DEMO_FAST_MODEL)
    p.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Manifest JSON (default: ASVspoof + In-The-Wild OOD if present)",
    )
    return p.parse_args()


def load_manifest(path: Path) -> tuple[str, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("dataset", path.parent.name), data["files"]


def eval_manifest(name: str, entries: list[dict], clf: FastAudioClassifier) -> dict:
    y_true, y_pred = [], []
    latencies = []
    errors = []
    for entry in entries:
        path = PROJECT_ROOT / entry["path"]
        audio, sr = sf.read(str(path), dtype="float32")
        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        pred = clf.predict_one(waveform, sr)
        gt_bonafide = entry["label"] == "real"
        pred_bonafide = pred.label == "bonafide"
        y_true.append(1 if gt_bonafide else 0)
        y_pred.append(1 if pred_bonafide else 0)
        latencies.append(pred.total_ms)
        if gt_bonafide != pred_bonafide:
            errors.append(
                {
                    "path": entry["path"],
                    "label": entry["label"],
                    "predicted": pred.label,
                    "confidence": pred.confidence,
                }
            )
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    acc = float(np.mean(y_true_arr == y_pred_arr))
    n_fake = int(np.sum(y_true_arr == 0))
    n_real = int(np.sum(y_true_arr == 1))
    fake_as_real = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    real_as_fake = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    return {
        "dataset": name,
        "n": len(entries),
        "accuracy": acc,
        "n_real": n_real,
        "n_fake": n_fake,
        "fake_called_real": fake_as_real,
        "real_called_fake": real_as_fake,
        "fake_as_real_rate": float(fake_as_real / n_fake) if n_fake else 0.0,
        "latency_ms_p50": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "errors": errors[:10],
    }


def main() -> None:
    args = parse_args()
    manifests = [Path(m) for m in args.manifest]
    if not manifests:
        defaults = [
            ROOT / "data" / "ood" / "asvspoof19-la" / "manifest.json",
            ROOT / "data" / "ood" / "in-the-wild" / "manifest.json",
        ]
        manifests = [p for p in defaults if p.exists()]

    if not manifests:
        print("No OOD manifests found. Run: python scripts/download_ood_testsets.py")
        sys.exit(1)

    if not args.model_path.exists():
        print(f"Model missing: {args.model_path}")
        sys.exit(1)

    clf = FastAudioClassifier.load(args.model_path)
    results = []
    print(f"Evaluating {args.model_path.name} on OOD sets\n")
    for path in manifests:
        name, entries = load_manifest(path)
        row = eval_manifest(name, entries, clf)
        results.append(row)
        print(
            f"{name}: acc={row['accuracy']*100:.1f}%  "
            f"fake->real={row['fake_called_real']}/{row['n_fake']}  "
            f"real->fake={row['real_called_fake']}/{row['n_real']}"
        )

    out = {"model": str(args.model_path), "results": results}
    out_path = RESULTS_DIR / "summary_ood_eval.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
