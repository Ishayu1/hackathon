#!/usr/bin/env python3
"""Download garystafford/deepfake-audio-detection to local FLAC files for demo testing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "demo" / "deepfake-audio-detection"
DATASET_ID = "garystafford/deepfake-audio-detection"


def main() -> None:
    real_dir = OUT_DIR / "real"
    fake_dir = OUT_DIR / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID, split="train")
    ds = ds.cast_column("audio", Audio(decode=False))

    manifest = []
    n = len(ds)
    for i, row in enumerate(ds):
        label = int(row["label"])
        subdir = real_dir if label == 0 else fake_dir
        label_name = "real" if label == 0 else "fake"

        audio = row["audio"]
        if audio.get("bytes"):
            out_path = subdir / f"{label_name}_{i:04d}.flac"
            out_path.write_bytes(audio["bytes"])
        elif audio.get("path"):
            src = Path(audio["path"])
            out_path = subdir / f"{label_name}_{i:04d}{src.suffix or '.flac'}"
            out_path.write_bytes(src.read_bytes())
        elif "array" in audio:
            out_path = subdir / f"{label_name}_{i:04d}.flac"
            sf.write(out_path, audio["array"], int(audio["sampling_rate"]), format="FLAC")
        else:
            raise ValueError(f"Unsupported audio format at index {i}")

        manifest.append(
            {"index": i, "path": str(out_path.relative_to(ROOT)), "label": label_name}
        )
        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f"  saved {i + 1}/{n}")

    meta = {
        "dataset": DATASET_ID,
        "url": "https://huggingface.co/datasets/garystafford/deepfake-audio-detection",
        "n_total": len(manifest),
        "n_real": sum(1 for m in manifest if m["label"] == "real"),
        "n_fake": sum(1 for m in manifest if m["label"] == "fake"),
        "label_map": {"0": "real", "1": "fake"},
        "files": manifest,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDone.")
    print(f"  Real: {meta['n_real']} -> {real_dir}")
    print(f"  Fake: {meta['n_fake']} -> {fake_dir}")
    print(f"  Manifest: {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
