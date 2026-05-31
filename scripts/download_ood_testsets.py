#!/usr/bin/env python3
"""Download out-of-domain test clips: ASVspoof 2019 LA + In-The-Wild (subset)."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ASVSPOOF_OUT = ROOT / "data" / "ood" / "asvspoof19-la"
ITW_OUT = ROOT / "data" / "ood" / "in-the-wild"
ITW_ZIP_DIR = ITW_OUT / "_download"
ITW_ZIP_NAME = "release_in_the_wild.zip"
ITW_REPO = "mueller91/In-The-Wild"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download OOD evaluation audio clips")
    p.add_argument("--per-class", type=int, default=50, help="Clips per class for each dataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-asvspoof", action="store_true")
    p.add_argument("--skip-itw", action="store_true")
    p.add_argument("--itw-max-extract-gb", type=float, default=7.0, help="Require zip at least this size (GB) before extract")
    return p.parse_args()


def write_manifest(out_dir: Path, dataset: str, url: str, files: list[dict]) -> None:
    meta = {
        "dataset": dataset,
        "url": url,
        "purpose": "out-of-domain evaluation (not used in demo model training)",
        "n_total": len(files),
        "n_real": sum(1 for f in files if f["label"] == "real"),
        "n_fake": sum(1 for f in files if f["label"] == "fake"),
        "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def label_is_bonafide(key) -> bool:
    if isinstance(key, str):
        k = key.strip().lower()
        return k in ("bonafide", "real", "0")
    if isinstance(key, (int, np.integer)):
        return int(key) == 0
    raise ValueError(f"Unsupported label: {key!r}")


def stratified_pick(n_bonafide: int, n_spoof: int, max_per_class: int, seed: int) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    nb = min(n_bonafide, max_per_class)
    ns = min(n_spoof, max_per_class)
    return int(nb), int(ns)


def save_audio_bytes(data: bytes, out_path: Path, target_sr: int = 16000) -> None:
    arr, sr = sf.read(io.BytesIO(data), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != target_sr:
        import torchaudio

        t = torch.from_numpy(np.asarray(arr, dtype=np.float32)).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, target_sr)
        arr = t.squeeze(0).numpy()
        sr = target_sr
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, arr, sr, format="FLAC")


def export_asvspoof(per_class: int, seed: int) -> None:
    real_dir = ASVSPOOF_OUT / "real"
    fake_dir = ASVSPOOF_OUT / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting ASVspoof 2019 LA validation subset (lab spoof attacks) ...")
    ds = load_dataset("Bisher/ASVspoof_2019_LA", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))

    labels = np.asarray([0 if label_is_bonafide(row["key"]) else 1 for row in ds])
    bonafide_idx = np.where(labels == 0)[0]
    spoof_idx = np.where(labels == 1)[0]
    rng = np.random.default_rng(seed)
    n_b = min(len(bonafide_idx), per_class)
    n_s = min(len(spoof_idx), per_class)
    pick_b = rng.choice(bonafide_idx, n_b, replace=False)
    pick_s = rng.choice(spoof_idx, n_s, replace=False)
    picks = [(int(i), "real") for i in pick_b] + [(int(i), "fake") for i in pick_s]
    rng.shuffle(picks)

    manifest: list[dict] = []
    for j, (idx, label_name) in enumerate(picks):
        row = ds[idx]
        subdir = real_dir if label_name == "real" else fake_dir
        out_path = subdir / f"{label_name}_{j:04d}.flac"
        audio = row["audio"]
        if audio.get("bytes"):
            save_audio_bytes(audio["bytes"], out_path)
        elif audio.get("path"):
            save_audio_bytes(Path(audio["path"]).read_bytes(), out_path)
        else:
            raise ValueError(f"Unsupported audio row at index {idx}")
        manifest.append(
            {
                "index": j,
                "source_index": idx,
                "path": str(out_path.relative_to(ROOT)),
                "label": label_name,
                "split": "validation",
            }
        )
        if (j + 1) % 25 == 0 or j + 1 == len(picks):
            print(f"  saved {j + 1}/{len(picks)}")

    write_manifest(
        ASVSPOOF_OUT,
        "Bisher/ASVspoof_2019_LA",
        "https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA",
        manifest,
    )
    print(f"ASVspoof OOD set -> {ASVSPOOF_OUT} ({len(manifest)} clips)")


def _itw_label_from_name(name: str) -> str | None:
    lower = name.lower()
    parts = Path(name).parts
    for part in parts:
        p = part.lower()
        if p in {"real", "bonafide", "genuine"}:
            return "real"
        if p in {"fake", "spoof", "deepfake", "synthetic"}:
            return "fake"
    if "_real" in lower or lower.startswith("real_") or lower.endswith("_real.wav"):
        return "real"
    if "_fake" in lower or lower.startswith("fake_") or lower.endswith("_fake.wav"):
        return "fake"
    return None


def _itw_label_from_csv(zip_path: Path, member_name: str, csv_cache: dict[str, dict[str, str]]) -> str | None:
    """Map wav filename to label using meta.csv inside the zip if present."""
    base = Path(member_name).name
    stem = Path(member_name).stem
    for csv_name, mapping in csv_cache.items():
        if stem in mapping:
            lab = mapping[stem].lower()
            if lab in {"real", "bonafide", "bona-fide", "bonafide", "0"}:
                return "real"
            if lab in {"fake", "spoof", "1"}:
                return "fake"
        if base in mapping:
            lab = mapping[base].lower()
            if lab in {"real", "bonafide", "bona-fide", "bonafide", "0"}:
                return "real"
            if lab in {"fake", "spoof", "1"}:
                return "fake"
    return None


def _load_csv_mappings(zip_path: Path) -> dict[str, dict[str, str]]:
    import csv

    mappings: dict[str, dict[str, str]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if not info.filename.lower().endswith(".csv"):
                continue
            with zf.open(info) as f:
                text = f.read().decode("utf-8", errors="replace").splitlines()
            if not text:
                continue
            reader = csv.DictReader(text)
            if not reader.fieldnames:
                continue
            fields = {h.lower(): h for h in reader.fieldnames}
            file_key = fields.get("file") or fields.get("filename") or fields.get("path")
            label_key = fields.get("label") or fields.get("class") or fields.get("target")
            if not file_key or not label_key:
                continue
            table: dict[str, str] = {}
            for row in reader:
                table[str(row[file_key]).strip()] = str(row[label_key]).strip()
            mappings[info.filename] = table
    return mappings


def ensure_itw_zip() -> Path:
    ITW_ZIP_DIR.mkdir(parents=True, exist_ok=True)
    local = ITW_ZIP_DIR / ITW_ZIP_NAME
    if local.exists() and local.stat().st_size > 1_000_000:
        return local
    print(f"Downloading {ITW_REPO}/{ITW_ZIP_NAME} (~8 GB) ...")
    path = hf_hub_download(
        repo_id=ITW_REPO,
        repo_type="dataset",
        filename=ITW_ZIP_NAME,
        local_dir=str(ITW_ZIP_DIR),
    )
    return Path(path)


def export_in_the_wild(per_class: int, seed: int, min_gb: float) -> None:
    zip_path = ensure_itw_zip()
    size_gb = zip_path.stat().st_size / (1024**3)
    if size_gb < min_gb * 0.95:
        print(f"In-The-Wild zip still downloading ({size_gb:.2f} GB). Re-run this script when complete.")
        return

    real_dir = ITW_OUT / "real"
    fake_dir = ITW_OUT / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning In-The-Wild zip for labeled audio ...")
    csv_maps = _load_csv_mappings(zip_path)
    rng = np.random.default_rng(seed)

    real_members: list[zipfile.ZipInfo] = []
    fake_members: list[zipfile.ZipInfo] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            lower = info.filename.lower()
            if not lower.endswith((".wav", ".flac", ".mp3")):
                continue
            label = _itw_label_from_name(info.filename) or _itw_label_from_csv(zip_path, info.filename, csv_maps)
            if label == "real":
                real_members.append(info)
            elif label == "fake":
                fake_members.append(info)

    if not real_members or not fake_members:
        raise RuntimeError(
            "Could not infer real/fake labels from In-The-Wild zip paths. "
            "Check zip layout or add a metadata CSV."
        )

    rng.shuffle(real_members)
    rng.shuffle(fake_members)
    pick_real = real_members[: min(per_class, len(real_members))]
    pick_fake = fake_members[: min(per_class, len(fake_members))]

    manifest: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        for label_name, members in (("real", pick_real), ("fake", pick_fake)):
            subdir = real_dir if label_name == "real" else fake_dir
            for j, info in enumerate(members):
                out_path = subdir / f"{label_name}_{j:04d}.flac"
                data = zf.read(info.filename)
                save_audio_bytes(data, out_path)
                manifest.append(
                    {
                        "index": len(manifest),
                        "path": str(out_path.relative_to(ROOT)),
                        "label": label_name,
                        "source_member": info.filename,
                    }
                )
            print(f"  extracted {len(members)} {label_name} clips")

    write_manifest(
        ITW_OUT,
        ITW_REPO,
        f"https://huggingface.co/datasets/{ITW_REPO}",
        manifest,
    )
    print(f"In-The-Wild OOD set -> {ITW_OUT} ({len(manifest)} clips)")


def main() -> None:
    args = parse_args()
    if not args.skip_asvspoof:
        export_asvspoof(args.per_class, args.seed)
    if not args.skip_itw:
        export_in_the_wild(args.per_class, args.seed, args.itw_max_extract_gb)
    print("\nUpload test clips from:")
    print(f"  {ASVSPOOF_OUT}/real  and  {ASVSPOOF_OUT}/fake")
    print(f"  {ITW_OUT}/real  and  {ITW_OUT}/fake")
    print("\nBatch eval: python scripts/eval_ood_testsets.py")


if __name__ == "__main__":
    main()
