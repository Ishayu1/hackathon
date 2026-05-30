#!/usr/bin/env python3
"""
Compare sklearn classifiers on the SAME cached MFCC features (extract once, train many).

Usage:
  python scripts/compare_fast_classifiers.py              # extract cache if missing, then compare
  python scripts/compare_fast_classifiers.py --cache-only # skip extraction, use existing cache
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from datasets import Audio, load_dataset
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_fast_baseline import build_features  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.fast_baseline import FastAudioClassifier  # noqa: E402
from src.metrics import accuracy_at_threshold, compute_eer, latency_stats  # noqa: E402

SPECTRA_TEST_EER = 0.723
CACHE_TRAIN = RESULTS_DIR / "feature_cache_mfcc_train.npz"
CACHE_EVAL = RESULTS_DIR / "feature_cache_mfcc_validation.npz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare fast classifiers on cached features")
    p.add_argument("--max-train", type=int, default=8000)
    p.add_argument("--max-eval", type=int, default=3000)
    p.add_argument("--eval-split", default="validation", choices=["validation", "test"])
    p.add_argument("--cache-only", action="store_true", help="Require existing feature cache")
    p.add_argument(
        "--classifiers",
        nargs="+",
        default=["logistic_regression", "random_forest", "linear_svc", "rbf_svc", "gradient_boosting"],
    )
    p.add_argument("--rbf-max-train", type=int, default=5000, help="Cap train size for RBF SVM")
    return p.parse_args()


def build_classifier(name: str):
    if name == "logistic_regression":
        return Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1500))]
        )
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    if name == "linear_svc":
        return Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(random_state=42))])
    if name == "rbf_svc":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)),
            ]
        )
    if name == "gradient_boosting":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)),
            ]
        )
    raise ValueError(f"Unknown classifier: {name}")


def decision_scores(clf, x: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(x)[:, 1]
    if hasattr(clf, "decision_function"):
        return np.asarray(clf.decision_function(x), dtype=np.float64)
    raise RuntimeError("Classifier has no predict_proba or decision_function")


def load_or_build_cache(
    split: str,
    cache_path: Path,
    max_samples: int,
    *,
    require_cache: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.exists():
        data = np.load(cache_path)
        print(f"  loaded cache {cache_path.name}: {data['x'].shape[0]} samples")
        return data["x"], data["y"]

    if require_cache:
        raise FileNotFoundError(f"Cache missing: {cache_path}. Run without --cache-only first.")

    print(f"  building cache for split={split} (max={max_samples}) ...")
    ds = load_dataset("Bisher/ASVspoof_2019_LA", split=split)
    ds = ds.cast_column("audio", Audio(decode=False))
    fe = FastAudioClassifier(feature_type="mfcc", model_type="logistic_regression")
    x, y, _ = build_features(ds, max_samples, fe, stratified=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=x, y=y)
    print(f"  saved {cache_path}")
    return x, y


def eval_classifier(
    name: str, clf, x_train, y_train, x_eval, y_eval, *, rbf_max_train: int = 5000
) -> dict:
    x_tr, y_tr = x_train, y_train
    if name == "rbf_svc":
        cap = rbf_max_train
        rng = np.random.default_rng(42)
        idx0 = np.where(y_train == 1)[0]
        idx1 = np.where(y_train == 0)[0]
        ratio = len(idx0) / len(y_train)
        n0 = min(len(idx0), max(1, int(round(cap * ratio))))
        n1 = min(len(idx1), cap - n0)
        sel = np.concatenate([rng.choice(idx0, n0, replace=False), rng.choice(idx1, n1, replace=False)])
        rng.shuffle(sel)
        x_tr, y_tr = x_train[sel], y_train[sel]

    t0 = time.perf_counter()
    clf.fit(x_tr, y_tr)
    train_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    scores = decision_scores(clf, x_eval)
    infer_ms = (time.perf_counter() - t1) * 1000.0
    per_sample_ms = infer_ms / max(len(x_eval), 1)

    bonafide = scores[y_eval == 1]
    spoof = scores[y_eval == 0]
    eer, eer_thr = compute_eer(bonafide, spoof)
    thr = 0.5 if hasattr(clf, "predict_proba") else 0.0
    acc = accuracy_at_threshold(bonafide, spoof, thr)

    return {
        "classifier": name,
        "n_train_used": int(len(y_tr)),
        "n_eval": int(len(y_eval)),
        "eer_percent": float(eer * 100.0),
        "eer_threshold": float(eer_thr),
        "accuracy_at_threshold": float(acc),
        "threshold": thr,
        "train_ms": train_ms,
        "eval_infer_ms": infer_ms,
        "infer_ms_per_sample": per_sample_ms,
    }


def main() -> None:
    args = parse_args()
    cache_eval = RESULTS_DIR / f"feature_cache_mfcc_{args.eval_split}.npz"
    if args.eval_split == "validation":
        cache_eval = CACHE_EVAL

    print("=== Fast classifier shootout (shared MFCC features) ===\n")

    x_train, y_train = load_or_build_cache(
        "train", CACHE_TRAIN, args.max_train, require_cache=args.cache_only
    )
    x_eval, y_eval = load_or_build_cache(
        args.eval_split, cache_eval, args.max_eval, require_cache=args.cache_only
    )

    # trim if cache is larger than requested max
    if args.max_train and x_train.shape[0] > args.max_train:
        x_train, y_train = x_train[: args.max_train], y_train[: args.max_train]
    if args.max_eval and x_eval.shape[0] > args.max_eval:
        x_eval, y_eval = x_eval[: args.max_eval], y_eval[: args.max_eval]

    print(f"\nTrain matrix: {x_train.shape}  Eval matrix: {x_eval.shape}\n")

    results = []
    for name in args.classifiers:
        print(f"--- {name} ---")
        try:
            clf = build_classifier(name)
            row = eval_classifier(
                name, clf, x_train, y_train, x_eval, y_eval, rbf_max_train=args.rbf_max_train
            )
            results.append(row)
            print(
                f"  EER={row['eer_percent']:.4f}%  acc={row['accuracy_at_threshold']*100:.2f}%  "
                f"train={row['train_ms']:.0f}ms  infer={row['infer_ms_per_sample']:.3f}ms/sample"
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results.append({"classifier": name, "error": str(exc)})

    results.sort(key=lambda r: r.get("eer_percent", float("inf")))

    out = {
        "n_train": int(x_train.shape[0]),
        "n_eval": int(x_eval.shape[0]),
        "eval_split": args.eval_split,
        "spectra_published_test_eer_percent": SPECTRA_TEST_EER,
        "results": results,
    }
    out_path = RESULTS_DIR / "classifier_shootout.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("\n=== RANKING (lower EER is better) ===")
    print(f"{'Classifier':<22} {'EER%':>8} {'Acc%':>8} {'ms/sample':>10} {'vs Spectra':>12}")
    print("-" * 64)
    for r in results:
        if "error" in r:
            print(f"{r['classifier']:<22} {'FAILED':>8}")
            continue
        vs = "faster" if r["infer_ms_per_sample"] < 190 else "slower"
        print(
            f"{r['classifier']:<22} {r['eer_percent']:8.4f} "
            f"{r['accuracy_at_threshold']*100:8.2f} {r['infer_ms_per_sample']:10.4f} "
            f"{vs} (~190ms)"
        )
    print(f"\nSpectra-AASIST3 published test EER: {SPECTRA_TEST_EER:.4f}%")
    print(f"Saved: {out_path}")

    best = next((r for r in results if "eer_percent" in r), None)
    if best:
        print(f"\nBest fast classifier here: {best['classifier']} @ {best['eer_percent']:.4f}% EER")


if __name__ == "__main__":
    main()
