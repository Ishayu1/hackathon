# Evaluation Guide

Scripts and workflows for measuring model performance on ASVspoof, demo, and out-of-domain datasets.

---

## Prerequisites

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Most eval scripts write results to `results/` as JSON summaries and TSV score files.

---

## ASVspoof 2019 LA

**Dataset:** [Bisher/ASVspoof_2019_LA](https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA) (~7.5 GB, auto-downloaded via HuggingFace `datasets`).

**Metric:** Equal Error Rate (EER) on bonafide vs. spoof detection.

```bash
# Smoke test (100 clips)
python scripts/eval_asvspoof.py --split validation --max-samples 100

# Full validation split
python scripts/eval_asvspoof.py --split validation --batch-size 16

# Full test split (compare to 0.723% published baseline)
python scripts/eval_asvspoof.py --split test --batch-size 16
```

**Outputs:**
- `results/scores_{split}.tsv` — per-file scores
- `results/summary_{split}.json` — EER, latency stats

---

## Fast Baseline

Train classical MFCC/LFCC classifiers on ASVspoof and measure EER/latency.

```bash
# Default: MFCC + RBF SVM
python scripts/eval_fast_baseline.py \
  --train-split train \
  --eval-split validation \
  --max-train-samples 8000 \
  --max-eval-samples 3000

# Alternative classifiers
python scripts/eval_fast_baseline.py --model-type random_forest
python scripts/eval_fast_baseline.py --model-type logistic_regression
python scripts/eval_fast_baseline.py --model-type linear_svc
```

**Outputs:**
- `results/fast_baseline_{feature}_{model}.joblib`
- `results/summary_fast_{feature}_{model}_{split}.json`
- `results/scores_fast_{feature}_{model}_{split}.tsv`

### Full pipeline (train + eval both splits)

```bash
python scripts/run_full_fast_pipeline.py
```

### Classifier shootout

```bash
python scripts/compare_fast_classifiers.py
python scripts/compare_backends.py
```

---

## Demo Dataset (Gary Stafford)

Local dataset: 1,866 FLAC clips (933 real / 933 TTS).

```bash
# Download if missing
python scripts/download_demo_dataset.py

# Train demo-tuned RBF SVM (80/20 stratified split)
python scripts/train_demo_rbf.py

# Batch eval all backends
python scripts/eval_demo_dataset.py
```

**Key results** (see `results/summary_demo_eval.json`):

| Model | Accuracy | Latency p50 |
|-------|----------|-------------|
| Spectra + argmax | 92.6% | ~185 ms |
| Fast RBF (demo-trained) | 99.0% | ~7 ms |

---

## Out-of-Domain (OOD) Evaluation

Held-out clips **not** used in demo training.

```bash
# ASVspoof validation subset (~100 clips)
python scripts/download_ood_testsets.py --skip-itw --per-class 50

# In-The-Wild deepfakes (~8 GB)
python scripts/download_ood_testsets.py --per-class 50

# Eval demo model on OOD manifests
python scripts/eval_ood_testsets.py
```

**Data locations:**
- `data/ood/asvspoof19-la/{real,fake}/`
- `data/ood/in-the-wild/{real,fake}/`

### Pooled model (demo + ASVspoof)

If OOD accuracy is poor:

```bash
python scripts/train_pooled_rbf.py
python scripts/eval_ood_testsets.py  # point --model-path to pooled joblib
```

---

## Explainability Profiles

Build feature profiles for fast-model XAI:

```bash
python scripts/build_feature_profiles.py
```

Output: `results/fast_demo_feature_profiles.json`

---

## Triage Rule Audit

Regression test for keyword classifier:

```bash
python phrase_audit.py
```

Runs canned benign, ambiguous, and distress phrases through `classify_message`.

---

## Smoke Tests

```bash
# MPS/GPU availability
python scripts/smoke_test_mps.py

# API health (with server running)
curl http://localhost:8000/health

# Quick classify
curl -X POST http://localhost:8000/classify \
  -F "file=@data/demo/deepfake-audio-detection/fake/fake_0500.flac"
```

---

## Interpreting Results

| Metric | Good | Concerning |
|--------|------|------------|
| ASVspoof EER (Spectra) | < 1.5% | > 3–5% |
| Demo accuracy (fast RBF) | > 95% | < 80% |
| API latency p95 (fast) | < 50 ms | > 200 ms |
| API latency p95 (Spectra) | < 500 ms | > 2 s |
| OOD accuracy | Domain-dependent | Large drop vs. demo |

Published Spectra baseline: **0.723% EER** on ASVspoof 2019 LA test.
