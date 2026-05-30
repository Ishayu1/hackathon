# Spectra-AASIST3 V1 — Audio Deepfake Detection

Inference pipeline using [Spectra-AASIST3](https://huggingface.co/lab260/Spectra-AASIST3) with ASVspoof 2019 LA evaluation and a FastAPI upload endpoint.

Also includes a **fast hackathon fallback**:
- `16k mono -> 4s crop/pad -> MFCC/LFCC + spectral features -> sklearn classifier`
- Supports `LogisticRegression`, `RandomForestClassifier`, and `LinearSVC`.

## Setup

```bash
cd Hackathon
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads ~2.5 GB (Spectra-AASIST3 weights + Wav2Vec2 encoder).

## Batch evaluation (ASVspoof 2019 LA)

Dataset: [Bisher/ASVspoof_2019_LA](https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA) (~7.5 GB).

```bash
# Smoke test
python scripts/eval_asvspoof.py --split validation --max-samples 100

# Dev split EER
python scripts/eval_asvspoof.py --split validation --batch-size 16

# Full test split EER (compare to model card baseline 0.723%)
python scripts/eval_asvspoof.py --split test --batch-size 16
```

Results are written to `results/scores_{split}.tsv` and `results/summary_{split}.json`.

## Fast baseline (recommended hackathon fallback)

Train on ASVspoof train split and evaluate quickly:

```bash
# Fast default: MFCC + Logistic Regression
python scripts/eval_fast_baseline.py \
  --train-split train \
  --eval-split validation \
  --max-train-samples 8000 \
  --max-eval-samples 3000 \
  --feature-type mfcc \
  --model-type logistic_regression

# Alternative: MFCC + Random Forest
python scripts/eval_fast_baseline.py \
  --feature-type mfcc \
  --model-type random_forest
```

Outputs:
- model artifact: `results/fast_baseline_{feature}_{model}.joblib`
- metrics: `results/summary_fast_{feature}_{model}_{split}.json`
- scores: `results/scores_fast_{feature}_{model}_{split}.tsv`

## API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Use Spectra (default):
- `MODEL_BACKEND=spectra`

Use fast baseline:

```bash
MODEL_BACKEND=fast \
FAST_MODEL_PATH=results/fast_baseline_mfcc_rbf_svc_demo.joblib \
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/classify -F "file=@sample.wav"
```

## Fast Baseline Explainability

For `MODEL_BACKEND=fast`, `/classify` includes an `explanation` object generated from the same 88 MFCC/LFCC and spectral features used by the classifier. The method compares the uploaded clip's feature values with bonafide and spoof class profiles computed from the Gary Stafford demo train split only (`scripts/build_feature_profiles.py`). Top signals are ranked by which class profile they are closer to and are phrased as corpus similarity, not causal proof. Spectra/Wav2Vec2 responses return `explanation: null`; MFCC explanations are not claimed for that backend. Rebuild profiles with:

```bash
python scripts/build_feature_profiles.py
```

## Go / no-go criteria

| Signal | Green | Red |
|--------|-------|-----|
| EER (ASVspoof19 LA test) | < ~1.5% | > 3–5% |
| API latency p95 (GPU) | < 500 ms | > 2 s |
| API latency p95 (CPU) | < 2 s | > 5 s |

Model card baseline: **0.723% EER** on ASVspoof19 LA.

## Frontend (Mission Audio Triage Dashboard)

```bash
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173`.
