# Spectra-AASIST3 V1 — Audio Deepfake Detection

Inference pipeline using [Spectra-AASIST3](https://huggingface.co/lab260/Spectra-AASIST3) with ASVspoof 2019 LA evaluation and a FastAPI upload endpoint.

Also includes a **fast hackathon fallback**:
- `16k mono -> 4s crop/pad -> MFCC features -> RBF SVM (default) or other sklearn classifiers`

## Setup

```bash
cd Hackathon
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads ~2.5 GB (Spectra-AASIST3 weights + Wav2Vec2 encoder).

## Full baseline results (ASVspoof 2019 LA, full splits)

Train on **all 25,380** train clips; evaluate on full **validation (dev)** and **test (eval)** splits:

```bash
python scripts/run_full_fast_pipeline.py
```

| Model | Split | EER | Latency (p95) |
|-------|-------|-----|---------------|
| **Fast baseline** (MFCC + RBF SVM) | dev (validation) | **~10%** (8k/3k shootout) | ~6 ms |
| **Fast baseline** (MFCC + LR, full) | test (eval) | **21.43%** | ~6.6 ms |
| **Spectra-AASIST3** (published) | test (eval) | **0.723%** | ~190 ms (MPS) |

Full comparison: `results/comparison_fast_vs_spectra.json`  
Trained model (default): `results/fast_baseline_mfcc_rbf_svc.joblib`

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
# Default: MFCC + RBF SVM
python scripts/eval_fast_baseline.py \
  --train-split train \
  --eval-split validation \
  --max-train-samples 8000 \
  --max-eval-samples 3000

# Alternatives
python scripts/eval_fast_baseline.py --model-type random_forest
python scripts/eval_fast_baseline.py --model-type logistic_regression
```

Outputs:
- model artifact: `results/fast_baseline_{feature}_{model}.joblib`
- metrics: `results/summary_fast_{feature}_{model}_{split}.json`
- scores: `results/scores_fast_{feature}_{model}_{split}.tsv`

## Demo dataset evaluation (Gary Stafford TTS)

Local demo set: **1,866** FLAC clips (933 real YouTube / 933 modern TTS). Train a demo-tuned fast model, then batch-eval all backends:

```bash
# Train RBF SVM on 80/20 stratified split (uses feature cache after first run)
python scripts/train_demo_rbf.py

# Batch eval: Spectra (threshold + argmax) + fast ASVspoof RBF + fast demo RBF
python scripts/eval_demo_dataset.py
```

| Model | Demo accuracy | Latency p50 |
|-------|---------------|-------------|
| Spectra + **threshold** (ASVspoof cutoff) | 62.3% | ~181 ms |
| Spectra + **argmax** | 92.6% | ~185 ms |
| Fast RBF (ASVspoof-trained) | 47.7% | ~7 ms |
| **Fast RBF (demo-trained)** | **99.0%** | **~7 ms** |

Demo-tuned model: `results/fast_baseline_mfcc_rbf_svc_demo.joblib`  
Full metrics: `results/summary_demo_eval.json`

**Hackathon recommendation:** use **fast demo RBF** for live uploads (speed + domain match). Cite Spectra **0.723% EER on ASVspoof** for accuracy narrative; use `MODEL_BACKEND=spectra` with `SPECTRA_DECISION=argmax` when you want the neural model on demo clips.

## API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Defaults (demo-ready):
- `MODEL_BACKEND=fast` — loads demo-tuned RBF if `results/fast_baseline_mfcc_rbf_svc_demo.joblib` exists
- `FAST_MODEL_PATH=...` — override fast model artifact
- `SPECTRA_DECISION=argmax` — when using Spectra (fixes threshold mismatch on modern TTS)

Use Spectra neural backend:

```bash
MODEL_BACKEND=spectra SPECTRA_DECISION=argmax uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Use ASVspoof-trained fast baseline instead of demo-tuned:

```bash
FAST_MODEL_PATH=results/fast_baseline_mfcc_rbf_svc.joblib \
MODEL_BACKEND=fast uvicorn api.main:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/classify -F "file=@sample.wav"
```

## Go / no-go criteria

| Signal | Green | Red |
|--------|-------|-----|
| EER (ASVspoof19 LA test) | < ~1.5% | > 3–5% |
| API latency p95 (GPU) | < 500 ms | > 2 s |
| API latency p95 (CPU) | < 2 s | > 5 s |

Model card baseline: **0.723% EER** on ASVspoof19 LA.

## Demo test clips

Local copy of [garystafford/deepfake-audio-detection](https://huggingface.co/datasets/garystafford/deepfake-audio-detection) (~1.9k FLAC files):

```bash
python scripts/download_demo_dataset.py   # re-download if needed
```

```
data/demo/deepfake-audio-detection/
  real/   # 933 authentic YouTube speech clips
  fake/   # 933 TTS clips (ElevenLabs, Polly, Speechify, etc.)
  manifest.json
```

Quick classify test:

```bash
curl -X POST http://localhost:8000/classify \
  -F "file=@data/demo/deepfake-audio-detection/real/real_0000.flac"

curl -X POST http://localhost:8000/classify \
  -F "file=@data/demo/deepfake-audio-detection/fake/fake_0500.flac"
```

## Frontend (Mission Audio Triage Dashboard)

Run **both** the API and the Vite dev server:

```bash
# Terminal 1 — backend (demo RBF default)
uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — frontend (proxies /api → :8000)
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`, click **Upload Audio**, and pick a `.flac` / `.wav` clip. The UI calls `POST /classify` and renders label, spoof score, latency, and risk band.

Optional: point at a remote API with `VITE_API_BASE=http://your-host:8000 npm run dev`.
