# Spectra-AASIST3 V1 — Audio Deepfake Detection

Inference pipeline using [Spectra-AASIST3](https://huggingface.co/lab260/Spectra-AASIST3) with ASVspoof 2019 LA evaluation and a FastAPI upload endpoint.

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

## API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
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
