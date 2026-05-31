# SignalShield AI — Mission Audio Triage

Detect synthetic speech, acoustic duress, and operational content from uploaded or live audio. Built for hackathon demos with both a high-accuracy neural baseline ([Spectra-AASIST3](https://huggingface.co/lab260/Spectra-AASIST3)) and a sub-10 ms classical fallback (MFCC + RBF SVM).

**Live dashboard:** React frontend + FastAPI backend. Upload a clip or record live → get authenticity score, duress analysis, transcript triage, and operator recommendations in seconds.

---

## Features

| Capability | Backend | Latency |
|------------|---------|---------|
| **Deepfake detection (neural)** | Spectra-AASIST3 + Wav2Vec2 | ~190 ms |
| **Deepfake detection (fast)** | MFCC/LFCC + RBF SVM | ~7 ms |
| **Acoustic duress** | Wav2Vec2 + BiLSTM | ~45 ms |
| **Speech transcription** | faster-whisper (`tiny.en`) | ~0.3× realtime |
| **Message triage** | Rule-based keyword classifier | < 1 ms |
| **Explainability** | Class-profile comparison (fast model) | Included in `/classify` |

---

## Quick Start

### 1. Backend setup

```bash
cd Hackathon
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First Spectra run downloads ~2.5 GB (model weights + Wav2Vec2 encoder). The default **fast** backend uses pre-trained `.joblib` artifacts in `results/` and does not require this download.

Place duress weights at project root (optional but recommended):

```
temporal_bilstm_duress.pth
```

### 2. Start the API

```bash
# Default: demo-tuned fast RBF (~7 ms, 99% on demo dataset)
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Verify:

```bash
curl http://localhost:8000/health
```

### 3. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). Upload a `.flac` / `.wav` clip or use **Record Live**.

The Vite dev server proxies `/api/*` → `http://127.0.0.1:8000`.

---

## Configuration

Environment variables (set before `uvicorn`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_BACKEND` | `fast` | `fast` or `spectra` |
| `FAST_MODEL_PATH` | demo `.joblib` if present | Override fast model artifact |
| `FAST_PROFILE_PATH` | `results/fast_demo_feature_profiles.json` | XAI feature profiles |
| `SPECTRA_DECISION` | `argmax` | `argmax` or `threshold` (spectra only) |
| `DURESS_ENABLED` | `1` | Set `0` to disable duress |
| `DURESS_MODEL_PATH` | `temporal_bilstm_duress.pth` | Duress weights |
| `DURESS_THRESHOLD` | `0.5` | Duress decision cutoff |
| `TRANSCRIBER_MODEL_SIZE` | `tiny.en` | faster-whisper model |
| `TRANSCRIBER_DEVICE` | `cpu` | Whisper device |
| `TRANSCRIBER_COMPUTE_TYPE` | `int8` | Quantization type |
| `TRANSCRIBER_VAD_FILTER` | `0` | Enable for long/noisy clips |
| `CORS_ORIGINS` | `localhost:5173` | Allowed frontend origins |

**Examples:**

```bash
# Spectra neural backend (best ASVspoof accuracy)
MODEL_BACKEND=spectra SPECTRA_DECISION=argmax uvicorn api.main:app --host 0.0.0.0 --port 8000

# ASVspoof-trained fast model instead of demo-tuned
FAST_MODEL_PATH=results/fast_baseline_mfcc_rbf_svc.joblib \
MODEL_BACKEND=fast uvicorn api.main:app --host 0.0.0.0 --port 8000

# Remote frontend
VITE_API_BASE=http://your-host:8000 npm run dev
```

---

## API Usage

### Classify (deepfake + duress)

```bash
curl -X POST http://localhost:8000/classify \
  -F "file=@data/demo/deepfake-audio-detection/real/real_0000.flac"
```

### Transcribe (speech + triage)

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@sample.wav" \
  -F "custom_keywords=convoy,medevac" \
  -F "deepfake_probability=0.15" \
  -F "duress_probability=0.0"
```

Full API reference: [docs/API.md](docs/API.md)

---

## Transcription & Triage (CLI)

Standalone transcription without the API:

```bash
pip install faster-whisper   # also requires ffmpeg
python transcriber.py audio.wav --device cpu --compute-type int8 --model-size tiny.en
```

Stream segment-level JSON for live dashboards:

```bash
python transcriber.py audio.wav --stream
```

Python usage:

```python
from transcriber import FastMilitaryTranscriber

transcriber = FastMilitaryTranscriber(model_size="tiny.en", device="cpu", compute_type="int8")
result = transcriber.transcribe("audio.wav", external_signals={"deepfake_probability": 0.12})
print(result.category, result.severity, result.transcript)
```

Categories: `administrative`, `command`, `intelligence`, `logistics`, `medical`, `emergency`, `authentication`, `unknown`.

Severity: `low` → `medium` → `high` → `critical` (rule-based keyword scoring).

Full triage reference: [docs/TRIAGE.md](docs/TRIAGE.md)

---

## Model Benchmarks

### ASVspoof 2019 LA (full splits)

Train on all 25,380 train clips; evaluate on validation and test:

```bash
python scripts/run_full_fast_pipeline.py
```

| Model | Split | EER | Latency (p95) |
|-------|-------|-----|---------------|
| Fast baseline (MFCC + RBF SVM) | dev | ~10% | ~6 ms |
| Fast baseline (MFCC + LR) | test | 21.43% | ~6.6 ms |
| **Spectra-AASIST3** (published) | test | **0.723%** | ~190 ms (MPS) |

### Demo dataset (Gary Stafford TTS, 1,866 clips)

```bash
python scripts/train_demo_rbf.py
python scripts/eval_demo_dataset.py
```

| Model | Demo accuracy | Latency p50 |
|-------|---------------|-------------|
| Spectra + threshold | 62.3% | ~181 ms |
| Spectra + argmax | 92.6% | ~185 ms |
| Fast RBF (ASVspoof-trained) | 47.7% | ~7 ms |
| **Fast RBF (demo-trained)** | **99.0%** | **~7 ms** |

**Hackathon recommendation:** use **fast demo RBF** for live uploads (speed + domain match). Cite Spectra **0.723% EER on ASVspoof** for accuracy narrative.

Artifacts:
- Demo model: `results/fast_baseline_mfcc_rbf_svc_demo.joblib`
- ASVspoof model: `results/fast_baseline_mfcc_rbf_svc.joblib`
- Metrics: `results/summary_demo_eval.json`, `results/comparison_fast_vs_spectra.json`

---

## Evaluation Scripts

### ASVspoof batch eval

Dataset: [Bisher/ASVspoof_2019_LA](https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA) (~7.5 GB).

```bash
python scripts/eval_asvspoof.py --split validation --max-samples 100   # smoke test
python scripts/eval_asvspoof.py --split validation --batch-size 16      # dev EER
python scripts/eval_asvspoof.py --split test --batch-size 16           # test EER
```

### Fast baseline train/eval

```bash
python scripts/eval_fast_baseline.py \
  --train-split train --eval-split validation \
  --max-train-samples 8000 --max-eval-samples 3000

python scripts/eval_fast_baseline.py --model-type random_forest
python scripts/eval_fast_baseline.py --model-type logistic_regression
```

### Demo dataset

```bash
python scripts/download_demo_dataset.py
python scripts/train_demo_rbf.py
python scripts/eval_demo_dataset.py
```

Dataset layout:

```
data/demo/deepfake-audio-detection/
  real/   # 933 authentic YouTube speech clips
  fake/   # 933 TTS clips (ElevenLabs, Polly, Speechify, etc.)
  manifest.json
```

### Out-of-domain test clips

```bash
python scripts/download_ood_testsets.py --skip-itw --per-class 50
python scripts/download_ood_testsets.py --per-class 50   # includes In-The-Wild (~8 GB)
python scripts/eval_ood_testsets.py
```

Clips: `data/ood/asvspoof19-la/{real,fake}/`, `data/ood/in-the-wild/{real,fake}/`

If OOD accuracy is poor, train a pooled model:

```bash
python scripts/train_pooled_rbf.py
```

---

## Explainability (Fast Model)

For `MODEL_BACKEND=fast`, `/classify` returns an `explanation` object comparing the clip's MFCC/LFCC features against bonafide and spoof class profiles from the demo train split.

Rebuild profiles:

```bash
python scripts/build_feature_profiles.py
```

Spectra responses return `explanation: null`.

---

## Go / No-Go Criteria

| Signal | Green | Red |
|--------|-------|-----|
| EER (ASVspoof19 LA test) | < ~1.5% | > 3–5% |
| API latency p95 (GPU) | < 500 ms | > 2 s |
| API latency p95 (CPU) | < 2 s | > 5 s |

Model card baseline: **0.723% EER** on ASVspoof19 LA.

---

## Project Structure

```
Hackathon/
├── api/main.py              # FastAPI service
├── transcriber.py           # Whisper + military triage
├── src/                     # ML core (inference, duress, fast baseline, XAI)
├── frontend/                # React dashboard (SignalShield AI)
├── scripts/                 # Training, eval, dataset download
├── results/                 # Models, metrics, feature profiles
├── data/demo/               # Gary Stafford demo dataset
├── data/ood/                # Out-of-domain test clips
├── vendor/spectra_aasist3/  # Vendored Spectra model
├── Agents.md                # AI pipeline / analyst module architecture
└── docs/
    ├── ARCHITECTURE.md      # System design
    ├── API.md               # HTTP API reference
    └── TRIAGE.md            # Transcription & triage rules
```

---

## Documentation Index

| Document | Contents |
|----------|----------|
| [Agents.md](Agents.md) | Analyst modules, signal fusion, configuration |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Components, preprocessing, deployment |
| [docs/API.md](docs/API.md) | `/health`, `/classify`, `/transcribe` reference |
| [docs/TRIAGE.md](docs/TRIAGE.md) | Categories, severity, keyword tuning |

---

## Dependencies

**Python** (`requirements.txt`): torch, torchaudio, transformers, huggingface_hub, fastapi, uvicorn, scikit-learn, librosa, faster-whisper, soundfile, datasets, numpy.

**Frontend** (`frontend/package.json`): React 18, Vite 5, Framer Motion, Lucide React.

**System:** Python 3.10+, Node 18+, ffmpeg (for transcription).

---

## License & Attribution

- Spectra-AASIST3: [lab260/Spectra-AASIST3](https://huggingface.co/lab260/Spectra-AASIST3)
- Demo dataset: [garystafford/deepfake-audio-detection](https://huggingface.co/datasets/garystafford/deepfake-audio-detection)
- ASVspoof eval: [Bisher/ASVspoof_2019_LA](https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA)
