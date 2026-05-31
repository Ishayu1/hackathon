# API Reference

Base URL (local dev): `http://127.0.0.1:8000`

When using the Vite dev server, the frontend calls `/api/*` which proxies to the backend.

Interactive docs: `http://127.0.0.1:8000/docs` (FastAPI auto-generated OpenAPI).

---

## GET /health

Returns service and model status.

**Response example:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu",
  "backend": "fast",
  "spectra_decision": null,
  "fast_model_path": "results/fast_baseline_mfcc_rbf_svc_demo.joblib",
  "xai_available": true,
  "xai_scope": "fast MFCC/LFCC class-profile comparison only",
  "transcriber_available": true,
  "transcriber_loaded": true,
  "transcriber_model_size": "tiny.en",
  "transcriber_vad_filter": false,
  "transcriber_error": null,
  "duress_enabled": true,
  "duress_model_path": "temporal_bilstm_duress.pth",
  "duress_threshold": 0.5,
  "duress_loaded": true,
  "duress_error": null
}
```

---

## POST /classify

Upload an audio file for deepfake detection, duress analysis, and (fast backend) explainability.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | yes | Audio file (`.wav`, `.flac`, `.mp3`, etc.) |

**Response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | `bonafide` or `spoof` |
| `is_spoof` | boolean | Whether classified as synthetic |
| `score_spoof` | float | Spoof logit or probability |
| `score_bonafide` | float | Bonafide logit or probability |
| `confidence` | float | Softmax confidence for predicted class |
| `threshold` | float | Decision threshold used |
| `preprocess_ms` | float | Decode + preprocess time |
| `feature_ms` | float | Feature extraction time (fast only) |
| `inference_ms` | float | Model forward pass time |
| `total_ms` | float | End-to-end latency |
| `filename` | string | Original upload filename |
| `backend` | string | `fast` or `spectra` |
| `explanation` | object \| null | Feature rationale (fast backend only) |
| `duress` | object | Acoustic duress result |

**Example:**

```bash
curl -X POST http://localhost:8000/classify \
  -F "file=@data/demo/deepfake-audio-detection/real/real_0000.flac"
```

**Explanation object (fast backend):**

```json
{
  "method": "class_profile_comparison",
  "prediction": "bonafide",
  "summary": "The classifier labeled this clip as bonafide because...",
  "top_signals": [
    {
      "name": "mfcc_3_mean",
      "label": "MFCC coefficient 3 mean",
      "value": -12.45,
      "direction": "toward_bonafide",
      "closeness_margin": 1.23
    }
  ],
  "disclaimer": "Signals reflect similarity to training distributions, not proof of synthesis method."
}
```

---

## POST /transcribe

Upload audio for speech-to-text and military message triage.

**Request:** `multipart/form-data`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | file | yes | — | Audio file |
| `custom_keywords` | string | no | `""` | Comma/semicolon/newline-separated watchlist terms |
| `deepfake_probability` | float | no | `0.0` | Spoof probability from `/classify` (0–1) |
| `duress_probability` | float | no | `0.0` | Duress probability from `/classify` (0–1) |

**Response (success):**

```json
{
  "available": true,
  "audio_path": "/tmp/abc123.wav",
  "transcript": "troops in contact need medevac",
  "category": "emergency",
  "severity": "critical",
  "severity_score": 90,
  "chunks": [
    {
      "start": 0.0,
      "end": 3.2,
      "text": "troops in contact need medevac",
      "category": "emergency",
      "severity": "critical",
      "severity_score": 90,
      "matched_terms": ["troops in contact", "medevac"]
    }
  ],
  "language": "en",
  "duration_seconds": 3.2,
  "processing_seconds": 1.1,
  "real_time_factor": 0.34,
  "processing_ms": 1100.0,
  "custom_keywords": "convoy, checkpoint"
}
```

**Response (failure):**

```json
{
  "available": false,
  "error": "Missing dependency: faster-whisper...",
  "custom_keywords": ""
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@sample.wav" \
  -F "custom_keywords=convoy,medevac" \
  -F "deepfake_probability=0.15" \
  -F "duress_probability=0.72"
```

---

## Error Responses

| Status | Condition |
|--------|-----------|
| `400` | Empty file, decode failure, inference error |
| `503` | Model not loaded yet (startup in progress) |

Error body: `{ "detail": "..." }`

---

## Environment Configuration

Set before starting uvicorn:

```bash
# Deepfake backend
export MODEL_BACKEND=fast                    # fast | spectra
export FAST_MODEL_PATH=results/fast_baseline_mfcc_rbf_svc_demo.joblib
export FAST_PROFILE_PATH=results/fast_demo_feature_profiles.json
export SPECTRA_DECISION=argmax               # argmax | threshold

# Duress
export DURESS_ENABLED=1
export DURESS_MODEL_PATH=temporal_bilstm_duress.pth
export DURESS_THRESHOLD=0.5

# Transcriber
export TRANSCRIBER_MODEL_SIZE=tiny.en
export TRANSCRIBER_DEVICE=cpu
export TRANSCRIBER_COMPUTE_TYPE=int8
export TRANSCRIBER_VAD_FILTER=0

# CORS (for non-proxied frontend)
export CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

**Start server:**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Use Spectra neural backend:**

```bash
MODEL_BACKEND=spectra SPECTRA_DECISION=argmax uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## Frontend Integration

The React dashboard (`frontend/src/api.js`) calls endpoints in this order:

1. `POST /classify` — synchronous; renders authenticity, duress, rationale immediately
2. `POST /transcribe` — async; updates transcript, category, severity, watchlist matches

Risk fusion happens client-side in `mapClassifyResponse` and `applyTranscriptionResult`.

**Remote API:**

```bash
VITE_API_BASE=http://your-host:8000 npm run dev
```
