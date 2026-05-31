# Fast Military Audio Transcriber

This workspace now has a small, composable transcription and triage module for the hackathon pipeline. It handles:

- audio-to-text with `faster-whisper`
- per-segment military message category classification
- severity scoring
- streaming JSON events so a UI can show category/severity before the final transcript is complete
- future deepfake and lie-detection scores through `external_signals`

## Install

```bash
pip install faster-whisper
```

You also need `ffmpeg` on your PATH.

## Fast CLI Usage

```bash
python transcriber.py audio.wav --device cpu --compute-type int8 --model-size tiny.en
```

For lower latency UI integration, stream segment-level events:

```bash
python transcriber.py audio.wav --stream
```

Each segment event contains the text decoded so far, category, severity, numeric severity score, timestamps, and matched trigger terms. A final event contains the full transcript and overall classification.

## Python Usage

```python
from transcriber import FastMilitaryTranscriber

transcriber = FastMilitaryTranscriber(
    model_size="tiny.en",
    device="cpu",
    compute_type="int8",
)

result = transcriber.transcribe(
    "audio.wav",
    external_signals={
        "deepfake_probability": 0.12,
        "lie_probability": 0.33,
    },
)

print(result.category, result.severity, result.severity_score)
print(result.transcript)
```

## Categories

- `administrative`
- `command`
- `intelligence`
- `logistics`
- `medical`
- `emergency`
- `authentication`
- `unknown`

## Severity

Severity is rule-based for speed and explainability:

- `critical`: casualties, troops in contact, mayday, medevac, IED, overrun, etc.
- `high`: hostile/enemy contact, engage, fire mission, compromised, drone/UAV, etc.
- `medium`: movement, resupply, execute, secure, checkpoint, delay, etc.
- `low`: routine status, training, schedule, roll call, etc.

The scoring logic is intentionally easy to tune in `transcriber.py` by editing `CATEGORY_KEYWORDS` and `SEVERITY_KEYWORDS`.

## Integration Notes

For a live dashboard, call `transcribe_then_emit(...)` or use `python transcriber.py audio.wav --stream` and consume each JSON line as it arrives.

For repeated files or a server process, prefer `FastMilitaryTranscriber` so the Whisper model stays warm in memory after the first load.
