# Military Audio Transcription & Triage

The transcription and triage pipeline converts speech to text, classifies operational content, and scores message severity. It is implemented in `transcriber.py` and exposed via `POST /transcribe` and the CLI.

---

## Quick Start

### Install

```bash
pip install faster-whisper
# ffmpeg must be on PATH
```

### CLI

```bash
# Basic transcription + triage
python transcriber.py audio.wav --device cpu --compute-type int8 --model-size tiny.en

# Stream segment-level JSON events (for live dashboards)
python transcriber.py audio.wav --stream

# Custom operator watchlist
python transcriber.py audio.wav --custom-keywords "convoy,checkpoint,medevac"
```

### Python API

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
        "duress_probability": 0.33,
    },
    custom_keywords="raven, sector seven",
)

print(result.category, result.severity, result.severity_score)
print(result.transcript)
```

### Streaming

```python
for event in transcriber.stream("audio.wav"):
    if event["type"] == "segment":
        print(event["chunk"]["text"], event["chunk"]["severity"])
    elif event["type"] == "final":
        print(event["result"]["transcript"])
```

Or via CLI: `python transcriber.py audio.wav --stream`

Each line is a JSON object with `type: "segment"` or `type: "final"`.

---

## Categories

| Category | Trigger examples |
|----------|------------------|
| `emergency` | mayday, SOS, casualties, troops in contact, medevac, ambush, chemical alert |
| `command` | execute, advance, hold position, withdraw, engage, fire mission, secure |
| `intelligence` | enemy, hostile, recon, drone, UAV, IED, coordinates, grid |
| `logistics` | ammo, fuel, resupply, convoy, maintenance, supply |
| `medical` | wounded, injured, medic, triage, evacuation, aid station |
| `authentication` | authenticate, challenge, password, callsign, codeword |
| `administrative` | status report, sitrep, routine, training, roll call, schedule |
| `unknown` | No keyword matches |

Classification uses keyword hit counts per category. Action verbs (`execute`, `engage`, etc.) boost the `command` score. Highest score wins.

---

## Severity Levels

| Level | Score range | Examples |
|-------|-------------|----------|
| `critical` | ≥ 75 | mayday, troops in contact, KIA, medevac, IED, overrun, chemical |
| `high` | 45–74 | hostile, engage, fire mission, compromised, drone, distress requests |
| `medium` | 15–44 | movement, resupply, execute, secure, checkpoint, delay |
| `low` | < 15 | routine, training, briefing, schedule, sitrep |

### Scoring modifiers

Beyond keyword weights, the scorer adds points for:

| Signal | Points |
|--------|--------|
| Distress request ("need help", "please help") | +25 |
| Assistance offer ("I can help") | +20 |
| Immediacy ("now", "immediately", "ASAP") | +15 |
| Scale ("multiple", "mass", "heavy") | +10 |
| Weapon commands ("open fire", "cease fire") | +15 |
| Custom keyword match | +20 each (configurable) |
| External signal ≥ 0.85 | +20 |
| External signal ≥ 0.65 | +10 |

External signals: `deepfake_probability`, `duress_probability`, `lie_probability` (reserved).

---

## Benign Context Suppression

To reduce false positives, certain phrases are ignored when benign context is detected:

| Term | Suppressed when |
|------|-----------------|
| `please help` | Followed by carry/lift/move |
| `get back` | Followed by "to me/you/us" |
| `open fire` | Followed by hydrant/place/pit |
| `under cover` | Followed by "of darkness/night" |

Benign help contexts: box, homework, groceries, furniture, etc.

Audit common phrases:

```bash
python phrase_audit.py
```

---

## Tuning

Edit keyword dictionaries in `transcriber.py`:

- `CATEGORY_KEYWORDS` — maps category → trigger terms
- `SEVERITY_KEYWORDS` — maps severity level → trigger terms
- `BENIGN_TERM_PATTERNS` — regex guards for false positives
- `custom_keyword_weight` — default 20; pass via API or constructor

---

## Integration with SignalShield

In the full dashboard flow:

1. `/classify` returns deepfake and duress scores
2. Frontend passes these as `deepfake_probability` and `duress_probability` to `/transcribe`
3. Triage severity incorporates acoustic risk signals
4. Frontend fuses transcript severity with authenticity/duress into overall risk band

See [Agents.md](../Agents.md) for the fusion orchestrator logic.
