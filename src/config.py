from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_ID = "lab260/Spectra-AASIST3"
SAMPLE_RATE = 16000
CLIP_LEN = 64600

# README default; model.py classify() uses -1.0625009 internally
DEFAULT_THRESHOLD = -1.460938

PREPROCESS_MODES = ("deterministic", "random")
