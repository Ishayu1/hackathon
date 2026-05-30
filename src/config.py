from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_ID = "lab260/Spectra-AASIST3"
SAMPLE_RATE = 16000
CLIP_LEN = 64600

# README default; model.py classify() uses -1.0625009 internally
DEFAULT_THRESHOLD = -1.460938

DEFAULT_FAST_MODEL = RESULTS_DIR / "fast_baseline_mfcc_rbf_svc.joblib"
DEFAULT_DEMO_FAST_MODEL = RESULTS_DIR / "fast_baseline_mfcc_rbf_svc_demo.joblib"
DEFAULT_FAST_PROFILE_PATH = RESULTS_DIR / "fast_demo_feature_profiles.json"
DEMO_MANIFEST = PROJECT_ROOT / "data" / "demo" / "deepfake-audio-detection" / "manifest.json"
DEMO_FEATURE_CACHE = RESULTS_DIR / "feature_cache_demo_mfcc.npz"

# Spectra label rule: "argmax" compares logits directly; "threshold" uses ASVspoof-tuned bonafide logit cutoff
DEFAULT_SPECTRA_DECISION = "argmax"

PREPROCESS_MODES = ("deterministic", "random")
