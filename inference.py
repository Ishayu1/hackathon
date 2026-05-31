"""Backward-compatible entrypoint for acoustic duress inference."""

from src.duress_inference import Wav2Vec2BiLSTMClassifier, analyze_duress_probability

__all__ = ["Wav2Vec2BiLSTMClassifier", "analyze_duress_probability"]

if __name__ == "__main__":
    test_audio_file = "live_recording.wav"
    weights_file = "temporal_bilstm_duress.pth"

    try:
        duress_chance = analyze_duress_probability(test_audio_file, weights_file)
        print("--- Inference Results ---")
        print(f"File: {test_audio_file}")
        print(f"Probability of Duress: {duress_chance:.2f}%")
    except FileNotFoundError as exc:
        print(f"Execution Error: {exc}")
        print("Ensure 'temporal_bilstm_duress.pth' exists and the audio path is correct.")
