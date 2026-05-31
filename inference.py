import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Model

# Ensure environment parity with the training script
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# =====================================================================
# 1. Rebuild Architecture 
# (In a formal ML pipeline, this class would be imported from a shared utils.py)
# =====================================================================
class Wav2Vec2BiLSTMClassifier(nn.Module):
    def __init__(self, wav2vec2_model_name="facebook/wav2vec2-base", hidden_dim=256, num_classes=2):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(wav2vec2_model_name)
        self.lstm = nn.LSTM(
            input_size=self.wav2vec2.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, waveforms):
        with torch.no_grad():
            embeddings = self.wav2vec2(waveforms).last_hidden_state 
        lstm_out, _ = self.lstm(embeddings)
        pooled_out = torch.mean(lstm_out, dim=1)
        return self.fc(pooled_out)

# =====================================================================
# 2. Inference & Probability Extraction
# =====================================================================
def analyze_duress_probability(audio_path, model_weights_path="duress_model_weights.pth", target_sample_rate=16000):
    """
    Loads a saved model, processes a raw audio file, and outputs the 
    percentage probability that the audio belongs to the 'Duress' class.
    Assumes Class 0 = Normal, Class 1 = Duress.
    """
    # 1. Initialize empty model architecture and map it to the active device
    model = Wav2Vec2BiLSTMClassifier(num_classes=2).to(device)
    
    # 2. Load the trained state dictionary into the model
    # map_location ensures safe loading if moving weights between GPU/MPS/CPU
    model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
    model.eval() # Freeze dropout layers and batch normalization

    # 3. Audio Preprocessing Pipeline
    waveform, sample_rate = torchaudio.load(audio_path)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        
    # Shape preparation: (sequence_length) -> (1, sequence_length) for batch processing
    waveform = waveform.squeeze(0).unsqueeze(0).to(device)

    # 4. Forward Pass & Probability Calculation
    with torch.no_grad():
        logits = model(waveform)
        
        # Convert raw network logits to a normalized probability distribution [0, 1]
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        
        # Extract Class 1 (Duress) probability and convert to percentage
        duress_prob = probabilities[1].item()
        percentage = duress_prob * 100

    return percentage

if __name__ == "__main__":
    test_audio_file = "live_recording.wav"
    weights_file = "duress_model_weights.pth"
    
    try:
        duress_chance = analyze_duress_probability(test_audio_file, weights_file)
        print(f"--- Inference Results ---")
        print(f"File: {test_audio_file}")
        print(f"Probability of Duress: {duress_chance:.2f}%")
        
    except FileNotFoundError as e:
        print(f"Execution Error: {e}")
        print("Ensure 'duress_model_weights.pth' exists and the audio path is correct.")