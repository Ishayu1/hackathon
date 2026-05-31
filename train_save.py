import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import soundfile as sf
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# ==========================================
# 1. ARCHITECTURE DEFINITION (BiLSTM)
#    Matches CNN_Classification.py exactly:
#    - input_size=768, hidden_size=128, num_layers=2, dropout=0.4, bidirectional
#    - Global max pooling over time dimension
#    - Linear(256, 1) binary head
#    - Trained with BCEWithLogitsLoss
# ==========================================
class TemporalBiLSTM(nn.Module):
    def __init__(self):
        super(TemporalBiLSTM, self).__init__()
        # Input shape from Wav2Vec2: (Batch, Sequence=149, Features=768)
        # 128 hidden units * 2 directions = 256 output features per time step
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.4,
            bidirectional=True
        )
        # Classification head mapping the 256 features to a binary logit
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        # x shape: (Batch, 149, 768)
        lstm_out, _ = self.lstm(x)  # (Batch, 149, 256)

        # Global max pooling — isolates strongest temporal signals,
        # naturally ignores zero-padding from short clips
        features, _ = torch.max(lstm_out, dim=1)  # (Batch, 256)

        logits = self.classifier(features)  # (Batch, 1)
        return logits


# ==========================================
# 2. FEATURE EXTRACTION (WITH STATIC PADDING)
#    Matches CNN_Classification.py exactly:
#    - wav2vec2-base-960h with Wav2Vec2Processor
#    - 16kHz, truncate/pad to exactly 3 seconds (48000 samples)
#    - Embeddings cached to disk to avoid re-extraction
#    - RAVDESS + CREMA-D emotion label mapping preserved
# ==========================================
def extract_and_cache_dataset(ravdess_dir, cremad_dir, cache_file):
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}...")
        return torch.load(cache_file, weights_only=False)

    print("Cache not found. Beginning deep extraction (this will take 10-15 minutes)...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model_name = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    wav2vec2 = Wav2Vec2Model.from_pretrained(model_name).to(device)
    wav2vec2.eval()

    TARGET_SR = 16000
    TARGET_SAMPLES = TARGET_SR * 3  # Exactly 3 seconds = 48000 samples

    embeddings_list = []
    labels_list = []

    # -- RAVDESS Logic --
    # Filename format: 03-01-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav
    # Emotions: 1=neutral,2=calm,3=happy,4=sad,5=angry,6=fearful,7=disgust,8=surprised
    # Skip disgust(7) and surprised(8); label 1 (duress) for sad/angry/fearful (4,5,6)
    ravdess_files = list(Path(ravdess_dir).rglob("*.wav"))
    print(f"Found {len(ravdess_files)} RAVDESS files.")
    for f in ravdess_files:
        if "__MACOSX" in f.parts or f.name.startswith("._"):
            continue
        parts = f.stem.split('-')
        if len(parts) != 7:
            continue
        emotion = int(parts[2])
        if emotion in [7, 8]:
            continue
        target = 1 if emotion in [4, 5, 6] else 0
        _process_audio(f, target, wav2vec2, processor, device,
                       TARGET_SR, TARGET_SAMPLES, embeddings_list, labels_list)

    # -- CREMA-D Logic --
    # Filename format: {actorID}_{sentence}_{emotion}_{intensity}.wav
    # ANG/FEA/SAD -> 1 (duress); NEU/HAP -> 0 (normal); others skipped
    cremad_files = list(Path(cremad_dir).rglob("*.wav"))
    print(f"Found {len(cremad_files)} CREMA-D files.")
    emotion_map = {'ANG': 1, 'FEA': 1, 'SAD': 1, 'NEU': 0, 'HAP': 0}
    for f in cremad_files:
        if "__MACOSX" in f.parts or f.name.startswith("._"):
            continue
        parts = f.stem.split('_')
        if len(parts) != 4:
            continue
        emo_code = parts[2]
        if emo_code not in emotion_map:
            continue
        target = emotion_map[emo_code]
        _process_audio(f, target, wav2vec2, processor, device,
                       TARGET_SR, TARGET_SAMPLES, embeddings_list, labels_list)

    print("Compiling and saving tensors...")
    dataset_dict = {
        'X': torch.cat(embeddings_list, dim=0),       # (N, 149, 768)
        'y': torch.tensor(labels_list, dtype=torch.float32),  # (N,)
    }
    torch.save(dataset_dict, cache_file)
    print(f"Extraction complete. Saved to {cache_file}.")
    return dataset_dict


def _process_audio(file_path, target, wav2vec2, processor, device,
                   sr, max_len, X_list, y_list):
    """Load, preprocess, encode with Wav2Vec2, and append to lists."""
    audio_data, file_sr = sf.read(file_path)
    waveform = torch.tensor(audio_data, dtype=torch.float32)

    # Mono
    if waveform.ndim > 1:
        waveform = torch.mean(waveform, dim=1)

    # Resample if needed
    if file_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_freq=file_sr, new_freq=sr)
        waveform = resampler(waveform)

    # Static truncation / zero-padding to exactly 3 seconds
    if waveform.shape[0] > max_len:
        waveform = waveform[:max_len]
    else:
        pad_amount = max_len - waveform.shape[0]
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

    with torch.no_grad():
        inputs = processor(waveform, return_tensors="pt",
                           sampling_rate=sr).input_values.to(device)
        hidden_states = wav2vec2(inputs).last_hidden_state.cpu()  # (1, 149, 768)

    X_list.append(hidden_states)
    y_list.append(target)


# ==========================================
# 3. FULL DATASET TRAINING AND SERIALIZATION
#    Matches CNN_Classification.py exactly:
#    - AdamW lr=3e-4, weight_decay=0.01
#    - ReduceLROnPlateau(mode='min', factor=0.5, patience=2)
#    - BCEWithLogitsLoss
#    - batch_size=32, epochs=15
#    - Saves only LSTM + classifier state_dict (not Wav2Vec2)
# ==========================================
def train_and_save_full_model(dataset_dict, save_path="temporal_bilstm_duress.pth", epochs=15):
    X = dataset_dict['X']                      # (N, 149, 768)
    y = dataset_dict['y'].unsqueeze(1)         # (N, 1)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    print(f"\nInitiating full dataset training on {device}...")
    print(f"Total samples: {len(X)} | Feature shape: {X.shape[1:]}")
    print("=" * 70)

    full_ds = TensorDataset(X, y)
    train_loader = DataLoader(full_ds, batch_size=32, shuffle=True)

    model = TemporalBiLSTM().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1:02d}/{epochs} | Average Loss: {avg_loss:.4f}")

    print("=" * 70)
    # Save only the LSTM + classifier weights — Wav2Vec2 is always loaded
    # fresh from HuggingFace at inference time, so it doesn't need to be stored.
    torch.save(model.state_dict(), save_path)
    print(f"Model state dictionary saved to: {save_path}")


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    RAVDESS_DIR = "/Users/adiarora/Downloads/RAVDESS"
    CREMAD_DIR  = "/Users/adiarora/Downloads/CREMA-D"
    CACHE_FILE  = "/Users/adiarora/Downloads/temporal_embeddings_cache.pt"
    MODEL_SAVE_PATH = MODEL_SAVE_PATH = "/Users/adiarora/Downloads/temporal_bilstm_duress.pth"

    # 1. Load cached embeddings or extract from scratch
    dataset = extract_and_cache_dataset(RAVDESS_DIR, CREMAD_DIR, CACHE_FILE)

    # 2. Train on full dataset and save
    train_and_save_full_model(dataset, save_path=MODEL_SAVE_PATH, epochs=15)