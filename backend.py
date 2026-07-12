"""
FastAPI backend for the fine-tuned Wav2Vec2 SER model.
Architecture matches Speech_Emotion_Recognition.ipynb EXACTLY.

Run: uvicorn backend:app --host 0.0.0.0 --port 8000

Expects final_model_ravdess_sp_emotion.pth in the same directory.

NOTE: this model was trained/evaluated on a RANDOM (not actor-independent) split.
The 76.85% test accuracy reported during training includes actor leakage between
train and test sets. On genuinely new speakers, expect performance closer to the
~54% actor-independent estimate from the earlier GroupKFold run, not 76.85%.
"""

import io
import torch
import torch.nn as nn
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import Wav2Vec2Model

BACKBONE_NAME = "superb/wav2vec2-base-superb-er"
WEIGHTS_PATH = "final_model_ravdess_sp_emotion.pth"

EMOTION_MAP = {
    1: "neutral", 2: "calm", 3: "happy", 4: "sad",
    5: "angry", 6: "fearful", 7: "disgust", 8: "surprised",
}
LABELS = list(EMOTION_MAP.values())
NUM_CLASSES = len(LABELS)

TARGET_SR = 16000
MAX_SECONDS = 4.0
MAX_LEN = int(TARGET_SR * MAX_SECONDS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- Model definition — must match training exactly ----------------
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, freeze_backbone=True):
        super(EmotionClassifier, self).__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(BACKBONE_NAME)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

    def forward(self, audio):
        with torch.no_grad():
            outputs = self.backbone(audio)
            hidden_states = outputs.last_hidden_state

        mean_pool = torch.mean(hidden_states, dim=1)
        max_pool = torch.max(hidden_states, dim=1)[0]
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        logits = self.classifier(pooled)
        return logits


app = FastAPI(title="Voice Emotion Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
def load_model():
    global model
    model = EmotionClassifier(num_classes=NUM_CLASSES, freeze_backbone=True).to(device)
    try:
        state_dict = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        raise RuntimeError(
            f"Could not find {WEIGHTS_PATH}. Place your fine-tuned model weights "
            f"in the same directory as backend.py."
        )
    model.eval()
    print(f"Model loaded on {device}.")


def preprocess_audio(audio_bytes: bytes) -> np.ndarray:
    import tempfile, os as _os
    # Write to a temp file so librosa/audioread/ffmpeg can sniff the real format
    # instead of relying on an in-memory buffer, which is unreliable for webm/opus.
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path, sr=TARGET_SR, mono=True)
    finally:
        _os.remove(tmp_path)
    y = librosa.util.normalize(y)
    if len(y) > MAX_LEN:
        y = y[:MAX_LEN]
    else:
        y = np.pad(y, (0, MAX_LEN - len(y)), mode='constant')
    return y.astype(np.float32)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    audio_bytes = await file.read()

    try:
        audio = preprocess_audio(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}")

    input_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)  # (1, MAX_LEN)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return {
        "probabilities": probs.tolist(),
        "labels": LABELS,
        "predicted": LABELS[int(np.argmax(probs))]
    }
