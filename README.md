# Voice Emotion Analyzer

A speech emotion recognition system that classifies short voice clips into one of eight emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised.

The model is a fine-tuned Wav2Vec2 (`superb/wav2vec2-base-superb-er`) backbone with a custom classifier head, trained on the RAVDESS dataset. The project includes a FastAPI backend for inference and a browser-based frontend for recording or uploading audio.

## How it works

1. Audio (recorded in-browser or uploaded) is sent to a FastAPI backend.
2. The backend loads the audio with librosa, resamples to 16kHz, normalizes, and pads/truncates to 4 seconds — matching the exact preprocessing used during training.
3. The processed waveform is passed through the Wav2Vec2 backbone. Mean-pooling and max-pooling of the hidden states are concatenated and passed through a 4-layer dense classifier.
4. The backend returns a probability distribution over the eight emotion classes, which the frontend renders as a ranked bar chart.

## Project structure

```
.
├── backend.py          FastAPI server, model definition, and /predict endpoint
├── index.html           Frontend: record/upload audio, display results
├── requirements.txt      Python dependencies
└── final_model_ravdess_sp_emotion.pth   Trained model weights (not included in repo, see below)
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Requires ffmpeg on the system PATH for decoding browser-recorded audio (webm/opus).

### 2. Add model weights

The trained weights file (`final_model_ravdess_sp_emotion.pth`, ~380MB) is not included in this repository due to GitHub's file size limits. Download it from [link to your Google Drive / release] and place it in the project root, alongside `backend.py`.

### 3. Run the backend

```bash
uvicorn backend:app --reload
```

The server starts at `http://127.0.0.1:8000`. Confirm it's running by visiting `http://127.0.0.1:8000/health`.

### 4. Run the frontend

Opening `index.html` directly (`file://`) works for file upload but blocks microphone access. To use in-browser recording, serve it over HTTP instead:

```bash
python -m http.server 5500
```

Then open `http://localhost:5500/index.html` in a browser. Confirm the backend URL field points to `http://localhost:8000/predict`.

## Training details

- Dataset: RAVDESS (Audio_Speech_Actors_01-24), 1,440 clips across 24 actors.
- Backbone: `superb/wav2vec2-base-superb-er`, frozen initially, then fine-tuned in two phases (frozen backbone first, then full unfreeze at a lower learning rate).
- Augmentation: time-stretch, pitch-shift, and additive noise applied during training.
- Class imbalance handled with `compute_class_weight` (RAVDESS has fewer neutral samples than other emotions).

## A note on the reported accuracy

The training notebook reports 76.85% test accuracy. This number comes from a **random** train/val/test split, which means audio from the same speaker (actor) can appear in both the training and test sets. This inflates accuracy, since the model can partly learn to recognize speaker-specific voice characteristics rather than emotion alone.

A separate actor-independent evaluation (GroupKFold cross-validation, where each fold holds out entirely unseen actors) on an earlier version of this pipeline gave a mean accuracy of 53.96% (plus or minus 5.97%). This is a more realistic estimate of how the model performs on a genuinely new speaker's voice. For context, RAVDESS-based speaker-independent results in published research typically fall in the 60-75% range even with more elaborate methods (multi-modal fusion, larger pretrained models, heavier augmentation).

If you plan to cite an accuracy figure for this project, the actor-independent estimate is the honest one to use.

## Known limitations

- Audio-only; RAVDESS also includes a video modality that is not used here and could improve accuracy if incorporated.
- Trained on acted emotion (professional actors reading fixed scripts), which does not necessarily generalize to spontaneous, real-world emotional speech.
- 4-second fixed window; longer or shorter clips are truncated or zero-padded, which may lose information for atypical clip lengths.
