# Sign Language Translator (ASL)

Real-time ASL-to-text-and-speech translator using MediaPipe landmarks and sequence models (LSTM/TCN/Transformer).
Scope: isolated sign recognition (250 signs), not continuous ASL translation. Continuous phrases are a stretch goal.

## Rules
- Never invent training results or metrics; only report numbers from actual runs.
- Explain each change before making it. Minimal, focused changes; no unrequested additions or new dependencies.
- Do not start long training runs without asking first.
- Do not modify or regenerate data/landmarks/ (extraction takes ~71 min).
- No emojis in README files.
- Environment: Windows, Git Bash, VS Code. Keep terminal commands short (heredocs break in Git Bash).

## Environment
- Project: D:\projects\sign-language-translator (Git Bash: /d/projects/sign-language-translator)
- Activate venv: source .venv/Scripts/activate
- Python, PyTorch 2.14.1 (CPU only), MediaPipe 1.1.1, OpenCV 5.0.0, pandas, scikit-learn, pyarrow, tqdm, streamlit, wandb, pyttsx3

## Data
- Google ASL Signs (Kaggle competition asl-signs): 94,477 videos, 250 signs, 21 signers. Raw files in data/raw/.
- src/extract.py output in data/landmarks/:
  - X.npy (94477, 64, 75, 3) float16
  - y.npy (94477,) int16, sign index
  - participants.npy (94477,) int32, signer id
- Point order: left_hand 0-20, pose 21-53, right_hand 54-74. Face excluded.
- Normalization: x,y centered on shoulder midpoint and divided by shoulder width (per video); z divided by shoulder width only.
- Each video resampled to 64 frames by evenly spaced indices. Missing landmarks = 0.
- Sanity check (src/check_data.py, 2000-video sample): no NaN/Inf; missing fraction left_hand 0.722, pose 0.000, right_hand 0.624; 299-415 videos per class; 3338-4968 per signer.
- Known: lower-body pose points reach y ~ +3.4 (likely out of frame, hallucinated); z is noisy. Candidates for later experiments.

## Code (all tested and committed)
- src/check_data.py: data sanity check
- src/dataset.py: signer-independent split (3 held-out signers, seed 42 -> 4718, 36257, 37779), augmentation (mirror with hand/pose swap, rotation +-15 deg, scale 0.8-1.2, shift +-0.1, temporal crop 80-100%; missing points kept at 0). Returns (64, 225) float32 tensors.
- src/models/lstm.py: BiLSTMClassifier (input proj 225->256, 2-layer BiLSTM, mean+max pooling, 2.95M params, 6.6 ms/sample CPU)
- src/train.py: AdamW, OneCycleLR, label smoothing 0.1, grad clip 1.0, saves best val top-1 to checkpoints/bilstm.pt. num_workers=0 default (Windows RAM).
- Run: python -m src.train --subset 4000 --epochs 2 (smoke test passed, ~28 s/epoch on 3.4k samples)

## Current step
- Learning check in progress: python -m src.train --subset 20000 --epochs 10
- Estimated full run on CPU: ~11 min/epoch, ~7.5 h for 40 epochs. Considering a Kaggle GPU instead.

## Next steps
1. Full BiLSTM training; record signer-independent val top-1/top-5.
2. Add held-out test signers for final reporting (val is used for checkpoint selection).
3. TCN and Transformer models; results table.
4. src/evaluate.py: top-1/top-5, per-class accuracy, confusion matrix, confusable signs.
5. Experiments: presence mask, drop z, drop lower-body pose, face landmarks.
6. src/realtime.py: webcam, MediaPipe Holistic, sliding window, smoothing, confidence threshold.
7. Sentence builder + text-to-speech.
8. Own recordings (10-20 words, multiple people) to test generalization.
9. Streamlit demo, deploy (Hugging Face Spaces or Docker), demo video, README.