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
- src/dataset.py: signer-independent split via split_indices() (3 held-out signers, seed 42 -> 4718, 36257, 37779), augmentation (mirror with hand/pose swap, rotation +-15 deg, scale 0.8-1.2, shift +-0.1, temporal crop 80-100%; missing points kept at 0). Returns (64, 225) float32 tensors.
- src/models/lstm.py: BiLSTMClassifier (input proj 225->256, 2-layer BiLSTM, mean+max pooling, 2.95M params, 6.6 ms/sample CPU)
- src/models/tcn.py: TCNClassifier (same input proj and mean+max pooling head as BiLSTM; 4 residual blocks of two non-causal Conv1d, kernel 5, dilation 1/2/4/8, BatchNorm, GELU, dropout 0.3; receptive field 121 frames, covers the 64-frame clip; 2.81M params, 4.7 ms/sample CPU). Smoke test (--model tcn --subset 4000 --epochs 2): ~42 s/epoch on 3.4k clips.
- src/train.py: AdamW, OneCycleLR, label smoothing 0.1, grad clip 1.0. Saves best val top-1 to checkpoints/<model>.pt and full training state every epoch to checkpoints/<model>_last.pt. --resume continues from _last.pt (refuses if epochs, batch size, lr, subset, seed or model differ). num_workers=0 default (Windows RAM).
- src/evaluate.py: top-1/top-5, per-class accuracy, confusion matrix, most confused pairs; rebuilds the val split with split_indices(). Loads <model>.pt and <model>_last.pt. Reviewed. Matches train.py metrics on the 20k-subset checkpoint (top-1 0.366, top-5 0.640). Epoch-30 BiLSTM (bilstm_last.pt, --full-val): top-1 0.573, top-5 0.798, per-class mean 0.569.
- Smoke test: python -m src.train --subset 4000 --epochs 2
- Full run (start and resume use the same command, run in your own terminal):
  python -u -m src.train --epochs 30 --resume 2>&1 | tee -a train_bilstm.log
  Logs (*.log) are not committed.

## Results (signer-independent: val signers 4718, 36257, 37779; 13,177 clips)
| Model | Epochs | Top-1 (final) | Top-5 (final) | Best top-1 (epoch) | Train acc (final) | CPU time/epoch | Params |
|---|---|---|---|---|---|---|---|
| BiLSTM | 30 | 0.573 | 0.798 | 0.574 (28) | 0.894 | ~700 s | 2.95M |
| TCN | 30 | 0.605 | 0.817 | 0.605 (30) | 0.820 | ~950 s | 2.81M |

- Protocol: fixed recipe (30 epochs, seed 42, same split) for every model; report final-epoch metrics, so no model selection uses the val signers. Do not tune hyperparameters on val; tuning requires separate test signers.
- BiLSTM observations: val top-1 plateaued from ~epoch 19; val loss bottomed ~epoch 20 (~1.97) while train loss kept falling -> overfitting to training signers (train 0.894 vs val 0.573). Top-5 is 22 points above top-1, suggesting confusion between similar signs.
- TCN observations: train-val gap 21.5 pts vs BiLSTM 32.1; TCN val loss still decreasing at epoch 30. Single seed and split, so the TCN vs BiLSTM difference is not yet shown to be robust.
- Earlier learning check (20k subset, 10 epochs): val top-1 0.366, top-5 0.640.

## Current step
Transformer model (src/models/transformer.py): same interface and recipe as BiLSTM/TCN.

## Next steps
1. TCN and Transformer models with the same recipe; add to the results table.
2. Experiments (Milestone 4): presence mask, drop z, drop lower-body pose, face landmarks, stronger regularization against signer overfitting. Add held-out test signers before any hyperparameter tuning.
3. src/realtime.py: webcam, MediaPipe Holistic, sliding window, smoothing, confidence threshold.
4. Sentence builder + text-to-speech.
5. Own recordings (10-20 words, multiple people) to test generalization.
6. Streamlit demo, deploy (Hugging Face Spaces or Docker), demo video, README.
7. Optional: fingerspelling (separate dataset) or drop from scope.