# ISL Real-Time Translator

A real-time **HandTalk India : Indian Sign Language (ISL) recognition system** built with TensorFlow, MediaPipe, and OpenCV.  
It runs entirely on CPU and supports two detection modes — letter-by-letter spelling and full-sentence signing.

---

## Features

- **Word Mode** — detects individual ISL hand signs (A–Z) and assembles them into words and sentences
- **Sentence Mode** — records a sequence of body/hand landmarks and classifies complete signed phrases
- **Text-to-Speech** — speaks recognised words and matched phrases aloud via `pyttsx3`
- **ISL Phrase Dictionary** — 50+ common ISL phrases auto-matched as you type
- **Live hold-progress ring** — visual feedback shows when a letter is about to commit
- **Screenshot saving** — press `S` at any time to save the current frame

---

## Project Structure

```
ISL/
│
├── word/                          # Word-mode artefacts (auto-created)
│   ├── isl_best_model.keras       # Final trained letter model  [gitignored]
│   ├── isl_phase1_best.keras      # Phase 1 checkpoint          [gitignored]
│   ├── label_map.json             # Class index → letter map    [gitignored]
│   └── plots/                     # Training curve images       [gitignored]
│
├── sentence/                      # Sentence-mode artefacts
│   ├── sentence_model.keras       # Trained sentence model      [gitignored]
│   ├── sentence_label_map.json    # Class index → phrase map    [gitignored]
│   └── sentence_model.py          # Model architecture + constants
│
├── saved_predictions/             # Screenshots saved at runtime [gitignored]
│
├── train.ipynb                    # ← Train the letter (word) model
├── word_model.py                  # MobileNetV2 architecture + fine-tune helpers
├── utils.py                       # Data generators, smoothers, drawing utils
├── predict.py                     # Real-time webcam inference (both modes)
├── sentence_builder.py            # Letter → word → sentence engine + phrase dict
│
├── .gitignore
└── README.md
```

---

## Requirements

- Python 3.9 – 3.11
- Webcam

Install dependencies:

```bash
pip install tensorflow==2.15 mediapipe opencv-python pyttsx3 scikit-learn matplotlib numpy
```

> **GPU users:** replace `tensorflow` with `tensorflow-gpu` for faster training.  
> Mixed-precision (`float16`) is automatically beneficial only on GPU — the training script leaves it off for CPU.

---

## Quick Start

### 1 — Prepare your dataset

Organise images into one sub-folder per letter:

```
isl_word/
├── A/   (1200 images)
├── B/   (1200 images)
...
└── Z/   (1200 images)
```

### 2 — Train the letter model

Open `train.ipynb` and update the dataset path at the top:

```python
DATASET_PATH = r"C:\path\to\isl_word"
```

Run all cells. Training runs in two phases:

| Phase | What happens | Typical duration |
|---|---|---|
| Phase 1 | Classification head trained, MobileNetV2 frozen | ~5–10 min |
| Phase 2 | Top 30 MobileNetV2 layers fine-tuned | ~5–8 min |

Output files saved automatically to `word/`.

### 3 — Run live detection

```bash
python predict.py
```

---

## Controls

### Both modes
| Key | Action |
|---|---|
| `M` | Toggle Word ↔ Sentence mode |
| `S` | Save screenshot |
| `Q` | Quit |

### Word Mode
| Key | Action |
|---|---|
| `SPACE` | Confirm current word, start next word |
| `BACKSPACE` | Delete last letter (or restore last word) |
| `ENTER` | Speak full sentence / show matched phrase |
| `C` | Clear everything |
| `H` | Toggle ISL phrase cheat-sheet |

### Sentence Mode
| Key | Action |
|---|---|
| `R` | Start / cancel a signing recording |

---

## Training Configuration

Key hyper-parameters in `train.ipynb` (tuned for 1200 images/class):

| Parameter | Value | Notes |
|---|---|---|
| `BATCH_SIZE` | 64 | Increase to 128 if ≥8 GB RAM free |
| `PHASE1_EPOCHS` | 20 | Early stopping kicks in earlier with more data |
| `PHASE1_LR` | 1e-3 | Safe with batch size 64 |
| `FINE_TUNE_LR` | 5e-5 | Conservative enough to avoid catastrophic forgetting |
| `UNFREEZE_N` | 30 | Top 30 MobileNetV2 layers unfrozen in Phase 2 |
| `DATA_WORKERS` | 8 | Set to your physical CPU core count |

---

## Model Architecture

```
Input (224×224×3)
    └── MobileNetV2 (ImageNet pretrained, frozen in Phase 1)
            └── GlobalAveragePooling2D
                └── BatchNormalization
                    └── Dense(256, relu) → Dropout(0.4)
                        └── Dense(128, relu) → Dropout(0.3)
                            └── Dense(26, softmax)
```

Total params: ~2.6 M | Trainable in Phase 1: ~367 K

---

## Inference Pipeline (Word Mode)

1. MediaPipe Hands detects the **primary hand** (closest to frame centre)
2. Hand ROI is cropped with 30% padding
3. Skin segmentation replaces the background with the training-set green
4. ROI resized to 224×224, normalised to `[0, 1]`
5. Letter model predicts class; majority-vote smoother (3 frames) removes flicker
6. `SentenceBuilder` commits a letter only after it is held stably for **10 frames** (~0.33 s at 30 fps)

---

## Known Limitations

- Designed for **static ISL hand signs** (A–Z alphabet); dynamic/motion signs are handled by Sentence Mode only
- Background segmentation uses HSV + YCrCb skin detection — performance may vary under poor or inconsistent lighting
- Sentence Mode requires a separately trained LSTM/GRU model (`train_sentence.py`, not included here)

---

## Dataset Reference

Letter vocabulary sourced from:
https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl

Phrase vocabulary sourced from:  
https://www.kaggle.com/datasets/biswajit002/isl-video-sentences-dataset-for-recognition

---

## License

MIT— free to use, modify, and distribute.

---

## Author

Creator of **HandTalk India** : **heykayy** & **komal05-web**
