# 🤟 ISL Real-Time Translator

A real-time **Indian Sign Language (ISL) translator** that uses computer vision and deep learning to recognize hand signs via webcam — supporting both **letter-by-letter word building** and **full-sentence gesture recognition**.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Keyboard Controls](#keyboard-controls)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Authors](#authors)
- [License](#license)

---

## Overview

The ISL Translator processes live webcam frames, detects hand landmarks using **MediaPipe**, crops the hand region, and classifies it using a fine-tuned **MobileNetV2** model. It supports two operating modes:

- **Word Mode** — Spell out words letter by letter using ISL alphabet signs. Letters are committed after a stable hold, then assembled into a sentence that can be spoken aloud via text-to-speech.
- **Sentence Mode** — Record a continuous gesture sequence (pose + both hands) and predict a complete ISL phrase using an LSTM-based sentence model.

---

## Features

- 🎥 **Real-time webcam inference** with threaded capture for low-latency processing
- 🖐️ **MediaPipe hand & holistic landmark detection**
- 🔤 **Word Mode** — letter-by-letter sign spelling with sentence builder
- 💬 **Sentence Mode** — full-phrase recognition from gesture sequences
- 🔊 **Text-to-speech** output via `pyttsx3`
- 📊 **FPS counter** and confidence display overlay
- 💾 **Screenshot saving** of predictions
- 📋 **Phrase cheat-sheet** overlay (toggleable)
- 🧠 **Prediction smoother** — sliding-window majority vote to eliminate flicker
- 🔁 **Fine-tuning support** — unfreeze top MobileNetV2 layers for improved accuracy

---

## Project Structure

```
ISL/
│
├── predict.py              # Main entry point — real-time dual-mode translator
├── word_model.py           # MobileNetV2 model architecture & fine-tuning helpers
├── utils.py                # Shared utilities: data loading, preprocessing, plotting
├── runtime.txt             # Python version specification (python-3.10.14)
├── LICENSE.txt             # MIT License
│
├── word/
│   ├── isl_best_model.keras    # Trained letter/word classification model
│   ├── label_map.json          # Class index → label mapping
│   ├── sentence_builder.py     # Letter-hold logic, word/sentence assembly
│   └── plots/                  # Training history plots (auto-generated)
│
└── sentence/
    ├── sentence_model.py        # LSTM sentence model definition & constants
    ├── sentence_model.keras     # Trained sentence classification model
    └── sentence_labels.json     # Sentence class label map
```

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Language | Python 3.10.14 |
| Deep Learning | TensorFlow / Keras |
| Base Model | MobileNetV2 (ImageNet weights) |
| Hand Detection | MediaPipe |
| Computer Vision | OpenCV |
| Text-to-Speech | pyttsx3 |
| Numerics | NumPy |
| Visualization | Matplotlib |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/heykayy/isl-translator.git
cd isl-translator
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install tensorflow==2.16 opencv-python mediapipe pyttsx3 numpy matplotlib scikit-learn
```

> **Note:** Python 3.10.14 is required (see `runtime.txt`).

### 4. Add trained models

Place the following files in their respective directories (trained separately):

- `word/isl_best_model.keras`
- `word/label_map.json`
- `sentence/sentence_model.keras`
- `sentence/sentence_labels.json`

---

## Usage

### Run the real-time translator

```bash
python predict.py
```

A webcam window titled **"ISL Real-Time Translator"** will open. By default it starts in **Word Mode**.

### Train the word/letter model

```bash
python word_model.py
```

Training history plots are saved to `word/plots/`.

---

## Keyboard Controls

### Global

| Key | Action |
|-----|--------|
| `M` | Toggle between Word / Sentence mode |
| `Q` | Quit the application |
| `S` | Save a screenshot of the current frame |

### Word Mode

| Key | Action |
|-----|--------|
| `SPACE` | Confirm current word → push to sentence |
| `BACKSPACE` | Delete last letter (or restore last word) |
| `ENTER` | Speak the full sentence aloud |
| `C` | Clear the sentence buffer |
| `H` | Toggle the phrase cheat-sheet overlay |

### Sentence Mode

| Key | Action |
|-----|--------|
| `R` | Start / cancel gesture recording |

> After recording `SEQUENCE_LEN` frames, the model auto-predicts and speaks the phrase.

---

## Model Architecture

### Word / Letter Model (`word_model.py`)

Built on **MobileNetV2** (pretrained on ImageNet) with a custom classification head:

```
MobileNetV2 (frozen base)
    → GlobalAveragePooling2D
    → BatchNormalization
    → Dense(256, relu) → Dropout(0.4)
    → Dense(128, relu) → Dropout(0.3)
    → Dense(num_classes, softmax)
```

**Two-phase training:**
1. **Phase 1** — Base frozen; only the classification head is trained (`lr = 1e-3`)
2. **Phase 2** — Top 30 base layers unfrozen for fine-tuning (`lr = 1e-5`)

### Sentence Model (`sentence/sentence_model.py`)

LSTM-based sequence classifier trained on MediaPipe holistic landmark sequences (pose + both hands). Input shape: `(SEQUENCE_LEN, FEATURE_DIM)`.

---

## Configuration

Key constants in `predict.py` and `utils.py`:

| Parameter | Default | Description |
|---|---|---|
| `IMG_SIZE` | `224` | Input image size for the model |
| `BATCH_SIZE` | `16` | Training batch size |
| `CONFIDENCE_THRESHOLD` | `0.55` | Minimum confidence to accept a word prediction |
| `SENTENCE_CONFIDENCE_THRESHOLD` | `0.55` | Minimum confidence for sentence prediction |
| `HOLD_FRAMES` | `10` | Frames a sign must be held to commit a letter (~0.33s) |
| `COOLDOWN_FRAMES` | `12` | Frames to ignore after a commit (~0.40s) |
| `BAD_FRAME_TOLERANCE` | `3` | Noisy frames tolerated before resetting hold counter |
| `SMOOTHING_WINDOW` | `3` | Sliding window size for prediction smoother |
| `PADDING_FRACTION` | `0.30` | Padding around the hand crop bounding box |
| `CAMERA_INDEX` | `0` | Webcam index |

---

## Authors

- **komal05-web** — [GitHub](https://github.com/komal05-web)

---

## License

This project is licensed under the **MIT License** — see [LICENSE.txt](LICENSE.txt) for details.

Copyright © 2026 heykayy & komal05-web
