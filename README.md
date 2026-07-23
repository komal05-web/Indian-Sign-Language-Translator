# 🤟 ISL Real-Time Translator — HandTalk India

A real-time **Indian Sign Language (ISL) translator** that uses computer vision and deep learning to recognize hand signs via webcam — supporting both **letter-by-letter word building** and **full-sentence gesture recognition**. Runs entirely on CPU.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation (standalone script)](#installation-standalone-script)
- [Web App (isl_web/)](#web-app-isl_web)
- [Usage](#usage)
- [Keyboard Controls](#keyboard-controls)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Dataset Reference](#dataset-reference)
- [Known Limitations](#known-limitations)
- [Authors](#authors)
- [License](#license)

---

## Overview

The ISL Translator processes live webcam frames, detects hand landmarks using **MediaPipe**, crops the hand region, and classifies it using a fine-tuned **MobileNetV2** model. It supports two operating modes:

- **Word Mode** — Spell out words letter by letter using ISL alphabet signs. Letters commit after a stable hold, then assemble into a sentence that can be spoken aloud via text-to-speech.
- **Sentence Mode** — Record a continuous gesture sequence (pose + both hands) and predict a complete ISL phrase using an LSTM-based sentence model.

There are two ways to run it: the original **standalone script** (`predict.py`, opens an OpenCV window), or the **web app** (`isl_web/`, a Django + Channels app with a browser UI, live camera feed, and the same trained models running behind a WebSocket).

---

## Features

- 🎥 Real-time webcam inference with threaded capture for low-latency processing
- 🖐️ MediaPipe hand & holistic landmark detection
- 🔤 **Word Mode** — letter-by-letter sign spelling with sentence builder
- 💬 **Sentence Mode** — full-phrase recognition from gesture sequences
- 🔊 Text-to-speech output via `pyttsx3` (script) / Web Speech API (web app)
- 📊 FPS counter and confidence display overlay
- 💾 Screenshot saving of predictions (`S` key)
- 📋 Phrase cheat-sheet overlay, toggleable (`H` key)
- 🧠 Prediction smoother — sliding-window majority vote to eliminate flicker
- 🌐 Full browser-based UI (`isl_web/`) with the same keyboard shortcuts, an ISL alphabet reference chart, and a demo video section

---

## Project Structure

```
ISL/
├── predict.py                  # Standalone real-time translator (OpenCV window)
├── word_model.py                # MobileNetV2 architecture + fine-tuning helpers
├── utils.py                     # Data generators, smoothers, drawing utils
├── runtime.txt                  # Python version for the standalone script
├── LICENSE.txt
│
├── word/
│   ├── isl_best_model.keras     # Trained letter/word model      [gitignored]
│   ├── label_map.json           # Class index → letter map       [gitignored]
│   ├── sentence_builder.py      # Letter → word → sentence engine + phrase dict
│   ├── train.ipynb              # Train the letter model
│   └── plots/                   # Training curve images          [gitignored]
│
├── sentence/
│   ├── sentence_model.keras     # Trained sentence model         [gitignored]
│   ├── sentence_label_map.json  # Class index → phrase map       [gitignored]
│   ├── sentence_model.py        # LSTM architecture + constants
│   └── train_sentence.ipynb     # Train the sentence model
│
└── isl_web/                     # Django + Channels web app (browser UI)
    ├── manage.py
    ├── requirements.txt
    ├── Procfile / build.sh / startup.py   # Render deployment config
    ├── isl_web/                 # Project settings, urls, asgi
    └── predictor/                # App: consumers.py, predictor_engine.py,
                                   # templates, and static assets
```

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Language | Python 3.10 – 3.11 |
| Deep Learning | TensorFlow / Keras |
| Base Model | MobileNetV2 (ImageNet weights) |
| Hand Detection | MediaPipe |
| Computer Vision | OpenCV |
| Text-to-Speech | pyttsx3 (script) / Web Speech API (web app) |
| Web Framework | Django + Channels + Daphne (web app only) |
| Numerics | NumPy |
| Visualization | Matplotlib |

---

## Installation (standalone script)

### 1. Clone the repository

```bash
git clone https://github.com/komal05-web/Indian-Sign-Language-Translator.git
cd Indian-Sign-Language-Translator
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

> Python 3.10–3.11 recommended (see `runtime.txt`). GPU users can swap in `tensorflow-gpu` for faster training.

### 4. Add trained models

Place these (trained separately, not included in the repo):

- `word/isl_best_model.keras`
- `word/label_map.json`
- `sentence/sentence_model.keras`
- `sentence/sentence_label_map.json`

### 5. Run it

```bash
python predict.py
```

A webcam window titled **"ISL Real-Time Translator"** opens, starting in Word Mode by default.

### Training the letter model

```bash
python word_model.py
```

Training history plots save to `word/plots/`.

---

## Web App (isl_web/)

A Django + Channels version with a full browser UI — live webcam feed, mode toggle, phrase cheat-sheet, ISL alphabet reference, and a demo video section, all driven by the same trained models over a WebSocket.

```bash
cd isl_web
python -m venv venv
venv\Scripts\activate            # Windows; source venv/bin/activate on Mac/Linux
pip install -r requirements.txt
copy .env.example .env            # then edit ISL_ROOT and SECRET_KEY inside
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/`, allow camera access, and start signing. See `isl_web/README.md` for full setup, environment variables, and Render deployment instructions.

---

## Usage

### Global

| Key | Action |
|-----|--------|
| `M` | Toggle between Word / Sentence mode |
| `Q` | Quit (script) / stop session (web app) |
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
1. **Phase 1** — Base frozen; only the classification head trained (`lr = 1e-3`)
2. **Phase 2** — Top 30 base layers unfrozen for fine-tuning (`lr = 5e-5`)

### Sentence Model (`sentence/sentence_model.py`)

LSTM-based sequence classifier trained on MediaPipe holistic landmark sequences (pose + both hands). Input shape: `(SEQUENCE_LEN, FEATURE_DIM)`.

---

## Configuration

Key constants in `predict.py` / `utils.py`:

| Parameter | Default | Description |
|---|---|---|
| `IMG_SIZE` | `224` | Input image size for the model |
| `BATCH_SIZE` | `16`–`64` | Training batch size |
| `CONFIDENCE_THRESHOLD` | `0.55` | Minimum confidence to accept a word prediction |
| `SENTENCE_CONFIDENCE_THRESHOLD` | `0.55` | Minimum confidence for sentence prediction |
| `HOLD_FRAMES` | `10` | Frames a sign must be held to commit a letter (~0.33s) |
| `COOLDOWN_FRAMES` | `12` | Frames to ignore after a commit (~0.40s) |
| `BAD_FRAME_TOLERANCE` | `3` | Noisy frames tolerated before resetting hold counter |
| `SMOOTHING_WINDOW` | `3` | Sliding window size for prediction smoother |
| `PADDING_FRACTION` | `0.30` | Padding around the hand crop bounding box |
| `CAMERA_INDEX` | `0` | Webcam index |

---

## Dataset Reference

Letter vocabulary sourced from:
https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl

Phrase vocabulary sourced from:
https://www.kaggle.com/datasets/biswajit002/isl-video-sentences-dataset-for-recognition

---

## Known Limitations

- Designed for **static ISL hand signs** (A–Z alphabet); dynamic/motion signs are handled by Sentence Mode only
- Background/skin segmentation may vary under poor or inconsistent lighting
- Sentence Mode requires a separately trained LSTM/GRU model

---

## Authors

Creators of **HandTalk India**: **komal05-web** ([GitHub](https://github.com/komal05-web)) & **heykayy**

---

## License

This project is licensed under the **MIT License** — see [LICENSE.txt](LICENSE.txt) for details.

Copyright © komal05-web
