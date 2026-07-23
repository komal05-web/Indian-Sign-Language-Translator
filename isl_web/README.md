# ISL Web — HandTalk India

Django + Channels web app that streams webcam frames over a WebSocket to a
TensorFlow/MediaPipe engine (`predictor/predictor_engine.py`) and predicts
ISL letters (Word mode) or full phrases (Sentence mode) in real time.

This folder is self-contained **except for your trained models**, which live
outside `isl_web/` in your original `ISL_project/` folder (`word/` and
`sentence/`). You don't need to copy them in — just point `ISL_ROOT` at them.

## 1. Folder structure expected

```
ISL_project/                  ← your existing project root
├── word/                     ← isl_best_model.keras, label_map.json, sentence_builder.py
├── sentence/                 ← isl_sentence_model.keras, sentence_label_map.json, sentence_model.py
└── isl_web/                  ← THIS folder (replace your old isl_web/ with it)
```

Keep `isl_web/` sitting next to `word/` and `sentence/` — `predictor_engine.py`
runs `import word.sentence_builder` and `import sentence.sentence_model`
directly, so Python needs to find those folders via `ISL_ROOT`.

## 2. Install

```powershell
cd ISL_project\isl_web
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Python 3.11.9 is pinned in `runtime.txt` — TensorFlow 2.16 / MediaPipe 0.10.13
are picky about version, so use 3.11 if you can.

## 3. Configure

```powershell
copy .env.example .env
```

Open `.env` and set:

```
ISL_ROOT=C:\Users\YourName\Downloads\ISL_project\ISL_project
```

(the folder that contains `word\` and `sentence\` — **not** `isl_web` itself).
Leave the four `ISL_..._MODEL_PATH` / `ISL_..._LABEL_PATH` lines blank unless
your model files live somewhere other than `word/` and `sentence/` inside
`ISL_ROOT` — the defaults already point there.

Also set a real `SECRET_KEY` (any random string is fine for local use).

## 4. Set up the database and run

```powershell
python manage.py migrate
python manage.py runserver
```

Watch the terminal output — you should see:

```
[Engine] ISL_ROOT = C:\Users\YourName\...\ISL_project
[Engine] Letter model loaded ✓
[Engine] Label map loaded ✓  (n classes)
[Engine] Sentence model loaded ✓
[Engine] Sentence labels loaded ✓
```

If any line says "NOT FOUND", double check the path in `.env` — it must be
the **exact** absolute folder on your machine.

If WebSocket predictions don't come through with `runserver`, try:

```powershell
daphne isl_web.asgi:application
```

## 5. Open it

Go to `http://127.0.0.1:8000/`, allow camera access, and start signing.

- **Word mode**: hold a letter steady → `Space` confirms it into the word →
  `Enter` speaks the sentence → `C` clears everything.
- **Sentence mode**: press `R` to record a short gesture; after enough frames
  it auto-predicts and speaks the phrase.
- **Global**: `M` toggles Word/Sentence mode, `H` toggles the phrase
  cheat-sheet, `S` saves a screenshot of the current frame.

## 6. Optional: add your own visuals

- Drop 26 images named `A.jpg` … `Z.jpg` into
  `predictor/static/predictor/alphabet/` to populate the alphabet reference
  chart.
- Drop a short screen recording named `demo.mp4` (and optionally
  `demo-poster.jpg`) into `predictor/static/predictor/demo/` to populate the
  "Watch a quick demo" section.

Both are optional — the page looks fine without them, it just shows plain
letters / a placeholder message until you add files.

## Troubleshooting

Paste any traceback back to Claude — it's usually a missing env var, a wrong
path, or a Python version mismatch, all fast to pin down from the exact
error message.
