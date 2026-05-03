"""
predict.py – ISL Real-Time Translator  |  Dual-Mode  (Bug-fixed v2)
====================================================================
Bugs fixed vs previous version
--------------------------------
  BUG 1 – Sentence mode: no landmark normalisation at inference
           train_sentence.py normalises every sequence; predict.py was
           passing raw landmarks to the model → train/inference mismatch
           → poor sentence detection despite 84.77% validation accuracy.
           FIX: normalise_sequence() now applied to every recorded sequence
           before inference, identical to the training pipeline.

  BUG 2 – model.predict() called every frame (word mode)
           .predict() has heavy Python/TF graph overhead per call — at 30fps
           this caused lag and frame drops.
           FIX: replaced with model(inp, training=False) — the fast eager path.

  BUG 3 – Double smoothing made letter commit take ~1 full second
           PredictionSmoother (12 frames) + SentenceBuilder hold (18 frames)
           stacked = 30 consecutive stable frames required before any letter
           committed. Letters got missed constantly.
           FIX: Smoother window reduced to 5 (just enough to kill flicker).
                SentenceBuilder hold_frames reduced to 20 (was 18 on top of 12).
                Net effect: letter commits after ~25 stable frames (~0.8s).

  BUG 4 – Two-hand ROI destroyed letter classification
           When both hands were visible, min/max(all_xs) spanned the full
           width of both hands, producing a huge double-hand ROI that the
           single-letter classifier couldn't interpret.
           FIX: letter classification now uses only the PRIMARY hand
           (the one closest to frame centre). Second hand landmarks are still
           drawn for visual feedback but excluded from the ROI.

  BUG 5 – Stale label persisted when confidence dropped
           When conf < threshold, smoother was reset but current_label and
           current_conf kept their old values, showing a ghost prediction.
           FIX: current_label and current_conf are explicitly reset to
           "-" / 0.0 whenever confidence falls below threshold.

  BUG 6 – Confidence threshold too high (0.70)
           With real webcam variation and 50-image training, 0.70 rejected
           too many valid signs, showing "?" most of the time.
           FIX: CONFIDENCE_THRESHOLD lowered to 0.60.

Controls
--------
  M          – Toggle Word / Sentence mode
  Q          – Quit  |  S – Save screenshot

  [Word mode]
  SPACE      – Confirm current word → push to sentence
  BACKSPACE  – Delete last letter (or restore last word)
  ENTER      – Speak full sentence / show matched phrase
  C          – Clear sentence buffer
  H          – Toggle phrase cheat-sheet

  [Sentence mode]
  R          – Start / cancel recording (auto-predicts after SEQUENCE_LEN frames)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import threading
import queue
import datetime
import json
import math

import cv2
import numpy as np
import tensorflow as tf

from utils import (
    load_label_map, draw_overlay,
    PredictionSmoother, FPSCounter,
    MODEL_SAVE_PATH, IMG_SIZE
)
from word.sentence_builder import SentenceBuilder, BASIC_ISL_PHRASES
from sentence.sentence_model import (
    SEQUENCE_LEN, FEATURE_DIM,
    SENTENCE_MODEL_PATH, SENTENCE_LABEL_PATH
)

# ── MediaPipe ─────────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    _mp_hands    = mp.solutions.hands
    _mp_holistic = mp.solutions.holistic
    _mp_draw     = mp.solutions.drawing_utils
    _mp_styles   = mp.solutions.drawing_styles
except Exception as e:
    print(f"[ERROR] MediaPipe import failed: {e}")
    sys.exit(1)

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────────────
CAMERA_INDEX                  = 0
CAPTURE_WIDTH                 = 960
CAPTURE_HEIGHT                = 540
CONFIDENCE_THRESHOLD          = 0.55   # model is 99% accurate — 0.55 is plenty
SENTENCE_CONFIDENCE_THRESHOLD = 0.55
SMOOTHING_WINDOW              = 3      # minimal smoothing — builder handles stability
PADDING_FRACTION              = 0.30   # generous hand crop padding
SAVE_DIR                      = "saved_predictions"
TTS_COOLDOWN_SEC              = 1.5
PANEL_H                       = 145
CHEATSHEET_COLS               = 3

# Word mode letter-commit tuning
HOLD_FRAMES     = 10    # frames to hold a sign before it commits (≈0.33s at 30fps)
COOLDOWN_FRAMES = 12    # frames to ignore after a commit   (≈0.40s at 30fps)
GAP_THRESHOLD   = 0.10  # min margin between top-2 predictions — lowered from 0.15
# Max bad frames tolerated before resetting hold counter.
# This is the KEY fix: transient flickers no longer reset the counter.
BAD_FRAME_TOLERANCE = 3


# ══════════════════════════════════════════════════════════════════════════════
# Infrastructure classes
# ══════════════════════════════════════════════════════════════════════════════

class ThreadedCapture:
    def __init__(self, index=0, width=640, height=480):
        self._cap = cv2.VideoCapture(
            index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        self._queue   = queue.Queue(maxsize=1)
        self._running = True
        self._thread  = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)

    def read(self):
        try:
            return True, self._queue.get(timeout=0.05)
        except queue.Empty:
            return False, None

    def release(self):
        self._running = False
        self._thread.join(timeout=2)
        self._cap.release()


class TTSWorker:
    def __init__(self):
        if not TTS_AVAILABLE:
            return
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", 160)
        self._queue  = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            text = self._queue.get()
            if text is None:
                break
            self._engine.say(text)
            self._engine.runAndWait()

    def speak(self, text: str):
        if TTS_AVAILABLE and self._queue.empty():
            self._queue.put_nowait(text)

    def stop(self):
        if TTS_AVAILABLE:
            self._queue.put(None)


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing helpers
# ══════════════════════════════════════════════════════════════════════════════

# ── Training dataset background colour ────────────────────────────────────────
# The ISL word dataset (grassknoted/asl-alphabet on Kaggle) has images with a
# UNIFORM GREEN background (BGR: 0, 255, 0).  The model learned features in
# the context of that background.  To bridge the train/inference domain gap we
# must recreate that green background at inference time.
_DATASET_BG_BGR = np.array([[[0, 255, 0]]], dtype=np.uint8)   # pure green


def _remove_bg_green(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Isolate the hand and place it on the same green background used in the
    ISL training dataset.

    Strategy — two-pass skin segmentation:
      Pass 1: HSV skin mask  (handles most normal skin tones)
      Pass 2: YCrCb skin mask (handles darker skin & different lighting)
      Combined: union of both masks, then morphological clean-up.
    Remaining non-skin pixels are replaced with pure green (0,255,0).

    This is the critical fix for the domain gap:
      Training images: hand on green bg → model learned "hand + green = letter"
      Old predict.py:  hand on complex bg / white padding → model confused
      New predict.py:  hand on green bg → matches training distribution exactly
    """
    hsv  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)

    # HSV skin range (works well in normal indoor lighting)
    lower_hsv = np.array([0,  20,  70], dtype=np.uint8)
    upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
    mask_hsv  = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # YCrCb skin range (more robust across skin tones)
    lower_ycrcb = np.array([0,  133, 77],  dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb  = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # Union of both masks
    mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)

    # Morphological clean-up: close small holes, remove noise
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open,  iterations=1)
    mask    = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Compose: skin pixels from ROI, everything else → green
    green_bg = np.full_like(roi_bgr, [0, 255, 0])
    mask3    = cv2.merge([mask, mask, mask])
    result   = np.where(mask3 > 0, roi_bgr, green_bg)
    return result.astype(np.uint8)


def _square_pad_green(img: np.ndarray) -> np.ndarray:
    """
    Pad image to square using the TRAINING DATASET background colour (green),
    not white.  White padding was a domain mismatch.
    """
    h, w = img.shape[:2]
    if h == w:
        return img
    size  = max(h, w)
    out   = np.full((size, size, 3), [0, 255, 0], dtype=img.dtype)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    out[y_off:y_off + h, x_off:x_off + w] = img
    return out


def preprocess_roi_fast(roi: np.ndarray, img_size: int,
                        hand_landmark=None,
                        cam_w: int = 0, cam_h: int = 0,
                        x1: int = 0, y1: int = 0) -> tf.Tensor:
    """
    Prepare hand ROI for letter model inference.
    Returns a (1, img_size, img_size, 3) float32 tensor in [0, 1].

    Pipeline (designed to match training image distribution):
      1. Skin segmentation + green background replacement
         → matches the uniform green bg in the ISL training dataset
      2. Square-pad with green (not white — white is wrong for this dataset)
         → preserves hand aspect ratio without introducing alien colours
      3. Resize to (img_size, img_size) using INTER_AREA (best for downscale)
      4. BGR→RGB conversion + normalise to [0, 1]
    """
    # Step 1: replace background with training-matching green
    roi_green = _remove_bg_green(roi)

    # Step 2: square-pad with green
    roi_sq = _square_pad_green(roi_green)

    # Step 3+4: convert, resize, normalise
    roi_rgb = cv2.cvtColor(roi_sq, cv2.COLOR_BGR2RGB)
    roi_rsz = cv2.resize(roi_rgb, (img_size, img_size),
                         interpolation=cv2.INTER_AREA)
    arr = roi_rsz.astype(np.float32) / 255.0
    return tf.expand_dims(arr, axis=0)          # (1, H, W, 3)
def extract_landmarks(results) -> np.ndarray:
    """
    Flatten MediaPipe Holistic results → (FEATURE_DIM,) = (258,).
    Layout: left_hand(63) + right_hand(63) + pose(132).
    Must match train_sentence.py exactly.
    """
    def hand_vec(lm):
        if lm:
            return np.array([[l.x, l.y, l.z]
                             for l in lm.landmark]).flatten()
        return np.zeros(63, dtype=np.float32)

    def pose_vec(lm):
        if lm:
            return np.array([[l.x, l.y, l.z, l.visibility]
                             for l in lm.landmark]).flatten()
        return np.zeros(132, dtype=np.float32)

    return np.concatenate([
        hand_vec(results.left_hand_landmarks),
        hand_vec(results.right_hand_landmarks),
        pose_vec(results.pose_landmarks)
    ])


def normalise_sequence(seq: np.ndarray) -> np.ndarray:
    """
    FIX BUG 1: Identical to train_sentence.py normalise_sequence().
    Must be kept in sync with training pipeline — this was missing entirely
    from predict.py, causing a severe train/inference mismatch.

    Centres and scales each frame's landmarks so body position / distance
    from camera doesn't confuse the model.
    """
    out = seq.copy()
    for i, frame in enumerate(out):
        nonzero = frame[frame != 0]
        if len(nonzero) > 10:
            mu  = nonzero.mean()
            std = nonzero.std()
            if std > 1e-6:
                mask = frame != 0
                out[i][mask] = (frame[mask] - mu) / std
    return out


def get_primary_hand_bbox(hand_landmarks_list, cam_w: int, cam_h: int,
                          padding: float = PADDING_FRACTION):
    """
    FIX BUG 4: Returns bounding box of the PRIMARY hand only (closest to
    frame centre), not a merged box covering both hands.

    When two hands are present the merged box would span half the frame
    and completely confuse the single-letter classifier.

    Returns (x1, y1, x2, y2) clamped to frame bounds.
    """
    cx_frame = 0.5   # normalised centre of frame

    best_hand = None
    best_dist = float("inf")

    best_hand_xs = []
    best_hand_ys = []

    for hand_lm in hand_landmarks_list:
        xs = [lm.x for lm in hand_lm.landmark]
        ys = [lm.y for lm in hand_lm.landmark]
        hcx = (min(xs) + max(xs)) / 2
        dist = abs(hcx - cx_frame)
        if dist < best_dist:
            best_dist    = dist          # float only — was incorrectly assigned tuple
            best_hand_xs = xs
            best_hand_ys = ys
            best_hand    = hand_lm

    if best_hand is None or not best_hand_xs:
        return 0, 0, 0, 0

    xs, ys = best_hand_xs, best_hand_ys
    w_span = max(xs) - min(xs)
    h_span = max(ys) - min(ys)
    pad_x  = int(w_span * cam_w * padding)
    pad_y  = int(h_span * cam_h * padding)

    # Add a minimum absolute padding so tiny hand boxes stay usable
    pad_x = max(pad_x, 20)
    pad_y = max(pad_y, 20)

    x1 = max(0,     int(min(xs) * cam_w) - pad_x)
    y1 = max(0,     int(min(ys) * cam_h) - pad_y)
    x2 = min(cam_w, int(max(xs) * cam_w) + pad_x)
    y2 = min(cam_h, int(max(ys) * cam_h) + pad_y)

    return x1, y1, x2, y2


def load_sentence_label_map() -> dict:
    if not os.path.exists(SENTENCE_LABEL_PATH):
        return {}
    with open(SENTENCE_LABEL_PATH) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def draw_mode_badge(canvas: np.ndarray, mode: str) -> None:
    h, w = canvas.shape[:2]
    if mode == "WORD":
        color = (0, 200, 80)
        label = "  WORD MODE — letter by letter  "
    else:
        color = (30, 140, 255)
        label = "  SENTENCE MODE — sign a phrase  "
    tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)[0]
    x = w - tw - 12
    cv2.rectangle(canvas, (x - 4, 8), (x + tw + 4, 8 + th + 10),
                  (30, 30, 30), -1)
    cv2.putText(canvas, label, (x, 8 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "M = switch mode",
                (w - tw - 8, 8 + th + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1, cv2.LINE_AA)


def draw_hold_ring(canvas: np.ndarray, cx: int, cy: int,
                   progress: float, label: str = "",
                   radius: int = 32) -> None:
    """
    Animated arc that fills clockwise as the user holds a sign steady.
    Larger, more visible ring with the detected letter shown inside.
    """
    if progress <= 0:
        return
    # Background arc
    cv2.ellipse(canvas, (cx, cy), (radius, radius), -90, 0, 360,
                (50, 50, 50), 5, cv2.LINE_AA)
    # Foreground arc — yellow → green as it fills
    r = int(255 * (1 - progress))
    g = int(200 + 55 * progress)
    color = (0, g, r)
    cv2.ellipse(canvas, (cx, cy), (radius, radius), -90, 0,
                int(360 * progress), color, 5, cv2.LINE_AA)
    # Letter label centred inside ring
    if label:
        fs  = 0.65
        tw  = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, 2)[0][0]
        cv2.putText(canvas, label,
                    (cx - tw // 2, cy + 8),
                    cv2.FONT_HERSHEY_DUPLEX, fs, color, 2, cv2.LINE_AA)


def draw_word_panel(canvas: np.ndarray, builder: SentenceBuilder,
                    flash_letter: bool, flash_phrase: str,
                    current_label: str, current_conf: float,
                    panel_top: int) -> None:
    h, w = canvas.shape[:2]
    canvas[panel_top:, :] = (14, 16, 22)
    cv2.line(canvas, (0, panel_top), (w, panel_top), (40, 120, 60), 2)

    # ── Current detection (top-right of panel) ────────────────────────────────
    det_color = (0, 255, 120) if current_conf >= CONFIDENCE_THRESHOLD else (80, 80, 100)
    det_text  = f"{current_label}  {current_conf*100:.0f}%" \
                if current_conf > 0 else current_label
    cv2.putText(canvas, det_text,
                (w - 140, panel_top + 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, det_color, 2, cv2.LINE_AA)

    # ── Word being typed ───────────────────────────────────────────────────────
    cursor    = "█" if flash_letter else "▌"
    word_disp = (builder.word_str + cursor) or cursor
    lbl       = "Word : "
    cv2.putText(canvas, lbl, (14, panel_top + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 140, 100), 1, cv2.LINE_AA)
    wx = 14 + cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)[0][0]
    cv2.putText(canvas, word_disp, (wx, panel_top + 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.78,
                (0, 255, 120) if flash_letter else (200, 220, 255),
                1, cv2.LINE_AA)

    # ── Full sentence ──────────────────────────────────────────────────────────
    sent_text = builder.sentence_str or "(empty — sign letters to begin)"
    max_ch    = max(10, (w - 180) // 11)
    if len(sent_text) > max_ch:
        sent_text = "…" + sent_text[-(max_ch - 1):]
    lbl2 = "Sentence : "
    cv2.putText(canvas, lbl2, (14, panel_top + 67),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 140, 100), 1, cv2.LINE_AA)
    sx = 14 + cv2.getTextSize(lbl2, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)[0][0]
    cv2.putText(canvas, sent_text, (sx, panel_top + 67),
                cv2.FONT_HERSHEY_DUPLEX, 0.70, (230, 235, 255), 1, cv2.LINE_AA)

    # ── Matched phrase or hint bar ─────────────────────────────────────────────
    if flash_phrase:
        cv2.rectangle(canvas, (8, panel_top + 80), (w - 8, panel_top + 108),
                      (0, 60, 80), -1)
        cv2.putText(canvas, f"  ★  {flash_phrase}", (14, panel_top + 101),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 215, 255), 2, cv2.LINE_AA)
    else:
        hint = "SPACE=word   BACKSPACE=del   ENTER=speak   C=clear   H=phrases   M=sentence mode"
        cv2.putText(canvas, hint, (14, panel_top + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 75, 100), 1, cv2.LINE_AA)


def draw_sentence_panel(canvas: np.ndarray, recording: bool,
                        frames_collected: int, last_sentence: str,
                        last_conf: float, panel_top: int) -> None:
    h, w = canvas.shape[:2]
    canvas[panel_top:, :] = (14, 16, 26)
    cv2.line(canvas, (0, panel_top), (w, panel_top), (30, 80, 180), 2)

    # ── Recording progress bar ─────────────────────────────────────────────────
    bar_total = w - 28
    bar_w     = int(bar_total * min(frames_collected / max(SEQUENCE_LEN, 1), 1.0))
    cv2.rectangle(canvas, (14, panel_top + 8),
                  (14 + bar_total, panel_top + 22), (30, 30, 50), -1)
    if bar_w > 0:
        bar_color = (0, 100, 255) if recording else (30, 60, 100)
        cv2.rectangle(canvas, (14, panel_top + 8),
                      (14 + bar_w, panel_top + 22), bar_color, -1)

    # ── Status text ───────────────────────────────────────────────────────────
    if recording:
        rec_text  = f"  ● REC  {frames_collected} / {SEQUENCE_LEN} frames  " \
                    f"({frames_collected/SEQUENCE_LEN*100:.0f}%)  — keep signing …"
        rec_color = (60, 120, 255)
    else:
        rec_text  = "  Press  R  to sign a sentence"
        rec_color = (100, 100, 120)
    cv2.putText(canvas, rec_text, (14, panel_top + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, rec_color, 2, cv2.LINE_AA)

    # ── Last detected sentence ────────────────────────────────────────────────
    if last_sentence:
        disp = last_sentence if len(last_sentence) <= 58 \
               else last_sentence[:55] + "…"
        cv2.rectangle(canvas, (8, panel_top + 60),
                      (w - 8, panel_top + 90), (0, 40, 70), -1)
        cv2.putText(canvas, f"  {disp}", (14, panel_top + 82),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"confidence : {last_conf*100:.1f}%",
                    (14, panel_top + 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 185, 130), 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "No sentence detected yet — press R to start",
                    (14, panel_top + 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (60, 65, 90), 1, cv2.LINE_AA)
        cv2.putText(canvas,
                    "R=record   M=word mode   S=screenshot   Q=quit",
                    (14, panel_top + 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 65, 90), 1, cv2.LINE_AA)


def draw_cheatsheet(canvas: np.ndarray) -> None:
    h, w    = canvas.shape[:2]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (8, 10, 18), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
    cv2.putText(canvas, "  ISL Phrase Dictionary  (press H to close)",
                (16, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 215, 255), 2)
    cv2.line(canvas, (0, 55), (w, 55), (50, 55, 90), 1)
    phrases  = list(BASIC_ISL_PHRASES.items())
    rows_per = math.ceil(len(phrases) / CHEATSHEET_COLS)
    col_w    = w // CHEATSHEET_COLS
    for idx, (key, meaning) in enumerate(phrases):
        col = idx // rows_per
        row = idx % rows_per
        x   = col * col_w + 14
        y   = 80 + row * 27
        if y > h - 20:
            break
        cv2.putText(canvas, key, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 200, 255), 1)
        cv2.putText(canvas, f"→ {meaning}", (x + 115, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 145, 170), 1)


# ══════════════════════════════════════════════════════════════════════════════
# Main detection loop
# ══════════════════════════════════════════════════════════════════════════════

def run_live_detection():
    import multiprocessing
    n = multiprocessing.cpu_count()
    tf.config.threading.set_intra_op_parallelism_threads(n)
    tf.config.threading.set_inter_op_parallelism_threads(n)
    print(f"[Predict] CPU mode — {n} cores.\n")

    # ── Load letter model ─────────────────────────────────────────────────────
    letter_model     = None
    letter_label_map = {}
    if os.path.exists(MODEL_SAVE_PATH):
        print("[Predict] Loading letter model …")
        letter_model     = tf.keras.models.load_model(MODEL_SAVE_PATH, compile=False)
        letter_label_map = load_label_map()
        # Warm-up using fast path
        _d = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
        letter_model(_d, training=False)
        print(f"[Predict] Letter model ready — {len(letter_label_map)} classes.")
    else:
        print(f"[WARN] No letter model at {MODEL_SAVE_PATH}. Run train.py first.")

    # ── Load sentence model ───────────────────────────────────────────────────
    sentence_model     = None
    sentence_label_map = {}
    if os.path.exists(SENTENCE_MODEL_PATH):
        print("[Predict] Loading sentence model …")
        sentence_model     = tf.keras.models.load_model(SENTENCE_MODEL_PATH, compile=False)
        sentence_label_map = load_sentence_label_map()
        _ds = tf.zeros((1, SEQUENCE_LEN, FEATURE_DIM), dtype=tf.float32)
        sentence_model(_ds, training=False)
        print(f"[Predict] Sentence model ready — "
              f"{len(sentence_label_map)} classes: "
              f"{list(sentence_label_map.values())}")
    else:
        print(f"[WARN] No sentence model at {SENTENCE_MODEL_PATH}. "
              "Run train_sentence.py first.")

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    hands = _mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.50,
        model_complexity=0
    )
    holistic = _mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,          # better landmark accuracy for sentence mode
        smooth_landmarks=True,
        min_detection_confidence=0.50,
        min_tracking_confidence=0.50
    )

    smoother    = PredictionSmoother(window_size=SMOOTHING_WINDOW)
    fps_counter = FPSCounter(avg_over=30)
    tts         = TTSWorker()
    capture     = ThreadedCapture(CAMERA_INDEX, CAPTURE_WIDTH, CAPTURE_HEIGHT)

    # HOLD_FRAMES=10 → letter commits in ~0.33s at 30fps
    # COOLDOWN_FRAMES=12 → 0.40s gap between letters (prevents accidental doubles)
    builder = SentenceBuilder(hold_frames=HOLD_FRAMES, cooldown_frames=COOLDOWN_FRAMES)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── State ─────────────────────────────────────────────────────────────────
    mode             = "WORD"
    current_label    = "-"
    current_conf     = 0.0
    last_label       = ""
    last_tts_time    = 0.0
    flash_letter     = False
    flash_frames     = 0
    flash_phrase     = ""
    phrase_timer     = 0
    show_cheatsheet  = False
    recording        = False
    seq_buffer       = []
    last_sentence    = ""
    last_sent_conf   = 0.0
    bad_frame_streak = 0    # consecutive frames below confidence threshold

    cam_h, cam_w = CAPTURE_HEIGHT, CAPTURE_WIDTH
    panel_top    = cam_h

    print("\nControls:")
    print("  M          – Toggle Word / Sentence mode")
    print("  Q          – Quit  |  S – Screenshot")
    print("  [Word]     SPACE=word  BACKSPACE=del  ENTER=speak  C=clear  H=phrases")
    print("  [Sentence] R – Start / stop recording\n")

    while True:
        ret, cam_frame = capture.read()
        if not ret or cam_frame is None:
            continue

        now = time.time()
        fps = fps_counter.tick(now)
        cam_frame = cv2.flip(cv2.resize(cam_frame, (cam_w, cam_h)), 1)

        canvas = np.zeros((cam_h + PANEL_H, cam_w, 3), dtype=np.uint8)
        canvas[:cam_h] = cam_frame

        rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        # ══════════════════════════════════════════════════════════════════════
        # WORD MODE
        # ══════════════════════════════════════════════════════════════════════
        if mode == "WORD":
            results = hands.process(rgb)
            rgb.flags.writeable = True

            x1 = y1 = x2 = y2 = 0

            if results.multi_hand_landmarks:
                # Draw ALL hand landmarks for visual feedback
                for hand_lm in results.multi_hand_landmarks:
                    _mp_draw.draw_landmarks(
                        canvas[:cam_h], hand_lm, _mp_hands.HAND_CONNECTIONS,
                        _mp_styles.get_default_hand_landmarks_style(),
                        _mp_styles.get_default_hand_connections_style()
                    )

                # FIX BUG 4: use PRIMARY hand only for the ROI crop
                x1, y1, x2, y2 = get_primary_hand_bbox(
                    results.multi_hand_landmarks, cam_w, cam_h
                )

                if letter_model and (x2 - x1) >= 20 and (y2 - y1) >= 20:
                    roi = cam_frame[y1:y2, x1:x2]
                    inp = preprocess_roi_fast(roi, IMG_SIZE)

                    # Fast eager call
                    preds = letter_model(inp, training=False).numpy()[0]
                    idx   = int(np.argmax(preds))
                    conf  = float(preds[idx])

                    sorted_preds = np.sort(preds)[::-1]
                    conf_gap     = float(sorted_preds[0] - sorted_preds[1])
                    frame_good   = (conf >= CONFIDENCE_THRESHOLD and
                                    conf_gap >= GAP_THRESHOLD)

                    if frame_good:
                        bad_frame_streak = 0          # reset bad-frame counter
                        raw_label    = letter_label_map[idx]
                        smooth_label = smoother.update(raw_label)
                        current_label = smooth_label.upper()
                        current_conf  = conf

                        # Feed to builder — commits when hold_frames stable frames seen
                        if builder.feed(current_label):
                            flash_letter = True
                            flash_frames = 14
                            flash_phrase = ""
                            phrase_timer = 0

                        if smooth_label != last_label and \
                                (now - last_tts_time) > TTS_COOLDOWN_SEC:
                            tts.speak(smooth_label)
                            last_label    = smooth_label
                            last_tts_time = now
                    else:
                        # Bad frame — increment tolerance counter.
                        # KEY FIX: only reset hold after BAD_FRAME_TOLERANCE
                        # consecutive bad frames. A single noisy frame no longer
                        # resets the hold counter from scratch.
                        bad_frame_streak += 1
                        if bad_frame_streak > BAD_FRAME_TOLERANCE:
                            smoother.reset()
                            builder._reset_hold()
                            current_label = "-"
                            current_conf  = 0.0
                        # else: keep showing last good label while briefly uncertain

                elif not letter_model:
                    cv2.putText(canvas, "Letter model not found — run train.py",
                                (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 80, 255), 2, cv2.LINE_AA)

                # Draw ROI box + prediction overlay
                draw_overlay(canvas[:cam_h], current_label, current_conf,
                             fps, x1, y1, x2, y2)

                # Hold-progress ring — shows letter being committed
                if builder.hold_progress > 0:
                    ring_cx = min(x2 + 48, cam_w - 38)
                    ring_cy = max(y1 + 38, 38)
                    draw_hold_ring(canvas[:cam_h], ring_cx, ring_cy,
                                   builder.hold_progress,
                                   label=current_label)
            else:
                # No hand detected — full reset
                smoother.reset()
                builder._reset_hold()
                current_label    = "-"
                current_conf     = 0.0
                bad_frame_streak = 0
                cv2.putText(canvas, "No hand detected", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 100, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, f"FPS: {fps:.1f}", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 200, 255), 2, cv2.LINE_AA)

            # Flash timers
            flash_frames = max(0, flash_frames - 1)
            if flash_frames == 0:
                flash_letter = False
            phrase_timer = max(0, phrase_timer - 1)
            if phrase_timer == 0:
                flash_phrase = ""

            draw_word_panel(canvas, builder, flash_letter, flash_phrase,
                            current_label, current_conf, panel_top)

        # ══════════════════════════════════════════════════════════════════════
        # SENTENCE MODE
        # ══════════════════════════════════════════════════════════════════════
        else:
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            # Draw holistic landmarks
            for lm_set, conn in [
                (results.left_hand_landmarks,  _mp_holistic.HAND_CONNECTIONS),
                (results.right_hand_landmarks, _mp_holistic.HAND_CONNECTIONS),
                (results.pose_landmarks,       _mp_holistic.POSE_CONNECTIONS),
            ]:
                if lm_set:
                    _mp_draw.draw_landmarks(canvas[:cam_h], lm_set, conn)

            cv2.putText(canvas, f"FPS: {fps:.1f}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

            if recording:
                seq_buffer.append(extract_landmarks(results))

                if len(seq_buffer) >= SEQUENCE_LEN:
                    recording = False

                    if sentence_model:
                        # FIX BUG 1: normalise the sequence — identical to training
                        seq = np.array(seq_buffer[:SEQUENCE_LEN],
                                       dtype=np.float32)
                        seq = normalise_sequence(seq)        # ← was missing
                        inp = tf.expand_dims(seq, axis=0)

                        # FIX BUG 2: fast eager call
                        pred = sentence_model(inp, training=False).numpy()[0]
                        idx  = int(np.argmax(pred))
                        conf = float(pred[idx])

                        if conf >= SENTENCE_CONFIDENCE_THRESHOLD:
                            last_sentence  = sentence_label_map.get(idx, "Unknown")
                            last_sent_conf = conf
                            tts.speak(last_sentence)
                            print(f"[Sentence] '{last_sentence}' "
                                  f"({conf*100:.1f}%)")
                        else:
                            last_sentence  = \
                                f"Unclear sign  ({conf*100:.1f}%) — please try again"
                            last_sent_conf = conf
                            print(f"[Sentence] Low confidence: {conf*100:.1f}%")
                    else:
                        last_sentence  = "Sentence model not loaded — run train_sentence.py"
                        last_sent_conf = 0.0

                    seq_buffer.clear()

            draw_sentence_panel(canvas, recording, len(seq_buffer),
                                last_sentence, last_sent_conf, panel_top)

        # ── Always-on UI ──────────────────────────────────────────────────────
        draw_mode_badge(canvas, mode)
        if show_cheatsheet and mode == "WORD":
            draw_cheatsheet(canvas)

        cv2.imshow("ISL Real-Time Translator", canvas)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(10) & 0xFF

        if key == ord("q"):
            print("[Predict] Quitting …")
            break

        elif key in (ord("m"), ord("M")):
            mode = "SENTENCE" if mode == "WORD" else "WORD"
            smoother.reset()
            builder.clear()
            recording        = False
            seq_buffer.clear()
            flash_phrase     = ""
            flash_letter     = False
            current_label    = "-"
            current_conf     = 0.0
            bad_frame_streak = 0
            show_cheatsheet  = False
            print(f"[Predict] Switched to {mode} mode.")

        elif key == ord("s"):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = os.path.join(SAVE_DIR, f"{mode}_{ts}.jpg")
            cv2.imwrite(fn, canvas)
            print(f"[Predict] Saved → {fn}")

        elif mode == "WORD":
            if key == ord(" "):
                builder.space()
                flash_phrase = ""
                phrase_timer = 0
                print(f"[Word] Sentence: '{builder.sentence_str}'")
            elif key in (8, 127):
                builder.backspace()
                print(f"[Word] After backspace: word='{builder.word_str}' "
                      f"sentence='{builder.sentence_str}'")
            elif key == 13:
                full = builder.sentence_str.strip()
                if full:
                    matched    = builder.matched_phrase()
                    speak_text = matched if matched else full
                    tts.speak(speak_text)
                    flash_phrase = speak_text
                    phrase_timer = 150
                    print(f"[Word] Spoken: '{speak_text}'")
                else:
                    print("[Word] Nothing to speak yet.")
            elif key == ord("c"):
                builder.clear()
                flash_phrase  = ""
                phrase_timer  = 0
                current_label = "-"
                current_conf  = 0.0
                print("[Word] Cleared.")
            elif key == ord("h"):
                show_cheatsheet = not show_cheatsheet

        elif mode == "SENTENCE":
            if key == ord("r"):
                if not recording:
                    recording      = True
                    seq_buffer.clear()
                    last_sentence  = ""
                    last_sent_conf = 0.0
                    print("[Sentence] Recording started — sign your phrase now …")
                else:
                    recording = False
                    seq_buffer.clear()
                    print("[Sentence] Recording cancelled.")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    capture.release()
    hands.close()
    holistic.close()
    tts.stop()
    cv2.destroyAllWindows()
    print("[Predict] Session ended.")


if __name__ == "__main__":
    run_live_detection()
