"""
predictor_engine.py  — fixed version
======================================
Fixes applied:
  1. SentenceBuilder fallback — works even if ISL imports fail
  2. Model path check prints actual path so you can see what's wrong
  3. rgb.flags.writeable fix — copy frame before passing to MediaPipe
  4. Debug logging at every key step so server terminal shows what's happening
  5. ISL_ROOT is inserted at index 0 and checked on every import attempt
"""

import os
import sys
import json
import threading
import numpy as np
import cv2
import tensorflow as tf

# ── Add ISL project root to path ─────────────────────────────────────────────
ISL_ROOT = os.getenv(
    "ISL_ROOT",
    r"C:\Users\Komal Pandey\Downloads\ISL\ISL"
)
print(f"[Engine] ISL_ROOT = {ISL_ROOT}")
if ISL_ROOT not in sys.path:
    sys.path.insert(0, ISL_ROOT)

# ── MediaPipe ─────────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    _mp_hands    = mp.solutions.hands
    _mp_holistic = mp.solutions.holistic
    MEDIAPIPE_OK = True
    print("[Engine] MediaPipe OK ✓")
except Exception as e:
    print(f"[Engine] MediaPipe FAILED: {e}")
    MEDIAPIPE_OK = False

# ── ISL project imports ───────────────────────────────────────────────────────
try:
    from word.sentence_builder import SentenceBuilder, BASIC_ISL_PHRASES
    print("[Engine] SentenceBuilder imported ✓")
    BUILDER_OK = True
except Exception as e:
    print(f"[Engine] SentenceBuilder import FAILED: {e}")
    print(f"[Engine]   → Make sure word/sentence_builder.py exists in {ISL_ROOT}")
    BUILDER_OK = False
    SentenceBuilder = None

try:
    from sentence.sentence_model import SEQUENCE_LEN, FEATURE_DIM
    print(f"[Engine] sentence_model imported ✓  SEQUENCE_LEN={SEQUENCE_LEN}")
    SENT_MODEL_OK = True
except Exception as e:
    print(f"[Engine] sentence_model import FAILED: {e}")
    SEQUENCE_LEN  = 45
    FEATURE_DIM   = 258
    SENT_MODEL_OK = False

# ── Constants ─────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD          = 0.55
SENTENCE_CONFIDENCE_THRESHOLD = 0.55
PADDING_FRACTION              = 0.30
IMG_SIZE                      = 224


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing helpers
# ══════════════════════════════════════════════════════════════════════════════

def _remove_bg_green(roi_bgr):
    hsv    = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    ycrcb  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    m_hsv  = cv2.inRange(hsv,   np.array([0,20,70]),   np.array([20,255,255]))
    m_ycr  = cv2.inRange(ycrcb, np.array([0,133,77]),  np.array([255,173,127]))
    mask   = cv2.bitwise_or(m_hsv, m_ycr)
    kc     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    ko     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  ko, iterations=1)
    mask   = cv2.GaussianBlur(mask, (5,5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    green  = np.full_like(roi_bgr, [0,255,0])
    mask3  = cv2.merge([mask,mask,mask])
    return np.where(mask3>0, roi_bgr, green).astype(np.uint8)


def _square_pad_green(img):
    h, w = img.shape[:2]
    if h == w:
        return img
    size = max(h, w)
    out  = np.full((size, size, 3), [0,255,0], dtype=img.dtype)
    out[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = img
    return out


def preprocess_roi(roi):
    roi = _remove_bg_green(roi)
    roi = _square_pad_green(roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    arr = roi.astype(np.float32) / 255.0
    return tf.expand_dims(arr, axis=0)


def extract_landmarks(results):
    def hand_vec(lm):
        if lm:
            return np.array([[l.x, l.y, l.z] for l in lm.landmark]).flatten()
        return np.zeros(63, dtype=np.float32)
    def pose_vec(lm):
        if lm:
            return np.array([[l.x, l.y, l.z, l.visibility] for l in lm.landmark]).flatten()
        return np.zeros(132, dtype=np.float32)
    return np.concatenate([
        hand_vec(results.left_hand_landmarks),
        hand_vec(results.right_hand_landmarks),
        pose_vec(results.pose_landmarks),
    ])


def normalise_sequence(seq):
    out = seq.copy()
    for i, frame in enumerate(out):
        nz = frame[frame != 0]
        if len(nz) > 10:
            mu, std = nz.mean(), nz.std()
            if std > 1e-6:
                mask = frame != 0
                out[i][mask] = (frame[mask] - mu) / std
    return out


def get_primary_hand_bbox(lm_list, cam_w, cam_h, padding=PADDING_FRACTION):
    best_xs, best_ys, best_dist = [], [], float("inf")
    for lm in lm_list:
        xs   = [l.x for l in lm.landmark]
        ys   = [l.y for l in lm.landmark]
        dist = abs((min(xs)+max(xs))/2 - 0.5)
        if dist < best_dist:
            best_dist, best_xs, best_ys = dist, xs, ys
    if not best_xs:
        return 0, 0, 0, 0
    pad_x = max(int((max(best_xs)-min(best_xs))*cam_w*padding), 20)
    pad_y = max(int((max(best_ys)-min(best_ys))*cam_h*padding), 20)
    x1 = max(0,     int(min(best_xs)*cam_w) - pad_x)
    y1 = max(0,     int(min(best_ys)*cam_h) - pad_y)
    x2 = min(cam_w, int(max(best_xs)*cam_w) + pad_x)
    y2 = min(cam_h, int(max(best_ys)*cam_h) + pad_y)
    return x1, y1, x2, y2


# ── Simple SentenceBuilder fallback (if ISL import fails) ────────────────────
class _FallbackBuilder:
    """Minimal SentenceBuilder so word mode still works without ISL imports."""
    def __init__(self):
        self.current_word = []
        self.sentence     = []
        self._hold        = 0
        self._last        = None
        self._cooldown    = 0
        self.hold_frames  = 10
        self.cooldown_frames = 12

    def feed(self, letter):
        if self._cooldown > 0:
            self._cooldown -= 1
            return False
        if letter == self._last:
            self._hold += 1
        else:
            self._last = letter
            self._hold = 1
            return False
        if self._hold >= self.hold_frames:
            self.current_word.append(letter)
            self._last    = None
            self._hold    = 0
            self._cooldown = self.cooldown_frames
            return True
        return False

    def space(self):
        w = self.word_str.strip()
        if w:
            self.sentence.append(w)
        self.current_word.clear()
        self._last = None; self._hold = 0

    def backspace(self):
        if self.current_word:
            self.current_word.pop()
        elif self.sentence:
            self.current_word = list(self.sentence.pop())

    def clear(self):
        self.current_word.clear()
        self.sentence.clear()
        self._last = None; self._hold = 0; self._cooldown = 0

    def matched_phrase(self):
        return ""

    @property
    def word_str(self):
        return "".join(self.current_word)

    @property
    def sentence_str(self):
        parts = self.sentence[:]
        if self.current_word:
            parts.append(self.word_str)
        return " ".join(parts)

    @property
    def hold_progress(self):
        if not self._hold or not self.hold_frames:
            return 0.0
        return min(self._hold / self.hold_frames, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# ISLEngine
# ══════════════════════════════════════════════════════════════════════════════

class ISLEngine:
    _model_lock   = threading.Lock()
    _letter_model = None
    _label_map    = None
    _sent_model   = None
    _sent_labels  = None

    @classmethod
    def load_models(cls, word_model_path, label_map_path,
                    sent_model_path, sent_label_path):
        print(f"[Engine] Loading models…")
        print(f"  word_model : {word_model_path}")
        print(f"  label_map  : {label_map_path}")
        print(f"  sent_model : {sent_model_path}")
        print(f"  sent_labels: {sent_label_path}")

        with cls._model_lock:
            if os.path.exists(word_model_path):
                try:
                    cls._letter_model = tf.keras.models.load_model(
                        word_model_path, compile=False)
                    print("[Engine] Letter model loaded ✓")
                except Exception as e:
                    print(f"[Engine] Letter model FAILED: {e}")
            else:
                print(f"[Engine] Letter model NOT FOUND at: {word_model_path}")

            if os.path.exists(label_map_path):
                with open(label_map_path) as f:
                    cls._label_map = {int(k): v for k, v in json.load(f).items()}
                print(f"[Engine] Label map loaded ✓  ({len(cls._label_map)} classes)")
            else:
                print(f"[Engine] Label map NOT FOUND at: {label_map_path}")

            if os.path.exists(sent_model_path):
                try:
                    cls._sent_model = tf.keras.models.load_model(
                        sent_model_path, compile=False)
                    print("[Engine] Sentence model loaded ✓")
                except Exception as e:
                    print(f"[Engine] Sentence model FAILED: {e}")
            else:
                print(f"[Engine] Sentence model NOT FOUND at: {sent_model_path}")

            if os.path.exists(sent_label_path):
                with open(sent_label_path) as f:
                    cls._sent_labels = {int(k): v for k, v in json.load(f).items()}
                print(f"[Engine] Sentence labels loaded ✓")
            else:
                print(f"[Engine] Sentence labels NOT FOUND at: {sent_label_path}")

    def __init__(self, mode="WORD"):
        self.mode      = mode
        self.seq_buffer = []
        self.recording  = False
        self._smoother_buf = []
        self._smooth_win   = 3

        # Builder
        if BUILDER_OK and SentenceBuilder is not None:
            self.builder = SentenceBuilder(hold_frames=10, cooldown_frames=12)
        else:
            self.builder = _FallbackBuilder()

        # MediaPipe
        if MEDIAPIPE_OK:
            self.hands = _mp_hands.Hands(
                static_image_mode=False, max_num_hands=2,
                min_detection_confidence=0.6, min_tracking_confidence=0.5)
            self.holistic = _mp_holistic.Holistic(
                static_image_mode=False, model_complexity=1,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        else:
            self.hands = self.holistic = None

    def set_mode(self, mode):
        self.mode = mode
        self._smoother_buf.clear()
        self.builder.clear()
        self.seq_buffer.clear()
        self.recording = False

    def toggle_recording(self):
        if self.recording:
            self.recording = False
            self.seq_buffer.clear()
            return {"action": "recording_cancelled"}
        self.recording = True
        self.seq_buffer.clear()
        return {"action": "recording_started"}

    def _smooth(self, label):
        self._smoother_buf.append(label)
        if len(self._smoother_buf) > self._smooth_win:
            self._smoother_buf.pop(0)
        return max(set(self._smoother_buf), key=self._smoother_buf.count)

    def process_frame(self, frame_bytes):
        # Decode JPEG from browser
        arr   = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "bad frame"}

        cam_h, cam_w = frame.shape[:2]

        # IMPORTANT: make a writeable copy for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()

        result = {
            "mode":            self.mode,
            "letter":          "-",
            "confidence":      0.0,
            "word":            self.builder.word_str,
            "sentence":        self.builder.sentence_str,
            "hold_progress":   0.0,
            "hand_detected":   False,
            "recording":       self.recording,
            "sentence_result": "",
            "sentence_conf":   0.0,
        }

        # ── WORD MODE ─────────────────────────────────────────────────────────
        if self.mode == "WORD":
            if not self.hands:
                result["error"] = "MediaPipe not available"
                return result

            hand_res = self.hands.process(rgb)

            if hand_res.multi_hand_landmarks:
                result["hand_detected"] = True
                x1, y1, x2, y2 = get_primary_hand_bbox(
                    hand_res.multi_hand_landmarks, cam_w, cam_h)

                if x2 > x1 and y2 > y1:
                    if self.__class__._letter_model is not None:
                        roi  = frame[y1:y2, x1:x2]
                        inp  = preprocess_roi(roi)
                        pred = self.__class__._letter_model(inp, training=False).numpy()[0]
                        idx  = int(np.argmax(pred))
                        conf = float(pred[idx])

                        if conf >= CONFIDENCE_THRESHOLD:
                            lbl = (self.__class__._label_map.get(idx, "?")
                                   if self.__class__._label_map else str(idx))
                            lbl = self._smooth(lbl)
                            committed = self.builder.feed(lbl)
                            result["letter"]           = lbl
                            result["confidence"]       = round(conf, 3)
                            result["hold_progress"]    = round(self.builder.hold_progress, 3)
                            result["letter_committed"] = committed
                    else:
                        # No model — at least show hand detected
                        result["letter"] = "?"
                        result["confidence"] = 0.0

            result["word"]     = self.builder.word_str
            result["sentence"] = self.builder.sentence_str

        # ── SENTENCE MODE ─────────────────────────────────────────────────────
        else:
            if not self.holistic:
                result["error"] = "MediaPipe not available"
                return result

            hol_res = self.holistic.process(rgb)

            if self.recording:
                self.seq_buffer.append(extract_landmarks(hol_res))
                result["recording"]     = True
                result["buffer_frames"] = len(self.seq_buffer)
                result["seq_len"]       = SEQUENCE_LEN

                if len(self.seq_buffer) >= SEQUENCE_LEN:
                    self.recording = False
                    seq = normalise_sequence(
                        np.array(self.seq_buffer[:SEQUENCE_LEN], dtype=np.float32))
                    self.seq_buffer.clear()

                    if self.__class__._sent_model is not None:
                        inp  = tf.expand_dims(seq, axis=0)
                        pred = self.__class__._sent_model(inp, training=False).numpy()[0]
                        idx  = int(np.argmax(pred))
                        conf = float(pred[idx])
                        if conf >= SENTENCE_CONFIDENCE_THRESHOLD:
                            label = (self.__class__._sent_labels.get(idx, "Unknown")
                                     if self.__class__._sent_labels else str(idx))
                        else:
                            label = f"Unclear ({conf*100:.0f}%) — try again"
                        result["sentence_result"] = label
                        result["sentence_conf"]   = round(conf, 3)
                        result["recording"]       = False
                    else:
                        result["sentence_result"] = "Sentence model not loaded"
                        result["recording"]       = False

        return result

    def word_command(self, cmd):
        if cmd == "SPACE":
            self.builder.space()
        elif cmd == "BACKSPACE":
            self.builder.backspace()
        elif cmd == "CLEAR":
            self.builder.clear()
        matched = self.builder.matched_phrase() if hasattr(self.builder, "matched_phrase") else ""
        return {
            "word":     self.builder.word_str,
            "sentence": self.builder.sentence_str,
            "matched":  matched,
        }

    def close(self):
        try:
            if self.hands:    self.hands.close()
            if self.holistic: self.holistic.close()
        except Exception:
            pass
