"""
utils.py - Shared Utilities for ISL Translator
Handles data loading, preprocessing, augmentation, and visualization helpers.
Uses on-disk generators to avoid loading entire dataset into RAM.
"""

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque
from sklearn.model_selection import train_test_split
import tensorflow as tf



# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE         = 224          # MobileNetV2 input size
BATCH_SIZE       = 16           # Default batch size (overridable via parameter)
VALIDATION_SPLIT = 0.20
RANDOM_SEED      = 42

# Paths (resolved relative to this file's directory)
BASE_DIR        = "C:\\Users\\Komal Pandey\\Downloads\\ISL\\ISL"
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "word\\isl_best_model.keras")   # final model
LABEL_MAP_PATH  = os.path.join(BASE_DIR, "word\\label_map.json")
PLOTS_DIR       = os.path.join(BASE_DIR, "word\\plots")


# ─── On-Disk Data Generators (NO full RAM loading) ────────────────────────────

def get_generators_from_directory(dataset_path, batch_size=16, img_size=224):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )

    # Normalize to [0,1]
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Prefetch for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds   = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    class_names = train_ds.class_names
    num_classes = len(class_names)
    label_map   = {i: name for i, name in enumerate(class_names)}

    return train_ds, val_ds, label_map, class_names, num_classes

# ─── Label Map I/O ────────────────────────────────────────────────────────────

def save_label_map(label_map: dict) -> None:
    """Persist label_map to JSON so predict.py can reload it without the dataset."""
    os.makedirs(os.path.dirname(LABEL_MAP_PATH), exist_ok=True)
    serialisable = {str(k): v for k, v in label_map.items()}
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"[Utils] Label map saved → {LABEL_MAP_PATH}")


def load_label_map() -> dict:
    """Load label_map from JSON. Returns {int: str}."""
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"Label map not found: {LABEL_MAP_PATH}. Run train.py first.")
    with open(LABEL_MAP_PATH, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_training_history(history) -> None:
    """Save accuracy and loss curves to the plots/ directory."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    epochs = range(1, len(history.history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history.history["accuracy"],     label="Train Acc",  linewidth=2)
    axes[0].plot(epochs, history.history["val_accuracy"], label="Val Acc",    linewidth=2)
    axes[0].set_title("Model Accuracy", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history.history["loss"],     label="Train Loss", linewidth=2)
    axes[1].plot(epochs, history.history["val_loss"], label="Val Loss",   linewidth=2)
    axes[1].set_title("Model Loss", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "training_history.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"[Utils] Training plot saved → {out_path}")


# ─── Real-Time Helpers ────────────────────────────────────────────────────────

class PredictionSmoother:
    """Majority-vote smoother over a sliding window to reduce prediction flickering."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self._buffer = deque(maxlen=window_size)

    def update(self, prediction: str) -> str:
        self._buffer.append(prediction)
        return max(set(self._buffer), key=self._buffer.count)

    def reset(self) -> None:
        self._buffer.clear()


class FPSCounter:
    """Rolling-average FPS calculator."""

    def __init__(self, avg_over: int = 30):
        self._times = deque(maxlen=avg_over)
        self._prev  = None

    def tick(self, current_time: float) -> float:
        if self._prev is not None:
            delta = current_time - self._prev
            if delta > 0:
                self._times.append(1.0 / delta)
        self._prev = current_time
        return float(np.mean(self._times)) if self._times else 0.0


def preprocess_roi(roi: np.ndarray) -> np.ndarray:
    """
    Prepare a hand ROI (BGR, any size) for model inference.
    Returns np.ndarray of shape (1, IMG_SIZE, IMG_SIZE, 3) in [0, 1].

    NOTE: Model.py no longer applies mobilenet_v2.preprocess_input internally.
    The /255.0 here is the only normalisation step — consistent with the
    training generators which also use rescale=1/255.
    """
    roi_rgb  = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_rsz  = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE))
    roi_norm = roi_rsz.astype(np.float32) / 255.0   # [0, 1] — matches generator
    return np.expand_dims(roi_norm, axis=0)


def draw_overlay(frame: np.ndarray, label: str, confidence: float,
                 fps: float, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Draw bounding box, label, confidence, and FPS on the frame."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label_text = f"{label}  {confidence * 100:.1f}%"
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 10, y1), (0, 255, 0), -1)
    cv2.putText(frame, label_text, (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)

    h = frame.shape[0]
    cv2.putText(frame, "Q: Quit  |  S: Save prediction", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    return frame
