"""
model.py - ISL Translator Model Architecture
Defines MobileNetV2-based transfer learning model optimized for real-time inference.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.optimizers import Adam


def build_model(num_classes: int, input_shape: tuple = (224, 224, 3),
                learning_rate: float = 1e-3):
    """
    Build a MobileNetV2-based transfer learning model for ISL gesture classification.

    Args:
        num_classes    : Number of gesture classes to classify.
        input_shape    : Input image dimensions (H, W, C).
        learning_rate  : Adam optimizer learning rate.

    Returns:
        (compiled Keras model, base_model reference)
    """
    # Load MobileNetV2 pretrained on ImageNet, excluding top classification layers
    base_model = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # ── Phase 1: Freeze all base layers for initial training ──────────────────
    base_model.trainable = False

    # Build the full model with a lightweight custom head
    inputs = tf.keras.Input(shape=input_shape)

    # NOTE: preprocessing is done by the data generator (rescale=1/255)
    # and by preprocess_roi at inference. Do NOT apply preprocess_input here
    # — doing so on already-normalised [0,1] inputs would produce [0,0.008],
    # which destroys MobileNetV2 features entirely (confirmed root cause of
    # model collapse to single-class prediction).
    x = base_model(inputs, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="ISL_MobileNetV2")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


def unfreeze_top_layers(model: tf.keras.Model, base_model: tf.keras.Model,
                        num_layers_to_unfreeze: int = 30,
                        learning_rate: float = 1e-5) -> tf.keras.Model:
    """
    Fine-tune by unfreezing the top N layers of the base model (Phase 2).

    Args:
        model                 : Full compiled model.
        base_model            : MobileNetV2 base model reference.
        num_layers_to_unfreeze: Number of top base layers to unfreeze.
        learning_rate         : Lower LR to prevent catastrophic forgetting.

    Returns:
        Re-compiled model with some base layers trainable.
    """
    base_model.trainable = True

    # Keep all layers frozen except the last `num_layers_to_unfreeze`
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"[Model] Unfroze top {num_layers_to_unfreeze} base layers for fine-tuning.")
    return model


def print_model_summary(model: tf.keras.Model) -> None:
    """Print model summary with trainable/non-trainable parameter counts."""
    model.summary()
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    non_trainable = sum(tf.size(w).numpy() for w in model.non_trainable_weights)
    print(f"\n[Model] Trainable params     : {trainable:,}")
    print(f"[Model] Non-trainable params : {non_trainable:,}")