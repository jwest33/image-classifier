#!/usr/bin/env python3
"""
train_classifier.py

Train a CNN classifier to identify cat breeds based on cropped cat images stored as:

cropped_cats/
    abyssinian/
        xxx.jpg
        yyy.jpg
    bengal/
        ...

The script recursively scans each subdirectory for JPEG images and uses the name of the
subdirectory as the class label.

Usage:
    python train_classifier.py --data_dir cropped_cats --output_model cat_breed_classifier.keras
"""

import argparse
from pathlib import Path
import tensorflow as tf

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = (128, 128)  # Image size fed to the network
BATCH_SIZE = 32        # Batch size for training/validation
VALIDATION_SPLIT = 0.2 # 80 % train, 20 % validation
SEED = 123             # For deterministic dataset splits

# -----------------------------
# Dataset Preparation
# -----------------------------

def build_datasets(data_dir: Path):
    """Create training & validation datasets from a directory structure.

    The helper uses ``tf.keras.utils.image_dataset_from_directory`` which **recursively** walks
    the supplied directory, infers the class label from each file’s **parent folder name**, and
    returns a `tf.data.Dataset` ready for model training.
    """

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",            # derive from folder names
        label_mode="categorical",     # one-hot encoded targets
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    autotune = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(buffer_size=autotune),
        val_ds.prefetch(buffer_size=autotune),
        train_ds.class_names,  # ["abyssinian", "bengal", ...]
    )

# -----------------------------
# Training Routine
# -----------------------------

def train_classifier(data_dir: str | Path, output_model_path: str, epochs: int = 10):
    data_dir = Path(data_dir)
    train_ds, val_ds, class_names = build_datasets(data_dir)
    num_classes = len(class_names)

    # Base model: MobileNetV2 (transfer learning)
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
        pooling="avg",
    )
    base_model.trainable = False  # Freeze convolutional backbone

    # Classification head
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(base_model.output)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save(output_model_path)

    print(f"Model trained on {num_classes} classes: {', '.join(class_names)}")
    print(f"Saved model to {output_model_path}")

# -----------------------------
# CLI Entry-point
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a cat-breed classifier from cropped images.")
    parser.add_argument("--data_dir", default="cropped_cats", help="Directory containing breed subfolders")
    parser.add_argument("--output_model", default="cat_breed_classifier.keras", help="Output path for the saved model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    train_classifier(args.data_dir, args.output_model, args.epochs)
