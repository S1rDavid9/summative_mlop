"""
model.py
--------
CNN architecture definition, training function, and model persistence
for the FER+ student-engagement emotion-recognition pipeline.

CLI usage:
    python src/model.py \
        --train_dir  data/train \
        --test_dir   data/test  \
        --model_path models/emotion_model.h5 \
        --epochs     50 \
        --batch_size 64
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from preprocessing import get_train_generator, get_test_generator, IMG_SIZE

# ── Reproducibility ─────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

NUM_CLASSES = 7  # FER+ dataset: angry, disgust, fear, happy, neutral, sad, surprise


# ── Architecture ─────────────────────────────────────────────────────────────

def build_emotion_cnn(input_shape: tuple = (IMG_SIZE, IMG_SIZE, 1),
                      num_classes: int = NUM_CLASSES) -> keras.Model:
    """
    Build a CNN with 3 convolutional blocks + dense classifier head.

    Architecture:
        Block 1: Conv(64)×2 → BN → MaxPool → Dropout(0.25)
        Block 2: Conv(128)×2 → BN → MaxPool → Dropout(0.25)
        Block 3: Conv(256)×2 → BN → MaxPool → Dropout(0.40)
        Head:    Flatten → Dense(512) → BN → Dropout(0.50)
                          → Dense(256) → Dropout(0.30)
                          → Dense(num_classes, softmax)
    """
    model = keras.Sequential([
        # ── Block 1 ────────────────────────────────────────────────────
        layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # ── Block 2 ────────────────────────────────────────────────────
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # ── Block 3 ────────────────────────────────────────────────────
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.40),

        # ── Dense head ─────────────────────────────────────────────────
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.50),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.30),
        layers.Dense(num_classes, activation='softmax')
    ], name='EmotionCNN')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── Training function ────────────────────────────────────────────────────────

def train_model(train_dir: str,
                test_dir: str,
                model_path: str = 'models/emotion_model.h5',
                epochs: int = 50,
                batch_size: int = 64,
                img_size: int = IMG_SIZE,
                validation_split: float = 0.15):
    """
    Train the emotion CNN and save the best checkpoint.

    Parameters
    ----------
    train_dir        : path to training images (class sub-folders)
    test_dir         : path to test images
    model_path       : where to save the best model (.h5)
    epochs           : max training epochs
    batch_size       : mini-batch size
    img_size         : spatial input resolution (default 48)
    validation_split : fraction of training data used for validation

    Returns
    -------
    history : keras History object
    model   : trained keras.Model (best weights loaded)
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────────
    train_gen, val_gen = get_train_generator(
        train_dir, batch_size=batch_size,
        img_size=img_size, validation_split=validation_split
    )
    num_classes = len(train_gen.class_indices)
    print(f'Classes ({num_classes}): {train_gen.class_indices}')

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_emotion_cnn(
        input_shape=(img_size, img_size, 1),
        num_classes=num_classes
    )

    # ── Callbacks ───────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # ── Train ───────────────────────────────────────────────────────────────
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Save class indices so prediction.py can recover the correct label mapping
    indices_path = Path(model_path).parent / 'class_indices.json'
    with open(indices_path, 'w') as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print(f'Class indices saved to {indices_path}')

    # Reload best weights (EarlyStopping restores them, but be explicit)
    if Path(model_path).exists():
        model = keras.models.load_model(model_path)
        print(f'Best model loaded from {model_path}')

    return history, model


# ── CLI entry point ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train EmotionCNN on FER+ data')
    p.add_argument('--train_dir',  default='data/train',
                   help='Path to training images')
    p.add_argument('--test_dir',   default='data/test',
                   help='Path to test images')
    p.add_argument('--model_path', default='models/emotion_model.h5',
                   help='Output model path (.h5)')
    p.add_argument('--epochs',     type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--img_size',   type=int, default=IMG_SIZE)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    history, model = train_model(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    print('Training complete.')
    print(f'Final val accuracy: {max(history.history["val_accuracy"]):.4f}')
