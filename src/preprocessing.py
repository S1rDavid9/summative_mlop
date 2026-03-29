"""
preprocessing.py
----------------
Utilities for loading, resizing, normalising, and augmenting images
for the FER+ student-engagement emotion-recognition pipeline.

Works for both a single image file and an entire folder of class-labelled
sub-directories (train/test structure expected by Keras generators).
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Constants ───────────────────────────────────────────────────────────────
IMG_SIZE    = 48          # FER+ images are 48×48
NUM_CLASSES = 7
# Sorted alphabetically to match flow_from_directory's class_indices ordering
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# ── Single-image helpers ────────────────────────────────────────────────────

def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk and return as a uint8 numpy array (H×W)."""
    img = Image.open(image_path).convert('L')          # grayscale
    return np.array(img)


def resize_image(image: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Resize a 2-D grayscale array to (size × size)."""
    pil_img = Image.fromarray(image.astype('uint8'))
    pil_img = pil_img.resize((size, size), Image.LANCZOS)
    return np.array(pil_img)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1]."""
    return image.astype('float32') / 255.0


def preprocess_single_image(image_path: str,
                             size: int = IMG_SIZE) -> np.ndarray:
    """
    Full pipeline for a single image file.

    Returns
    -------
    np.ndarray  shape (1, size, size, 1) — ready for model.predict()
    """
    img = load_image(image_path)
    img = resize_image(img, size)
    img = normalize_image(img)
    img = np.expand_dims(img, axis=-1)   # add channel dim  → (H, W, 1)
    img = np.expand_dims(img, axis=0)    # add batch dim    → (1, H, W, 1)
    return img


# ── Batch / folder helpers ──────────────────────────────────────────────────

def load_images_from_folder(folder_path: str,
                             size: int = IMG_SIZE
                             ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load every image from a class-labelled folder structure:

        folder_path/
            ClassName1/img1.png …
            ClassName2/img1.png …

    Returns
    -------
    images  : np.ndarray  shape (N, size, size, 1), float32 normalised
    labels  : np.ndarray  shape (N,), integer class indices
    classes : list[str]   sorted class names
    """
    classes = sorted([
        d for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d))
    ])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    images, labels = [], []
    for cls in classes:
        cls_dir = Path(folder_path) / cls
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            try:
                img = load_image(str(cls_dir / fname))
                img = resize_image(img, size)
                img = normalize_image(img)
                img = np.expand_dims(img, axis=-1)
                images.append(img)
                labels.append(class_to_idx[cls])
            except Exception as e:
                print(f'[Warning] Could not load {cls_dir / fname}: {e}')

    return np.array(images, dtype='float32'), np.array(labels), classes


# ── Keras data generators ───────────────────────────────────────────────────

def get_train_generator(train_dir: str,
                         batch_size: int = 64,
                         img_size: int = IMG_SIZE,
                         validation_split: float = 0.15):
    """Return (train_generator, val_generator) with augmentation."""
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    val_gen = datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    return train_gen, val_gen


def get_test_generator(test_dir: str,
                        batch_size: int = 64,
                        img_size: int = IMG_SIZE):
    """Return a test generator (no augmentation)."""
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    return datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )


# ── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        arr = preprocess_single_image(path)
        print(f'Preprocessed shape: {arr.shape}, dtype: {arr.dtype}')
        print(f'Min: {arr.min():.4f}  Max: {arr.max():.4f}')
    else:
        print('Usage: python preprocessing.py <image_path>')
