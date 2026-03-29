"""
prediction.py
-------------
Load the saved emotion model and run inference on a single image.

CLI usage:
    python src/prediction.py --image path/to/face.png
    python src/prediction.py --image path/to/face.png --model models/emotion_model.h5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from preprocessing import preprocess_single_image, IMG_SIZE

# ── Default paths ────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH   = Path(__file__).resolve().parent.parent / 'models' / 'emotion_model.h5'
CLASS_INDICES_PATH   = Path(__file__).resolve().parent.parent / 'models' / 'class_indices.json'

# Fallback label list (alphabetical, matches flow_from_directory ordering)
_FALLBACK_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def _load_labels(model_path: str) -> list[str]:
    """
    Load class names from the saved class_indices.json next to the model.
    Falls back to the alphabetically sorted default list if the file is absent.
    """
    indices_file = Path(model_path).parent / 'class_indices.json'
    if indices_file.exists():
        with open(indices_file) as f:
            idx_map = json.load(f)  # {'angry': 0, 'disgust': 1, …}
        return [cls for cls, _ in sorted(idx_map.items(), key=lambda x: x[1])]
    return _FALLBACK_LABELS


# ── Model loader (cached) ────────────────────────────────────────────────────

_model_cache: dict = {}


def load_model(model_path: str = str(DEFAULT_MODEL_PATH)) -> keras.Model:
    """
    Load the Keras model from *model_path*.

    The model is cached in memory so repeated calls within the same process
    do not re-read the file from disk.
    """
    if model_path not in _model_cache:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f'Model file not found: {model_path}\n'
                'Train the model first (run the notebook or src/model.py).'
            )
        _model_cache[model_path] = keras.models.load_model(model_path)
        print(f'[prediction] Model loaded from {model_path}')
    return _model_cache[model_path]


def clear_model_cache():
    """Force-reload the model on the next call (e.g. after retraining)."""
    _model_cache.clear()


# ── Inference ────────────────────────────────────────────────────────────────

def predict_emotion(image_path: str,
                    model_path: str = str(DEFAULT_MODEL_PATH),
                    img_size: int = IMG_SIZE
                    ) -> dict:
    """
    Predict the emotion for a single face image.

    Parameters
    ----------
    image_path : path to the input image (any format PIL can read)
    model_path : path to the saved .h5 model
    img_size   : resize target (must match training)

    Returns
    -------
    dict with keys:
        predicted_class  : str   e.g. 'Happy'
        confidence       : float e.g. 0.9231  (0–1)
        all_probabilities: dict  {class_name: probability, …}
    """
    model = load_model(model_path)

    # Determine class names from saved indices (or fallback)
    labels = _load_labels(model_path)
    num_classes = model.output_shape[-1]

    # Preprocess
    img_array = preprocess_single_image(image_path, size=img_size)

    # Run inference
    probabilities = model.predict(img_array, verbose=0)[0]
    predicted_idx = int(np.argmax(probabilities))
    predicted_class = labels[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    all_probs = {labels[i]: float(probabilities[i])
                 for i in range(num_classes)}

    return {
        'predicted_class':   predicted_class,
        'confidence':        confidence,
        'all_probabilities': all_probs
    }


# ── CLI entry point ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Predict emotion from a face image')
    p.add_argument('--image',      required=True,
                   help='Path to the input image')
    p.add_argument('--model',      default=str(DEFAULT_MODEL_PATH),
                   help='Path to the saved model (.h5)')
    p.add_argument('--img_size',   type=int, default=IMG_SIZE,
                   help='Image resize target (default: 48)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    result = predict_emotion(
        image_path=args.image,
        model_path=args.model,
        img_size=args.img_size
    )
    print(f"\nPredicted emotion : {result['predicted_class']}")
    print(f"Confidence        : {result['confidence']:.4f} ({result['confidence']*100:.1f}%)")
    print('\nAll class probabilities:')
    for cls, prob in sorted(result['all_probabilities'].items(),
                             key=lambda x: x[1], reverse=True):
        bar = '█' * int(prob * 30)
        print(f'  {cls:<12} {prob:.4f}  {bar}')
