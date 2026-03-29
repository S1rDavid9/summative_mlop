"""
retrain.py
----------
Background retraining logic triggered by POST /retrain.

The retraining runs in a separate thread so the API stays responsive.
Progress is written to a JSON status file that /health and /metrics can read.
"""

import os
import json
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
TRAIN_DIR    = BASE_DIR / 'data' / 'train'
MODEL_PATH   = BASE_DIR / 'models' / 'emotion_model.h5'
STATUS_FILE  = BASE_DIR / 'models' / 'retrain_status.json'

# ── Status helpers ───────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(status: str, message: str, accuracy: float = None):
    payload = {
        'status':        status,       # 'idle' | 'running' | 'done' | 'error'
        'message':       message,
        'last_updated':  _now_iso(),
    }
    if accuracy is not None:
        payload['accuracy'] = round(accuracy, 4)
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, 'w') as f:
        json.dump(payload, f, indent=2)


def read_status() -> dict:
    """Return the current retrain status dict (safe for any caller)."""
    if not STATUS_FILE.exists():
        return {'status': 'idle', 'message': 'No retraining has been run yet.'}
    with open(STATUS_FILE) as f:
        return json.load(f)


# ── Retraining worker ────────────────────────────────────────────────────────

def _retrain_worker(train_dir: str,
                    model_path: str,
                    epochs: int,
                    batch_size: int):
    """Runs in a daemon thread. Trains the model and replaces the .h5 file."""
    try:
        _write_status('running', f'Retraining started — {_now_iso()}')

        # Lazy-import to avoid loading TF at startup if not needed
        import numpy as np
        import tensorflow as tf
        from tensorflow import keras

        # Add src/ to path so we can import our modules
        import sys
        src_dir = str(Path(__file__).resolve().parent.parent / 'src')
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        from preprocessing import get_train_generator
        from model import build_emotion_cnn

        np.random.seed(42)
        tf.random.set_seed(42)

        # ── Data generators ─────────────────────────────────────────────
        _write_status('running', 'Loading data generators…')
        train_gen, val_gen = get_train_generator(
            train_dir, batch_size=batch_size
        )
        num_classes = len(train_gen.class_indices)
        print(f'[retrain] Classes: {train_gen.class_indices}')

        # ── Build model ─────────────────────────────────────────────────
        _write_status('running', 'Building model…')
        model = build_emotion_cnn(num_classes=num_classes)

        # ── Callbacks ───────────────────────────────────────────────────
        tmp_path = str(model_path) + '.tmp.h5'
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=tmp_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=0
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=0
            ),
            _StatusCallback(_write_status)
        ]

        # ── Train ───────────────────────────────────────────────────────
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        # ── Swap model files ─────────────────────────────────────────────
        best_val_acc = float(max(history.history.get('val_accuracy', [0])))
        if Path(tmp_path).exists():
            import shutil
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(tmp_path, model_path)
            print(f'[retrain] Model saved to {model_path}')

        # Invalidate prediction cache so next request uses the new model
        try:
            sys.path.insert(0, src_dir)
            import prediction
            prediction.clear_model_cache()
        except Exception:
            pass

        _write_status('done',
                      f'Retraining complete. Best val accuracy: {best_val_acc:.4f}',
                      accuracy=best_val_acc)
        print('[retrain] Done.')

    except Exception as exc:
        tb = traceback.format_exc()
        print(f'[retrain] ERROR:\n{tb}')
        _write_status('error', f'Retraining failed: {exc}')


class _StatusCallback(keras.callbacks.Callback):
    """Writes epoch progress to the status file during training."""
    def __init__(self, write_fn):
        super().__init__()
        self._write = write_fn

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = (
            f"Epoch {epoch + 1} — "
            f"loss: {logs.get('loss', 0):.4f}  "
            f"acc: {logs.get('accuracy', 0):.4f}  "
            f"val_loss: {logs.get('val_loss', 0):.4f}  "
            f"val_acc: {logs.get('val_accuracy', 0):.4f}"
        )
        self._write('running', msg)


# ── Public API ───────────────────────────────────────────────────────────────

_retrain_lock = threading.Lock()


def trigger_retrain(train_dir: str = str(TRAIN_DIR),
                    model_path: str = str(MODEL_PATH),
                    epochs: int = 30,
                    batch_size: int = 64) -> dict:
    """
    Start a background retraining thread (non-blocking).

    Returns immediately with the current status.
    Only one retraining run is allowed at a time.
    """
    current = read_status()
    if current.get('status') == 'running':
        return {'detail': 'Retraining already in progress.',
                'status': 'running'}

    with _retrain_lock:
        _write_status('running', 'Initialising retraining…')
        t = threading.Thread(
            target=_retrain_worker,
            args=(train_dir, model_path, epochs, batch_size),
            daemon=True
        )
        t.start()

    return {'detail': 'Retraining started in background.', 'status': 'running'}
