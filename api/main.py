"""
main.py — FastAPI backend for the Student Engagement Detection system
---------------------------------------------------------------------

Endpoints:
    POST /predict        — upload an image, return emotion + confidence
    POST /upload-data    — upload zip or images, save to data/train/<label>
    POST /retrain        — trigger background model retraining
    GET  /health         — model status, uptime, last trained timestamp
    GET  /metrics        — model accuracy + class distribution

Run:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import io
import os
import sys
import json
import shutil
import zipfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Path setup ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR  = BASE_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(BASE_DIR / 'api'))

from prediction import predict_emotion, load_model, clear_model_cache, DEFAULT_MODEL_PATH
from retrain import trigger_retrain, read_status

TRAIN_DIR  = BASE_DIR / 'data' / 'train'
MODEL_PATH = BASE_DIR / 'models' / 'emotion_model.h5'

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title='Student Engagement Detection API',
    description='Facial emotion recognition using a CNN trained on FER+ data.',
    version='1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

_start_time = datetime.now(timezone.utc)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _model_exists() -> bool:
    return MODEL_PATH.exists()


def _last_trained() -> str | None:
    if not MODEL_PATH.exists():
        return None
    ts = os.path.getmtime(MODEL_PATH)
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _uptime_seconds() -> float:
    return (datetime.now(timezone.utc) - _start_time).total_seconds()


def _count_class_images(data_dir: Path) -> dict:
    counts = {}
    if not data_dir.exists():
        return counts
    for cls_dir in sorted(data_dir.iterdir()):
        if cls_dir.is_dir():
            imgs = [f for f in cls_dir.iterdir()
                    if f.suffix.lower() in ALLOWED_EXTENSIONS]
            counts[cls_dir.name] = len(imgs)
    return counts


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get('/health')
def health():
    """Return model status, uptime, and last trained timestamp."""
    retrain_status = read_status()
    return {
        'model_loaded':   _model_exists(),
        'model_path':     str(MODEL_PATH),
        'uptime_seconds': _uptime_seconds(),
        'last_trained':   _last_trained(),
        'retrain_status': retrain_status.get('status', 'idle'),
        'retrain_message': retrain_status.get('message', ''),
        'server_time':    datetime.now(timezone.utc).isoformat()
    }


@app.get('/metrics')
def metrics():
    """Return current model accuracy (from last retrain) and class distribution."""
    retrain_status = read_status()
    class_dist = _count_class_images(TRAIN_DIR)
    return {
        'accuracy':          retrain_status.get('accuracy'),
        'class_distribution': class_dist,
        'total_train_images': sum(class_dist.values()),
        'last_trained':      _last_trained(),
    }


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    """
    Accept an uploaded image, return emotion prediction + confidence.

    Returns
    -------
    JSON: {predicted_class, confidence, all_probabilities}
    """
    if not _model_exists():
        raise HTTPException(
            status_code=503,
            detail='Model not found. Train the model first.'
        )

    suffix = Path(file.filename or 'img.png').suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f'Unsupported file type: {suffix}. '
                   f'Allowed: {ALLOWED_EXTENSIONS}'
        )

    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result = predict_emotion(tmp_path, model_path=str(MODEL_PATH))
        return JSONResponse(content=result)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.post('/upload-data')
async def upload_data(
    files: list[UploadFile] = File(...),
    label: str = Form(default=None)
):
    """
    Accept bulk image uploads (individual files or a single zip).

    - If *label* is provided, all images are saved to data/train/<label>/
    - If a .zip is uploaded, the archive must contain sub-folders per class.
    """
    saved = []
    errors = []

    for upload in files:
        fname = upload.filename or 'upload'
        suffix = Path(fname).suffix.lower()
        contents = await upload.read()

        if suffix == '.zip':
            # Extract zip: expected structure <class>/<image.png>
            try:
                with zipfile.ZipFile(io.BytesIO(contents)) as zf:
                    for member in zf.namelist():
                        parts = Path(member).parts
                        if len(parts) < 2:
                            continue
                        cls_name, img_name = parts[0], parts[-1]
                        if Path(img_name).suffix.lower() not in ALLOWED_EXTENSIONS:
                            continue
                        dest_dir = TRAIN_DIR / cls_name
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest_file = dest_dir / img_name
                        with zf.open(member) as src, open(dest_file, 'wb') as dst:
                            dst.write(src.read())
                        saved.append(str(dest_file.relative_to(BASE_DIR)))
            except Exception as exc:
                errors.append(f'{fname}: {exc}')

        elif suffix in ALLOWED_EXTENSIONS:
            if not label:
                errors.append(f'{fname}: no label provided for single-image upload.')
                continue
            dest_dir = TRAIN_DIR / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / fname
            with open(dest_file, 'wb') as f:
                f.write(contents)
            saved.append(str(dest_file.relative_to(BASE_DIR)))

        else:
            errors.append(f'{fname}: unsupported type {suffix}')

    return {
        'saved_files': saved,
        'errors':      errors,
        'total_saved': len(saved)
    }


@app.post('/retrain')
def retrain(epochs: int = 30, batch_size: int = 64):
    """
    Trigger background model retraining using data in data/train/.

    The endpoint returns immediately; training runs in a daemon thread.
    Poll GET /health for progress.
    """
    if not TRAIN_DIR.exists() or not any(TRAIN_DIR.iterdir()):
        raise HTTPException(
            status_code=400,
            detail='No training data found in data/train/.'
        )
    result = trigger_retrain(
        train_dir=str(TRAIN_DIR),
        model_path=str(MODEL_PATH),
        epochs=epochs,
        batch_size=batch_size
    )
    if result.get('status') == 'running' and 'already' in result.get('detail', ''):
        raise HTTPException(status_code=409, detail=result['detail'])
    return result
