"""
app.py — Streamlit dashboard for the Student Engagement Detection system
------------------------------------------------------------------------

Pages:
    1. Monitor       — model status, uptime, last trained timestamp
    2. Visualisations — class distribution, sample images, confidence scores
    3. Actions       — predict, bulk upload, trigger retraining

Run:
    streamlit run ui/app.py
"""

import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image

# ── Config ───────────────────────────────────────────────────────────────────
API_BASE    = os.getenv('API_BASE_URL', 'https://akach1-student-engagement-api.hf.space')
BASE_DIR    = Path(__file__).resolve().parent.parent
TRAIN_DIR   = BASE_DIR / 'data' / 'train'
POLL_INTERVAL = 3   # seconds between retraining status polls

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

ENGAGEMENT_MAP = {
    'happy':    ('Engaged',    'green'),
    'neutral':  ('Neutral',    'blue'),
    'surprise': ('Attentive',  'orange'),
    'sad':      ('Disengaged', 'red'),
    'angry':    ('Disengaged', 'red'),
    'fear':     ('Anxious',    'purple'),
    'disgust':  ('Disengaged', 'red'),
}

st.set_page_config(
    page_title='Student Engagement Detection',
    layout='wide'
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def api_get(endpoint: str) -> dict | None:
    try:
        r = requests.get(f'{API_BASE}{endpoint}', timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(endpoint: str, **kwargs) -> dict | None:
    try:
        r = requests.post(f'{API_BASE}{endpoint}', timeout=60, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {'error': str(exc)}


def format_uptime(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f'{h}h {m}m {s}s'


def load_class_distribution() -> dict:
    """Read class distribution from local disk (faster than API call)."""
    counts = {}
    if TRAIN_DIR.exists():
        for cls_dir in sorted(TRAIN_DIR.iterdir()):
            if cls_dir.is_dir():
                n = len([f for f in cls_dir.iterdir()
                         if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
                counts[cls_dir.name] = n
    return counts


# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title('Student Engagement Detection')
page = st.sidebar.radio(
    'Navigate',
    ['Monitor', 'Visualisations', 'Actions'],
    index=0
)
st.sidebar.markdown('---')
st.sidebar.caption(f'API: `{API_BASE}`')

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Monitor
# ═════════════════════════════════════════════════════════════════════════════
if page == 'Monitor':
    st.title('System Monitor')
    st.markdown('Live overview of the emotion-recognition model and API.')

    health = api_get('/health')
    metrics = api_get('/metrics')

    col1, col2, col3, col4 = st.columns(4)

    if health:
        col1.metric('API Status', 'Online')
        col2.metric('Model Loaded', 'Yes' if health.get('model_loaded') else 'No')
        col3.metric('Uptime', format_uptime(health.get('uptime_seconds', 0)))
        last_trained = health.get('last_trained') or 'Never'
        col4.metric('Last Trained', last_trained[:19].replace('T', ' ') if last_trained != 'Never' else 'Never')
    else:
        st.error('Cannot reach the API. Make sure FastAPI is running at ' + API_BASE)
        col1.metric('API Status', 'Offline')
        col2.metric('Model Loaded', '—')
        col3.metric('Uptime', '—')
        col4.metric('Last Trained', '—')

    st.markdown('---')

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader('Model Accuracy')
        if metrics and metrics.get('accuracy') is not None:
            acc = metrics['accuracy']
            st.metric('Val Accuracy (last retrain)', f'{acc*100:.2f}%')
            st.progress(acc)
        else:
            st.info('Accuracy will appear here after the first retraining run.')

    with col_b:
        st.subheader('Retraining Status')
        if health:
            rs = health.get('retrain_status', 'idle')
            rm = health.get('retrain_message', '')
            st.write(f'**Status:** {rs.upper()}')
            if rm:
                st.caption(rm)
        else:
            st.write('API offline.')

    st.markdown('---')
    st.subheader('Training Data Overview')
    dist = load_class_distribution()
    if dist:
        df = pd.DataFrame(list(dist.items()), columns=['Emotion', 'Images'])
        df = df.sort_values('Images', ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f'Total training images: **{sum(dist.values())}**')
    else:
        st.info('No training data found in data/train/.')


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Visualisations
# ═════════════════════════════════════════════════════════════════════════════
elif page == 'Visualisations':
    st.title('Data Visualisations')

    # ── 2a. Class distribution ────────────────────────────────────────────
    st.subheader('Class Distribution (Training Set)')
    dist = load_class_distribution()
    if dist:
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = sns.color_palette('Set2', len(dist))
        bars = ax.bar(dist.keys(), dist.values(), color=colors, edgecolor='white')
        ax.bar_label(bars)
        ax.set_xlabel('Emotion')
        ax.set_ylabel('Number of Images')
        ax.set_title('Training Set — Emotion Class Distribution')
        ax.tick_params(axis='x', rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        total = sum(dist.values())
        st.markdown(
            '**Insight:** The FER+ dataset is naturally imbalanced. '
            f'`Happy` and `Neutral` dominate with {dist.get("Happy", 0) + dist.get("Neutral", 0)} '
            f'combined images ({100*(dist.get("Happy",0)+dist.get("Neutral",0))/max(total,1):.1f}% of total), '
            'while `Disgust` and `Fear` are minority classes. '
            'This imbalance can bias the model toward predicting majority classes — '
            'a class-weighted loss or oversampling strategy can mitigate this.'
        )
    else:
        st.info('No training data found.')

    st.markdown('---')

    # ── 2b. Sample images per class ──────────────────────────────────────
    st.subheader('Sample Images per Emotion Class')
    if TRAIN_DIR.exists():
        classes = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
        cols = st.columns(min(len(classes), 4))
        for i, cls_dir in enumerate(classes):
            imgs = sorted([f for f in cls_dir.iterdir()
                           if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
            if imgs:
                img = Image.open(imgs[0]).convert('L').resize((96, 96))
                with cols[i % 4]:
                    st.image(img, caption=cls_dir.name, use_column_width=True)
    else:
        st.info('No images found in data/train/.')

    st.markdown('---')

    # ── 2c. Confidence score distribution (session predictions) ──────────
    st.subheader('Confidence Score Distribution (this session)')
    if 'prediction_confidences' in st.session_state and st.session_state.prediction_confidences:
        confs = st.session_state.prediction_confidences
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(confs, bins=20, color='steelblue', edgecolor='white')
        ax.axvline(np.mean(confs), color='coral', linestyle='--',
                   label=f'Mean: {np.mean(confs):.2f}')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Prediction Confidences')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info('Make some predictions on the Actions page to see the confidence distribution here.')

    st.markdown('---')

    # ── 2d. Class interpretation ──────────────────────────────────────────
    st.subheader('Engagement Interpretation of Emotion Classes')
    interp_data = {
        'Emotion':    list(ENGAGEMENT_MAP.keys()),
        'Engagement': [v[0] for v in ENGAGEMENT_MAP.values()],
        'Notes': [
            'Positive engagement; student is enjoying the material.',
            'Strong negative reaction; content may be unpleasant.',
            'Anxiety or uncertainty; reassurance may be needed.',
            'High interest in the current moment; leverage this peak.',
            'Attentive but passive; could be processing information.',
            'Low engagement; may be experiencing difficulty.',
            'Student is frustrated or confused — intervention may help.',
        ]
    }
    df_interp = pd.DataFrame(interp_data)
    st.dataframe(df_interp, use_container_width=True, hide_index=True)

    st.markdown("""
**Key Insights:**

1. **Happy & Surprise** map to high engagement — when the model predicts these,
   the student is likely actively participating. Teachers can use this as a
   positive signal to continue their current approach.

2. **Sad & Angry** both map to *Disengaged* — these are early-warning signals.
   A sustained prediction of these emotions across a session warrants
   a check-in with the student.

3. **Neutral** is the most ambiguous class. It represents passive attention
   and should be interpreted alongside session context (lecture vs. activity).
   It is also the largest class in FER+, so the model will predict it most often.
""")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Actions
# ═════════════════════════════════════════════════════════════════════════════
elif page == 'Actions':
    st.title('Actions')

    # ── 3a. Single image prediction ───────────────────────────────────────
    st.subheader('1. Predict Emotion from Image')
    uploaded_file = st.file_uploader(
        'Upload a face image', type=['png', 'jpg', 'jpeg', 'bmp', 'webp'],
        key='predict_upload'
    )
    if uploaded_file:
        col_img, col_res = st.columns([1, 2])
        with col_img:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption='Uploaded image', use_column_width=True)

        with col_res:
            with st.spinner('Running prediction…'):
                uploaded_file.seek(0)
                resp = api_post('/predict',
                                files={'file': (uploaded_file.name,
                                                uploaded_file.read(),
                                                'image/png')})
            if resp and 'error' not in resp:
                emotion   = resp['predicted_class']
                conf      = resp['confidence']
                all_probs = resp.get('all_probabilities', {})
                engagement, eng_color = ENGAGEMENT_MAP.get(emotion, ('Unknown', 'gray'))

                st.success(f'**Predicted Emotion: {emotion}**')
                st.markdown(f'Engagement level: **:{eng_color}[{engagement}]**')
                st.metric('Confidence', f'{conf*100:.1f}%')

                # Confidence bars for all classes
                if all_probs:
                    probs_df = pd.DataFrame(
                        sorted(all_probs.items(), key=lambda x: x[1], reverse=True),
                        columns=['Emotion', 'Probability']
                    )
                    fig, ax = plt.subplots(figsize=(6, 3))
                    colors = ['#2ecc71' if e == emotion else '#95a5a6'
                              for e in probs_df['Emotion']]
                    ax.barh(probs_df['Emotion'], probs_df['Probability'], color=colors)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Probability')
                    ax.set_title('Class Probabilities')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                # Store confidence for visualisations page
                if 'prediction_confidences' not in st.session_state:
                    st.session_state.prediction_confidences = []
                st.session_state.prediction_confidences.append(conf)
            else:
                err = resp.get('error', 'Unknown error') if resp else 'API unreachable'
                st.error(f'Prediction failed: {err}')

    st.markdown('---')

    # ── 3b. Bulk upload ───────────────────────────────────────────────────
    st.subheader('2. Bulk Upload Training Images')
    st.markdown(
        'Upload a **zip file** (must contain class sub-folders) '
        'or **multiple images** with a label.'
    )
    upload_mode = st.radio('Upload mode', ['Zip file', 'Images with label'], horizontal=True)

    if upload_mode == 'Zip file':
        zip_file = st.file_uploader('Upload zip', type=['zip'], key='bulk_zip')
        if zip_file and st.button('Upload Zip'):
            with st.spinner('Uploading…'):
                resp = api_post('/upload-data',
                                files=[('files', (zip_file.name,
                                                  zip_file.read(),
                                                  'application/zip'))])
            if resp and 'error' not in resp:
                st.success(f'Saved {resp["total_saved"]} images.')
                if resp.get('errors'):
                    st.warning('Errors:\n' + '\n'.join(resp['errors']))
            else:
                st.error(str(resp))

    else:
        label = st.text_input('Emotion label (e.g. Happy)')
        img_files = st.file_uploader('Upload images', type=['png', 'jpg', 'jpeg'],
                                      accept_multiple_files=True, key='bulk_imgs')
        if img_files and label and st.button('Upload Images'):
            with st.spinner('Uploading…'):
                files_payload = [
                    ('files', (f.name, f.read(), 'image/png'))
                    for f in img_files
                ]
                resp = api_post('/upload-data',
                                files=files_payload,
                                data={'label': label})
            if resp and 'error' not in resp:
                st.success(f'Saved {resp["total_saved"]} images to data/train/{label}/.')
            else:
                st.error(str(resp))

    st.markdown('---')

    # ── 3c. Trigger retraining ────────────────────────────────────────────
    st.subheader('3. Trigger Model Retraining')
    col_e, col_b = st.columns(2)
    epochs = col_e.number_input('Max epochs', min_value=1, max_value=200, value=30)
    batch  = col_b.selectbox('Batch size', [16, 32, 64, 128], index=2)

    if st.button('Start Retraining', type='primary'):
        with st.spinner('Sending retraining request…'):
            resp = api_post(f'/retrain?epochs={epochs}&batch_size={batch}')
        if resp and 'error' not in resp:
            st.success(resp.get('detail', 'Retraining triggered.'))
        else:
            err = resp.get('error', str(resp)) if resp else 'API unreachable'
            st.error(f'Failed to trigger retraining: {err}')

    # Live status polling
    health = api_get('/health')
    if health:
        rs = health.get('retrain_status', 'idle')
        rm = health.get('retrain_message', '')
        st.info(f'**Retrain status:** {rs.upper()} — {rm}')

        if rs == 'running':
            if st.button('Refresh status'):
                st.rerun()
    else:
        st.warning('Cannot reach API.')
