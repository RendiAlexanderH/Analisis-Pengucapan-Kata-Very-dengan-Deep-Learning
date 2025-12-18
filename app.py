import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import os
import json
import hashlib
from datetime import datetime

import tensorflow as tf

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(page_title="Speech Pronunciation Analysis", layout="wide")

# ========================
# DIRS
# ========================
EXPERIMENT_DIR = "experiments"
MODEL_DIR = "models"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "autoencoder_model")

# ========================
# BASIC STYLE
# ========================
st.markdown("""
<style>
.block-container { padding: 2rem; }
h1 { color: #1f2c56; }
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ========================
# UTILITY FUNCTIONS
# ========================
def display_spectrogram(y, sr, title):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(8,4))
    img = librosa.display.specshow(
        mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax
    )
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)
    plt.close(fig)

def time_shift(y, sr):
    return np.roll(y, np.random.randint(-sr//10, sr//10))

def add_noise(y):
    return y + 0.005 * np.random.randn(len(y))

def pitch_shift(y, sr):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=2)

def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def prepare_autoencoder_input(y, sr, n_mels=128, duration=2.0):
    max_len = int(sr * duration)
    y = y[:max_len]
    y = np.pad(y, (0, max_len - len(y)))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

    mel_db = np.expand_dims(mel_db, axis=-1)
    mel_db = np.expand_dims(mel_db, axis=0)
    return mel_db

def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Please train the model first.")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

# ========================
# SIDEBAR
# ========================
st.sidebar.title("Control Panel")
menu = st.sidebar.radio("Navigation", ["Home", "Audio Analysis", "Experiment Logs"])
augmentation = st.sidebar.selectbox(
    "Audio Augmentation",
    ["None", "Time Shift", "Noise Addition", "Pitch Shift"]
)

# ========================
# HOME PAGE
# ========================
if menu == "Home":
    st.title("Speech Pronunciation Analysis")
    st.markdown("""
    <div class="card">
    <b>Autoencoder-based pronunciation analysis</b> for the word <b>very</b>.
    <ul>
        <li>Single-class training</li>
        <li>Reconstruction loss as pronunciation score</li>
        <li>Lower loss = better pronunciation</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ========================
# AUDIO ANALYSIS
# ========================
elif menu == "Audio Analysis":
    st.title("Audio Analysis and Pronunciation Scoring")
    uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

    if uploaded_file:
        audio_bytes = uploaded_file.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)

        y_aug = y
        if augmentation == "Time Shift":
            y_aug = time_shift(y, sr)
        elif augmentation == "Noise Addition":
            y_aug = add_noise(y)
        elif augmentation == "Pitch Shift":
            y_aug = pitch_shift(y, sr)

        col1, col2 = st.columns(2)
        with col1:
            display_spectrogram(y, sr, "Original Audio")
        with col2:
            display_spectrogram(y_aug, sr, "Processed Audio")

        if st.button("Run Pronunciation Analysis"):
            model = load_model()
            if model:
                X = prepare_autoencoder_input(y_aug, sr)
                X_hat = model.predict(X, verbose=0)

                loss = float(np.mean((X - X_hat) ** 2))
                score = max(0.0, 1.0 - loss)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_data = {
                    "timestamp": timestamp,
                    "metrics": {
                        "reconstruction_loss": loss,
                        "pronunciation_score": score
                    },
                    "file_hash": get_file_hash(audio_bytes)
                }

                filename = f"{EXPERIMENT_DIR}/run_{timestamp}.json"
                with open(filename, "w") as f:
                    json.dump(log_data, f, indent=4)

                st.success("Analysis completed")
                st.metric("Reconstruction Loss", f"{loss:.6f}")
                st.metric("Pronunciation Score", f"{score:.2f}")

                st.download_button(
                    "Download Experiment Log",
                    data=open(filename, "rb"),
                    file_name=os.path.basename(filename),
                    mime="application/json"
                )

# ========================
# EXPERIMENT LOGS
# ========================
elif menu == "Experiment Logs":
    st.title("Experiment History")
    files = sorted(os.listdir(EXPERIMENT_DIR), reverse=True)
    if not files:
        st.info("No experiments yet")
    else:
        for file in files:
            with open(f"{EXPERIMENT_DIR}/{file}") as f:
                st.json(json.load(f))

st.markdown("---")
st.caption("Autoencoder-based Speech Pronunciation Analysis")
