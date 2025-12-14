import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import hashlib
import os
import json
from datetime import datetime

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Speech Pronunciation Analysis",
    page_icon="🎙",
    layout="wide"
)

# ===============================
# EXPERIMENT LOG CONFIG (MLflow-style)
# ===============================
EXPERIMENT_DIR = "experiments"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# ===============================
# CSS UI
# ===============================
st.markdown("""
<style>
.block-container { padding: 2.5rem; }
h1 { color: #1f2c56; }
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ===============================
# UTILITY FUNCTIONS
# ===============================
def display_spectrogram(y, sr, title):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 4))
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

def time_shift(y, sr):
    return np.roll(y, np.random.randint(-sr // 10, sr // 10))

def add_noise(y):
    return y + 0.005 * np.random.randn(len(y))

def pitch_shift(y, sr):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2)

def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def log_experiment(params, metrics):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = {
        "timestamp": timestamp,
        "params": params,
        "metrics": metrics
    }
    with open(f"{EXPERIMENT_DIR}/run_{timestamp}.json", "w") as f:
        json.dump(data, f, indent=4)

# ===============================
# SIDEBAR – CONTROL CENTER
# ===============================
st.sidebar.markdown("## 🎛 Control Panel")

menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🎧 Audio Analysis", "📊 Experiment Logs"]
)

st.sidebar.markdown("---")

model_type = st.sidebar.selectbox(
    "🤖 Model Architecture",
    ["CNN", "CRNN", "Transformer"]
)

learning_rate = st.sidebar.select_slider(
    "⚙ Learning Rate",
    options=[0.0001, 0.001, 0.01],
    value=0.001
)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
epoch = st.sidebar.slider("Epoch", 10, 100, 30)

augmentasi = st.sidebar.selectbox(
    "🔊 Audio Augmentation",
    ["Tanpa Augmentasi", "Time Shift", "Noise Addition", "Pitch Shift"]
)

# ===============================
# HOME PAGE
# ===============================
if menu == "🏠 Home":
    st.title("🎙 Speech Pronunciation Analysis")
    st.markdown("""
    <div class="card">
    <b>Application Capabilities:</b><br><br>
    ✅ Speech pronunciation analysis (*word: very*)<br>
    ✅ Mel-Spectrogram visualization<br>
    ✅ Deep Learning inference simulation<br>
    ✅ Experiment tracking (MLflow-style, file-based)<br><br>
    <i>Designed for academic research, thesis, and MLOps demonstration.</i>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# AUDIO ANALYSIS PAGE
# ===============================
elif menu == "🎧 Audio Analysis":
    st.title("🎧 Audio Analysis & Inference")

    uploaded_file = st.file_uploader("Upload WAV Audio", type=["wav"])

    if uploaded_file:
        audio_bytes = uploaded_file.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)

        if augmentasi == "Time Shift":
            y_aug = time_shift(y, sr)
        elif augmentasi == "Noise Addition":
            y_aug = add_noise(y)
        elif augmentasi == "Pitch Shift":
            y_aug = pitch_shift(y, sr)
        else:
            y_aug = y

        col1, col2 = st.columns(2)
        with col1:
            display_spectrogram(y, sr, "Original Audio")
        with col2:
            display_spectrogram(y_aug, sr, f"After {augmentasi}")

        if st.button("🚀 Run Inference"):
            confidence = float(np.random.uniform(0.75, 0.95))
            label = "Pengucapan Benar" if confidence > 0.5 else "Perlu Perbaikan"

            params = {
                "model": model_type,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epoch": epoch,
                "augmentation": augmentasi,
                "dataset_hash": get_file_hash(audio_bytes)
            }

            metrics = {
                "confidence": confidence,
                "loss": 1 - confidence
            }

            log_experiment(params, metrics)

            st.success(f"Prediksi: {label} ({confidence*100:.2f}%)")
            st.progress(confidence)

# ===============================
# EXPERIMENT LOG PAGE
# ===============================
elif menu == "📊 Experiment Logs":
    st.title("📊 Experiment History")

    files = sorted(os.listdir(EXPERIMENT_DIR), reverse=True)

    if not files:
        st.info("Belum ada eksperimen yang tercatat.")
    else:
        for f in files[:5]:
            with open(f"{EXPERIMENT_DIR}/{f}") as file:
                data = json.load(file)
                st.json(data)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("🚀 Streamlit • Audio AI • MLOps-ready (MLflow-style Logging)")
