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

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Speech Pronunciation Analysis",
    layout="wide"
)

# =====================================================
# EXPERIMENT LOG CONFIGURATION
# =====================================================
EXPERIMENT_DIR = "experiments"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# =====================================================
# BASIC STYLE
# =====================================================
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

# =====================================================
# UTILITY FUNCTIONS
# =====================================================
def display_spectrogram(y, sr, title):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 4))

    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        ax=ax
    )

    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    st.pyplot(fig)
    plt.close(fig)


def time_shift(y, sr):
    shift = np.random.randint(-sr // 10, sr // 10)
    return np.roll(y, shift)

def add_noise(y):
    noise = np.random.randn(len(y))
    return y + 0.005 * noise

def pitch_shift(y, sr):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2)

def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def log_experiment(params, metrics):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = {
        "timestamp": timestamp,
        "parameters": params,
        "metrics": metrics
    }
    filename = f"{EXPERIMENT_DIR}/run_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    return filename

def plot_metrics(confidence, loss):
    fig, ax = plt.subplots()
    ax.bar(["Confidence", "Loss"], [confidence, loss])
    ax.set_ylim(0, 1)
    ax.set_title("Inference Metrics")
    st.pyplot(fig)
    plt.close(fig)

# =====================================================
# SIDEBAR CONTROL PANEL
# =====================================================
st.sidebar.title("Control Panel")

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Audio Analysis", "Experiment Logs"]
)

st.sidebar.markdown("---")

model_type = st.sidebar.selectbox(
    "Model Architecture",
    ["CNN", "CRNN", "Transformer"]
)

learning_rate = st.sidebar.selectbox(
    "Learning Rate",
    [0.0001, 0.001, 0.01],
    index=1
)

batch_size = st.sidebar.selectbox(
    "Batch Size",
    [16, 32, 64],
    index=1
)

epoch = st.sidebar.slider(
    "Epoch",
    10, 100, 30
)

augmentation = st.sidebar.selectbox(
    "Audio Augmentation",
    ["None", "Time Shift", "Noise Addition", "Pitch Shift"]
)

# =====================================================
# HOME PAGE
# =====================================================
if menu == "Home":
    st.title("Speech Pronunciation Analysis")
    st.markdown("""
    <div class="card">
    This application analyzes the pronunciation of the word <b>very</b> using
    Mel-Spectrogram visualization and simulated deep learning inference.
    <br><br>
    Features:
    <ul>
        <li>Audio visualization</li>
        <li>Data augmentation</li>
        <li>Inference simulation</li>
        <li>Experiment tracking with downloadable logs</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# AUDIO ANALYSIS PAGE
# =====================================================
elif menu == "Audio Analysis":
    st.title("Audio Analysis and Inference")

    uploaded_file = st.file_uploader("Upload WAV audio file", type=["wav"])

    if uploaded_file:
        audio_bytes = uploaded_file.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)

        if augmentation == "Time Shift":
            y_aug = time_shift(y, sr)
        elif augmentation == "Noise Addition":
            y_aug = add_noise(y)
        elif augmentation == "Pitch Shift":
            y_aug = pitch_shift(y, sr)
        else:
            y_aug = y

        col1, col2 = st.columns(2)
        with col1:
            display_spectrogram(y, sr, "Original Audio")
        with col2:
            display_spectrogram(y_aug, sr, "Augmented Audio")

        if st.button("Run Inference"):
            confidence = float(np.random.uniform(0.75, 0.95))
            loss = 1 - confidence

            params = {
                "model": model_type,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epoch": epoch,
                "augmentation": augmentation,
                "dataset_hash": get_file_hash(audio_bytes)
            }

            metrics = {
                "confidence": confidence,
                "loss": loss
            }

            log_file = log_experiment(params, metrics)

            st.success("Inference completed successfully")

            st.write(f"Confidence: {confidence:.2f}")
            st.write(f"Loss: {loss:.2f}")

            plot_metrics(confidence, loss)

            with open(log_file, "rb") as f:
                st.download_button(
                    label="Download Experiment Log",
                    data=f,
                    file_name=os.path.basename(log_file),
                    mime="application/json"
                )

# =====================================================
# EXPERIMENT LOG PAGE
# =====================================================
elif menu == "Experiment Logs":
    st.title("Experiment History")

    files = sorted(os.listdir(EXPERIMENT_DIR), reverse=True)

    if not files:
        st.info("No experiments recorded yet.")
    else:
        for file in files:
            with open(f"{EXPERIMENT_DIR}/{file}") as f:
                st.json(json.load(f))

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("Speech Pronunciation Analysis Application")
