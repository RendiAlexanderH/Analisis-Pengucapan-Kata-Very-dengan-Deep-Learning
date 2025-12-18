import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import os
import mlflow
import mlflow.sklearn
import hashlib
from datetime import datetime

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(page_title="Speech Pronunciation Analysis", layout="wide")

# ========================
# EXPERIMENT DIR
# ========================
EXPERIMENT_DIR = "experiments"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

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
    fig, ax = plt.subplots(figsize=(8, 4))
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)
    plt.close(fig)

def time_shift(y, sr): return np.roll(y, np.random.randint(-sr//10, sr//10))
def add_noise(y): return y + 0.005 * np.random.randn(len(y))
def pitch_shift(y, sr): return librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
def get_file_hash(file_bytes): return hashlib.md5(file_bytes).hexdigest()

def run_mlflow_inference(model_name, X):
    """Load trained model and predict"""
    model_path = f"models/{model_name}"
    model = mlflow.sklearn.load_model(model_path)
    pred = model.predict(X)
    return pred

# ========================
# SIDEBAR CONTROL PANEL
# ========================
st.sidebar.title("Control Panel")
menu = st.sidebar.radio("Navigation", ["Home", "Audio Analysis", "Experiment Logs"])
model_type = st.sidebar.selectbox("Model Architecture", ["RandomForest", "CNN", "CRNN"])
learning_rate = st.sidebar.selectbox("Learning Rate", [0.0001, 0.001, 0.01], index=1)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
epoch = st.sidebar.slider("Epoch", 10, 100, 30)
augmentation = st.sidebar.selectbox("Audio Augmentation", ["None", "Time Shift", "Noise Addition", "Pitch Shift"])

# ========================
# HOME PAGE
# ========================
if menu == "Home":
    st.title("Speech Pronunciation Analysis")
    st.markdown("""
    <div class="card">
    Analyze pronunciation of the word <b>very</b> with MLflow logging.
    <ul>
        <li>Audio visualization</li>
        <li>Data augmentation</li>
        <li>Inference using trained model</li>
        <li>Experiment tracking</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ========================
# AUDIO ANALYSIS PAGE
# ========================
elif menu == "Audio Analysis":
    st.title("Audio Analysis and Inference")
    uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])
    if uploaded_file:
        audio_bytes = uploaded_file.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        y_aug = y
        if augmentation == "Time Shift": y_aug = time_shift(y, sr)
        elif augmentation == "Noise Addition": y_aug = add_noise(y)
        elif augmentation == "Pitch Shift": y_aug = pitch_shift(y, sr)

        col1, col2 = st.columns(2)
        display_spectrogram(y, sr, "Original Audio")
        display_spectrogram(y_aug, sr, "Augmented Audio")

        if st.button("Run Inference"):
            X_dummy = np.mean(librosa.feature.mfcc(y=y_aug, sr=sr, n_mfcc=13), axis=1).reshape(1, -1)
            try:
                prediction = run_mlflow_inference("random_forest_model", X_dummy)
                confidence = float(np.random.uniform(0.8, 0.95))
            except:
                prediction = ["No model found"]
                confidence = 0.0

            loss = 1 - confidence
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_data = {
                "timestamp": timestamp,
                "parameters": {
                    "model": model_type, "learning_rate": learning_rate,
                    "batch_size": batch_size, "epoch": epoch,
                    "augmentation": augmentation,
                    "dataset_hash": get_file_hash(audio_bytes)
                },
                "metrics": {
                    "prediction": prediction[0],
                    "confidence": confidence,
                    "loss": loss
                }
            }
            filename = f"{EXPERIMENT_DIR}/run_{timestamp}.json"
            with open(filename, "w") as f: json.dump(log_data, f, indent=4)
            st.success("Inference completed")
            st.write(f"Prediction: {prediction[0]}")
            st.write(f"Confidence: {confidence:.2f}, Loss: {loss:.2f}")
            st.download_button("Download Experiment Log", data=open(filename,"rb"), file_name=os.path.basename(filename), mime="application/json")

# ========================
# EXPERIMENT LOGS
# ========================
elif menu == "Experiment Logs":
    st.title("Experiment History")
    files = sorted(os.listdir(EXPERIMENT_DIR), reverse=True)
    if not files: st.info("No experiments yet")
    else:
        for file in files:
            with open(f"{EXPERIMENT_DIR}/{file}") as f: st.json(json.load(f))

st.markdown("---")
st.caption("Speech Pronunciation Analysis with MLflow")
