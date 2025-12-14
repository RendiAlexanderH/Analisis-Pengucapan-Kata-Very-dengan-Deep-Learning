import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import mlflow
import hashlib

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Speech Pronunciation Analysis",
    page_icon="🎙",
    layout="wide"
)

# ===============================
# MLFLOW CONFIG
# ===============================
mlflow.set_tracking_uri(
    "sqlite:///D:/Project Machine Learning Operations/mlruns/mlflow.db"
)
mlflow.set_experiment("Analisis_Pengucapan_Very_CNN")

# ===============================
# CSS UI
# ===============================
st.markdown("""
<style>
.block-container { padding: 2.5rem; }
h1 { color: #1f2c56; }
.sidebar-title { font-size: 20px; font-weight: bold; }
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
    return np.roll(y, np.random.randint(-sr//10, sr//10))

def add_noise(y):
    return y + 0.005 * np.random.randn(len(y))

def pitch_shift(y, sr):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2)

def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

# ===============================
# SIDEBAR (CONTROL CENTER)
# ===============================
st.sidebar.markdown("### 🎛 Control Panel")

menu = st.sidebar.radio(
    "Navigasi",
    ["🏠 Home", "🎧 Analisis Audio", "📊 MLflow Info"]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 🤖 Model Selection")
model_type = st.sidebar.selectbox(
    "Pilih Model",
    ["CNN", "CRNN", "Transformer"]
)

st.sidebar.markdown("### ⚙ Hyperparameter")
learning_rate = st.sidebar.select_slider(
    "Learning Rate",
    options=[0.0001, 0.001, 0.01],
    value=0.001
)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
epoch = st.sidebar.slider("Epoch", 10, 100, 30)

st.sidebar.markdown("### 🔊 Audio Augmentation")
augmentasi = st.sidebar.selectbox(
    "Augmentasi",
    ["Tanpa Augmentasi", "Time Shift", "Noise Addition", "Pitch Shift"]
)

# ===============================
# HOME PAGE
# ===============================
if menu == "🏠 Home":
    st.title("🎙 Speech Pronunciation Analysis")
    st.markdown("""
    <div class="card">
    <b>Aplikasi ini bertujuan untuk:</b><br><br>
    ✅ Analisis pengucapan kata <b><i>very</i></b><br>
    ✅ Visualisasi Mel-Spectrogram<br>
    ✅ Simulasi inferensi model Deep Learning<br>
    ✅ Experiment Tracking menggunakan <b>MLflow</b><br><br>
    <i>Dirancang untuk riset, skripsi, dan demonstrasi MLOps.</i>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# ANALISIS AUDIO PAGE
# ===============================
elif menu == "🎧 Analisis Audio":
    st.title("🎧 Analisis Audio & Inferensi")

    uploaded_file = st.file_uploader("Upload Audio WAV", type=["wav"])

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
            with mlflow.start_run(run_name="Inference_Run"):
                confidence = float(np.random.uniform(0.75, 0.95))

                mlflow.log_params({
                    "model": model_type,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "epoch": epoch,
                    "augmentation": augmentasi,
                    "dataset_hash": get_file_hash(audio_bytes)
                })

                mlflow.log_metrics({
                    "confidence": confidence,
                    "loss": 1 - confidence
                })

                st.success(f"Prediksi: Pengucapan Benar ({confidence*100:.2f}%)")

# ===============================
# MLFLOW INFO PAGE
# ===============================
elif menu == "📊 MLflow Info":
    st.title("📊 MLflow Experiment Tracking")
    st.markdown("""
    <div class="card">
    <b>Experiment Name:</b> Analisis_Pengucapan_Very_CNN<br>
    <b>Tracking URI:</b> SQLite<br><br>
    Jalankan perintah berikut untuk membuka MLflow UI:
    <pre>mlflow ui</pre>
    Lalu buka <b>http://localhost:5000</b>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("🚀 Built with Streamlit • MLflow • Librosa | Research & MLOps Ready")
