import mlflow
import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import hashlib
import io

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Analisis Pengucapan 'Very'",
    page_icon="🎙",
    layout="wide"
)

# ===============================
# MLFLOW CONFIG (SET TRACKING URI KE SQLITE)
# ===============================
mlflow.set_tracking_uri("sqlite:///D:/Project Machine Learning Operations/mlruns/mlflow.db")  # Menyimpan eksperimen di SQLite
mlflow.set_experiment("Analisis_Pengucapan_Very_CNN")

# ===============================
# CSS KUSTOM
# ===============================
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.block-container {
    padding: 2rem;
}
h1 {
    color: #2c3e50;
}
h2, h3 {
    color: #34495e;
}
.stButton>button {
    background-color: #3498db;
    color: white;
    border-radius: 8px;
    font-size: 16px;
    padding: 10px 20px;
}
.stButton>button:hover {
    background-color: #2980b9;
}
.metric-card {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ===============================
# FUNGSI UNTUK VISUALISASI MEL-SPECTROGRAM
# ===============================
def display_spectrogram(y, sr, title):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(9, 4))
    img = librosa.display.specshow(
        mel_db,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

# ===============================
# KODE LAINNYA
# ===============================
# Anda dapat melanjutkan dengan kode yang telah ada untuk upload audio, augmentasi, dsb.
