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

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Analisis Pengucapan Kata",
    layout="wide"
)

# ======================================================
# FOLDER PENYIMPANAN
# ======================================================
FOLDER_EKSPERIMEN = "experiments"
os.makedirs(FOLDER_EKSPERIMEN, exist_ok=True)

# ======================================================
# STYLE TAMPILAN
# ======================================================
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

# ======================================================
# FUNGSI UTILITAS
# ======================================================
def tampilkan_spektrogram(y, sr, judul):
    """Menampilkan Mel Spectrogram audio"""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 4))
    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        ax=ax
    )
    ax.set_title(judul)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)
    plt.close(fig)


def augment_time_shift(y, sr):
    """Augmentasi pergeseran waktu"""
    shift = np.random.randint(-sr // 10, sr // 10)
    return np.roll(y, shift)


def augment_noise(y):
    """Augmentasi penambahan noise"""
    noise = 0.005 * np.random.randn(len(y))
    return y + noise


def augment_pitch(y, sr):
    """Augmentasi pitch shift"""
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2)


def hash_file(file_bytes):
    """Hash file audio untuk identifikasi"""
    return hashlib.md5(file_bytes).hexdigest()


def ekstraksi_fitur(y, sr, durasi=2.0, n_mels=128):
    """Ekstraksi fitur Mel Spectrogram"""
    panjang_maks = int(sr * durasi)

    if len(y) > panjang_maks:
        y = y[:panjang_maks]
    else:
        y = np.pad(y, (0, panjang_maks - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

    return mel_db


def inferensi_dummy(fitur):
    """
    Inferensi SEMENTARA (dummy)
    Digunakan agar app bisa jalan tanpa TensorFlow
    """
    confidence = np.random.uniform(0.6, 0.9)
    if confidence > 0.75:
        prediksi = "Pengucapan Benar"
    else:
        prediksi = "Pengucapan Kurang Tepat"
    return prediksi, float(confidence)

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("Panel Kontrol")

menu = st.sidebar.radio(
    "Navigasi",
    ["Beranda", "Analisis Audio", "Log Eksperimen"]
)

augmentasi = st.sidebar.selectbox(
    "Jenis Augmentasi Audio",
    ["Tidak Ada", "Time Shift", "Noise", "Pitch Shift"]
)

# ======================================================
# HALAMAN BERANDA
# ======================================================
if menu == "Beranda":
    st.title("Analisis Pengucapan Kata 'Very'")

    st.markdown("""
    <div class="card">
        <p>
        Aplikasi ini digunakan untuk menganalisis pengucapan kata <b>very</b>
        menggunakan sinyal audio.
        </p>
        <ul>
            <li>Visualisasi Mel Spectrogram</li>
            <li>Augmentasi data audio</li>
            <li>Inferensi pengucapan</li>
            <li>Pencatatan hasil eksperimen</li>
        </ul>
        <p><b>Status:</b> Kompatibel Streamlit Cloud ✅</p>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# HALAMAN ANALISIS AUDIO
# ======================================================
elif menu == "Analisis Audio":
    st.title("Analisis Audio & Inferensi")

    file_audio = st.file_uploader(
        "Upload file audio (.wav)",
        type=["wav"]
    )

    if file_audio:
        audio_bytes = file_audio.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)

        y_aug = y.copy()
        if augmentasi == "Time Shift":
            y_aug = augment_time_shift(y, sr)
        elif augmentasi == "Noise":
            y_aug = augment_noise(y)
        elif augmentasi == "Pitch Shift":
            y_aug = augment_pitch(y, sr)

        col1, col2 = st.columns(2)
        with col1:
            tampilkan_spektrogram(y, sr, "Audio Asli")
        with col2:
            tampilkan_spektrogram(y_aug, sr, "Audio Setelah Augmentasi")

        if st.button("Jalankan Inferensi"):
            fitur = ekstraksi_fitur(y_aug, sr)
            prediksi, confidence = inferensi_dummy(fitur)
            loss = 1 - confidence

            waktu = datetime.now().strftime("%Y%m%d_%H%M%S")

            log = {
                "timestamp": waktu,
                "augmentasi": augmentasi,
                "prediksi": prediksi,
                "confidence": confidence,
                "loss": loss,
                "hash_file": hash_file(audio_bytes)
            }

            nama_file = f"{FOLDER_EKSPERIMEN}/run_{waktu}.json"
            with open(nama_file, "w") as f:
                json.dump(log, f, indent=4)

            st.success("Inferensi berhasil dilakukan")
            st.metric("Prediksi", prediksi)
            st.metric("Confidence", f"{confidence:.2f}")
            st.metric("Loss", f"{loss:.2f}")

            st.download_button(
                "Unduh Log Eksperimen",
                data=json.dumps(log, indent=4),
                file_name=os.path.basename(nama_file),
                mime="application/json"
            )

# ======================================================
# HALAMAN LOG EKSPERIMEN
# ======================================================
elif menu == "Log Eksperimen":
    st.title("Riwayat Eksperimen")

    files = sorted(os.listdir(FOLDER_EKSPERIMEN), reverse=True)

    if not files:
        st.info("Belum ada eksperimen.")
    else:
        for file in files:
            with open(os.path.join(FOLDER_EKSPERIMEN, file)) as f:
                st.json(json.load(f))

st.markdown("---")
st.caption("Analisis Pengucapan Kata | Streamlit Cloud Ready ✅")
