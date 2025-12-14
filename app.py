import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io


# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Analisis Pengucapan 'Very'",
    page_icon="🎙️",
    layout="wide"
)


# CSS KUSTOM (DESAIN)

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


# JUDUL & DESKRIPSI

st.title("🎙️ Analisis Pengucapan Kata *Very*")
st.write("""
Aplikasi ini menganalisis **pengucapan kata *very*** menggunakan **ekstraksi fitur audio (Mel-Spectrogram)**  
dan **simulasi Deep Learning (CNN)**.

> ⚠️ *Catatan:*  
> Model CNN **dilatih secara offline**.  
> Versi Streamlit ini berfungsi sebagai **visualisasi, augmentasi, dan simulasi inferensi**.
""")

st.markdown("---")

# FUNGSI AUDIO
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

def time_shift(y, sr, max_shift=0.2):
    shift = np.random.randint(-int(sr * max_shift), int(sr * max_shift))
    return np.roll(y, shift)

def add_noise(y, factor=0.005):
    noise = np.random.randn(len(y))
    return y + factor * noise

def pitch_shift(y, sr, steps=2):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=steps)


# SIDEBAR

st.sidebar.header("⚙️ Pengaturan")
augmentasi = st.sidebar.selectbox(
    "Pilih Augmentasi Audio",
    ["Tanpa Augmentasi", "Time Shift", "Noise Addition", "Pitch Shift"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
📌 **Alur Sistem**
1. Upload audio  
2. Ekstraksi mel-spectrogram  
3. Augmentasi (opsional)  
4. Simulasi prediksi CNN
""")


# UPLOAD AUDIO

st.header("📤 Upload Audio")
uploaded_file = st.file_uploader(
    "Upload file audio (WAV)",
    type=["wav"]
)

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format="audio/wav")

    # Load audio
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    st.success(f"Sample Rate: {sr} Hz | Durasi: {duration:.2f} detik")

    # Augmentasi
    y_aug = y.copy()
    if augmentasi == "Time Shift":
        y_aug = time_shift(y, sr)
    elif augmentasi == "Noise Addition":
        y_aug = add_noise(y)
    elif augmentasi == "Pitch Shift":
        y_aug = pitch_shift(y, sr)

    # ===============================
    # VISUALISASI
    # ===============================
    st.markdown("---")
    st.header("📊 Visualisasi Mel-Spectrogram")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Audio Original")
        display_spectrogram(y, sr, "Mel-Spectrogram (Original)")

    with col2:
        st.subheader(f"Audio Setelah {augmentasi}")
        display_spectrogram(y_aug, sr, f"Mel-Spectrogram ({augmentasi})")

    # SIMULASI PREDIKSI
  
    st.markdown("---")
    st.header("🎯 Simulasi Prediksi CNN")

    if st.button("🚀 Jalankan Prediksi (Demo)"):
        confidence = np.random.uniform(0.75, 0.95)
        label = "Pengucapan Benar " if confidence > 0.5 else "Perlu Perbaikan ⚠️"

        colm1, colm2 = st.columns(2)
        with colm1:
            st.metric("Hasil Prediksi", label)
        with colm2:
            st.metric("Confidence", f"{confidence*100:.2f}%")

        st.progress(confidence)

        with st.expander("📌 Detail Probabilitas"):
            st.write(f"- Benar: {confidence*100:.2f}%")
            st.write(f"- Salah: {(1-confidence)*100:.2f}%")

else:
    st.info("⬆️ Silakan upload file audio WAV untuk memulai analisis.")

# INFORMASI TAMBAHAN

st.markdown("---")
with st.expander("📘 Catatan Implementasi (Untuk TA)"):
    st.markdown("""
- Model CNN dilatih **secara offline** menggunakan TensorFlow  
- Aplikasi Streamlit digunakan untuk:
  - Visualisasi sinyal audio  
  - Augmentasi data  
  - Simulasi inferensi  
- Pendekatan ini digunakan karena **keterbatasan environment cloud**
""")

st.markdown("---")
st.caption("Made with ❤️ by Kami")

