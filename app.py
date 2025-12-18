import streamlit as st
import tensorflow as tf
import librosa
import numpy as np

st.set_page_config(page_title="Speech Pronunciation Analysis", layout="centered")

st.title("🔊 Analisis Pengucapan Kata 'Very'")
st.write("Upload file audio (.wav) untuk mengecek kualitas pengucapan")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/very_model.keras")

model = load_model()

# Feature extraction
def extract_mel(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db[:, :60]   # SAMAKAN DENGAN TRAINING
    mel_db = mel_db.reshape(1, 128, 60, 1)
    return mel_db

uploaded_file = st.file_uploader("Upload audio (.wav)", type=["wav"])

if uploaded_file:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("temp.wav")

    features = extract_mel("temp.wav")
    prediction = model.predict(features)

    score = float(prediction[0][0])

    st.subheader("Hasil Prediksi")
    st.write(f"Confidence score: **{score:.2f}**")

    if score > 0.5:
        st.success("✅ Pengucapan 'Very' BENAR")
    else:
        st.error("❌ Pengucapan 'Very' KURANG TEPAT")
