import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import io

# Menambahkan CSS untuk memperbaiki desain aplikasi
st.markdown("""
    <style>
        .main .block-container {
            padding: 20px;
        }
        h1 {
            color: #2C3E50;
            font-size: 36px;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            font-size: 18px;
            border-radius: 5px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #2980b9;
        }
    </style>
""", unsafe_allow_html=True)

# Judul aplikasi
st.title("🎙️ Analisis Pengucapan Kata 'Very' dengan Deep Learning")

# Deskripsi aplikasi
st.write("""
    Aplikasi ini menggunakan Deep Learning dan Data Augmentation untuk mengklasifikasikan 
    pengucapan kata 'very' oleh orang Indonesia. Unggah file audio untuk melihat analisis 
    spektrogram dan simulasi prediksi model.
""")

# Fungsi untuk menampilkan Mel Spectrogram
def display_spectrogram(y, sr, title="Mel Spectrogram"):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

# Fungsi untuk membuat model CNN
def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

# Fungsi untuk melakukan augmentasi data
def time_shift_audio(y, sr, shift_max=0.2):
    """Menggeser audio dalam domain waktu"""
    shift = np.random.randint(int(sr * -shift_max), int(sr * shift_max))
    return np.roll(y, shift)

def add_noise_audio(y, noise_factor=0.005):
    """Menambahkan noise pada audio"""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def pitch_shift_audio(y, sr, n_steps=2):
    """Mengubah pitch audio"""
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

# Fungsi untuk ekstraksi mel-spectrogram
def get_spectrogram(y, sr, n_mels=128, target_shape=(128, 128)):
    """Ekstraksi mel-spectrogram dari audio"""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Resize ke ukuran target
    from scipy.ndimage import zoom
    zoom_factors = (target_shape[0] / mel_db.shape[0], target_shape[1] / mel_db.shape[1])
    mel_resized = zoom(mel_db, zoom_factors, order=1)
    
    # Normalisasi
    mel_normalized = (mel_resized - mel_resized.min()) / (mel_resized.max() - mel_resized.min())
    
    return np.expand_dims(mel_normalized, axis=-1)

# Upload file audio
st.header("📤 Unggah File Audio untuk Klasifikasi")
uploaded_file = st.file_uploader("Pilih file audio (format WAV)", type=["wav"])

if uploaded_file is not None:
    # Tampilkan audio player
    st.audio(uploaded_file, format="audio/wav")
    
    # Load audio
    y, sr = librosa.load(io.BytesIO(uploaded_file.read()), sr=None)
    
    # Informasi audio
    duration = librosa.get_duration(y=y, sr=sr)
    st.info(f"📊 **Informasi Audio:** Sample Rate: {sr} Hz | Durasi: {duration:.2f} detik")
    
    # Sidebar untuk augmentasi
    st.sidebar.header("🔧 Data Augmentation")
    augmentation = st.sidebar.selectbox(
        "Pilih Jenis Augmentasi", 
        ["Tanpa Augmentasi", "Time Shift", "Noise Addition", "Pitch Shift"]
    )
    
    # Buat dua kolom untuk perbandingan
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎵 Audio Original")
        display_spectrogram(y, sr, "Mel Spectrogram - Original")
    
    # Terapkan augmentasi
    y_aug = y.copy()
    if augmentation == "Time Shift":
        y_aug = time_shift_audio(y, sr)
        st.sidebar.success("✅ Time shift diterapkan")
    elif augmentation == "Noise Addition":
        y_aug = add_noise_audio(y)
        st.sidebar.success("✅ Noise addition diterapkan")
    elif augmentation == "Pitch Shift":
        y_aug = pitch_shift_audio(y, sr)
        st.sidebar.success("✅ Pitch shift diterapkan")
    else:
        st.sidebar.info("ℹ️ Tidak ada augmentasi")
    
    with col2:
        st.subheader(f"🎵 Audio Setelah {augmentation}")
        display_spectrogram(y_aug, sr, f"Mel Spectrogram - {augmentation}")
    
    # Ekstraksi fitur
    st.header("🔬 Ekstraksi Fitur dan Prediksi")
    
    with st.spinner("Memproses spektrogram..."):
        spectrogram = get_spectrogram(y_aug, sr)
    
    # Buat model
    model = create_model(spectrogram.shape)
    
    # Tampilkan arsitektur model
    with st.expander("📋 Lihat Arsitektur Model"):
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        st.code('\n'.join(model_summary))
    
    # Training simulasi dan prediksi
    st.subheader("🎯 Training Model dan Prediksi")
    
    # Tombol untuk simulasi training
    if st.button("🚀 Latih Model (Simulasi)", type="primary"):
        # Simulasi proses training
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulasi data training
        st.write("### 📊 Proses Training")
        epochs = 10
        train_acc_history = []
        val_acc_history = []
        train_loss_history = []
        val_loss_history = []
        
        # Simulasi training per epoch
        for epoch in range(epochs):
            # Simulasi akurasi yang meningkat
            train_acc = min(0.5 + (epoch * 0.05) + np.random.uniform(0, 0.02), 0.98)
            val_acc = min(0.45 + (epoch * 0.045) + np.random.uniform(0, 0.03), 0.92)
            train_loss = max(0.7 - (epoch * 0.06) + np.random.uniform(-0.02, 0.02), 0.05)
            val_loss = max(0.75 - (epoch * 0.055) + np.random.uniform(-0.03, 0.03), 0.12)
            
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            
            # Update progress
            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
        
        progress_bar.empty()
        status_text.empty()
        st.success("✅ Training selesai!")
        
        # Tampilkan metrik akhir
        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
        with col_metric1:
            st.metric("Train Accuracy", f"{train_acc_history[-1]*100:.2f}%")
        with col_metric2:
            st.metric("Val Accuracy", f"{val_acc_history[-1]*100:.2f}%")
        with col_metric3:
            st.metric("Train Loss", f"{train_loss_history[-1]:.4f}")
        with col_metric4:
            st.metric("Val Loss", f"{val_loss_history[-1]:.4f}")
        
        # Plot training history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        ax1.plot(range(1, epochs+1), train_acc_history, 'bo-', label='Training Accuracy', linewidth=2)
        ax1.plot(range(1, epochs+1), val_acc_history, 'ro-', label='Validation Accuracy', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(range(1, epochs+1), train_loss_history, 'bo-', label='Training Loss', linewidth=2)
        ax2.plot(range(1, epochs+1), val_loss_history, 'ro-', label='Validation Loss', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Confusion Matrix simulasi
        st.write("### 📈 Confusion Matrix")
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Simulasi prediksi
        y_true = np.random.randint(0, 2, 100)
        y_pred = y_true.copy()
        # Tambahkan beberapa error
        error_indices = np.random.choice(100, 15, replace=False)
        y_pred[error_indices] = 1 - y_pred[error_indices]
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['Salah', 'Benar'], 
                    yticklabels=['Salah', 'Benar'])
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)
        
        # Classification Report
        from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        st.write("### 📋 Classification Report")
        col_cr1, col_cr2, col_cr3, col_cr4 = st.columns(4)
        with col_cr1:
            st.metric("Accuracy", f"{accuracy*100:.2f}%")
        with col_cr2:
            st.metric("Precision", f"{precision*100:.2f}%")
        with col_cr3:
            st.metric("Recall", f"{recall*100:.2f}%")
        with col_cr4:
            st.metric("F1-Score", f"{f1*100:.2f}%")
        
        # Store trained status in session state
        st.session_state.model_trained = True
        st.session_state.final_accuracy = val_acc_history[-1]
    
    # Prediksi pada audio yang diupload
    st.write("---")
    st.subheader("🎤 Prediksi Audio yang Diupload")
    
    if 'model_trained' in st.session_state and st.session_state.model_trained:
        # Prediksi dengan model "terlatih"
        confidence = np.random.uniform(0.75, 0.95)
        prediction_class = 1 if confidence > 0.5 else 0
        prediction = "Pengucapan Benar ✅" if prediction_class == 1 else "Pengucapan Perlu Diperbaiki ⚠️"
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Prediksi", prediction)
        with col4:
            st.metric("Confidence", f"{confidence*100:.1f}%")
        
        # Progress bar untuk confidence
        st.progress(confidence)
        
        # Detail prediksi
        with st.expander("📊 Detail Probabilitas"):
            prob_correct = confidence if prediction_class == 1 else 1 - confidence
            prob_incorrect = 1 - prob_correct
            
            st.write("**Probabilitas per Kelas:**")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.write(f"- Pengucapan Salah: {prob_incorrect*100:.2f}%")
            with col_p2:
                st.write(f"- Pengucapan Benar: {prob_correct*100:.2f}%")
        
        st.info(f"ℹ️ Model telah dilatih dengan akurasi validasi: {st.session_state.final_accuracy*100:.2f}%")
    else:
        st.warning("⚠️ Model belum dilatih. Klik tombol 'Latih Model (Simulasi)' di atas untuk melatih model terlebih dahulu.")
        st.info("💡 Setelah model dilatih, Anda dapat melihat prediksi untuk audio yang diupload.")
    
    # Tips untuk training
    with st.expander("💡 Tips untuk Melatih Model"):
        st.markdown("""
        **Langkah-langkah untuk melatih model:**
        1. Kumpulkan dataset audio pengucapan 'very' (benar dan salah)
        2. Label data dengan kategori yang sesuai
        3. Ekstraksi mel-spectrogram dari semua audio
        4. Split data menjadi train/validation/test set
        5. Latih model dengan `model.fit()`
        6. Evaluasi performa dengan data test
        7. Save model dengan `model.save('model_pronunciation.h5')`
        
        **Dataset yang direkomendasikan:**
        - Minimal 500-1000 sampel per kelas
        - Durasi audio 1-3 detik
        - Variasi speaker yang beragam
        - Kondisi rekaman yang berbeda
        """)

else:
    st.info("👆 Silakan unggah file audio untuk memulai analisis")
    
    # Contoh penggunaan
    with st.expander("📖 Cara Menggunakan Aplikasi"):
        st.markdown("""
        1. **Unggah Audio**: Klik tombol "Browse files" dan pilih file WAV
        2. **Pilih Augmentasi**: Gunakan sidebar untuk memilih teknik augmentasi
        3. **Lihat Hasil**: Bandingkan spektrogram sebelum dan sesudah augmentasi
        4. **Analisis Prediksi**: Lihat hasil prediksi model (simulasi)
        
        **Format Audio yang Didukung:**
        - Format: WAV
        - Sample Rate: Bebas (akan di-resample otomatis)
        - Durasi: 1-5 detik (optimal)
        """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️")
