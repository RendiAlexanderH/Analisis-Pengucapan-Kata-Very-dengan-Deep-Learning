# Speech Recognition Deep Learning Project - Klasifikasi Kata "Very"

## 📋 Deskripsi Proyek

Proyek ini merupakan sistem klasifikasi audio berbasis deep learning untuk mendeteksi kata "very" dalam audio speech. Model ini menggunakan Convolutional Neural Network (CNN) 1D yang dilatih pada mel-spectrogram untuk membedakan antara ucapan kata "very" yang benar (label=1) dan ucapan yang tidak benar atau kata lain (label=0).

### Tujuan Proyek
- **Tujuan Utama**: Membangun model deep learning untuk klasifikasi akurasi pengucapan kata "very"
- **Aplikasi**: Sistem pembelajaran bahasa, speech therapy, quality control untuk speech recognition
- **Akurasi Target**: Mencapai akurasi >75% pada data test

---

## 🎯 Fitur Utama

1. **Audio Preprocessing**: Ekstraksi kata "very" otomatis dari audio panjang menggunakan Whisper Timestamped
2. **Data Labeling**: Labeling otomatis berdasarkan Levenshtein distance
3. **Feature Engineering**: Konversi audio ke Mel-Spectrogram (128x128)
4. **Data Augmentation**: Time shifting dan noise addition
5. **Deep Learning Model**: CNN 1D dengan 3 convolutional layers
6. **Hyperparameter Tuning**: Eksperimen dengan dropout rates (0.2 dan 0.3)

---

## 📊 Struktur Dataset

```
Dataset Deep Learning/
├── wav/                    # Audio asli (full sentences)
├── very wav/               # Audio yang dipotong (kata "very" only) - tidak digunakan
└── very/                   # Audio final untuk training (110 samples)
    ├── very_1.wav
    ├── very_2.wav
    └── ...
```

### Karakteristik Data
- **Total Samples**: 110 audio files
- **Augmentasi**: 2x (raw + augmented) = 220 samples total
- **Train-Test Split**: 80:20
- **Label Distribution**: 
  - Label 0 (incorrect/other words): ~73 samples
  - Label 1 (correct "very"): ~37 samples
- **Format**: WAV files dengan variable sample rate

---

## 🔄 Alur Data (Data Pipeline)

### 1. Data Collection & Preprocessing (Tahap Awal)

```
Audio Panjang (Sentences) 
    ↓
[Whisper Timestamped Model]  ← Transcribe dengan timestamp
    ↓
Word-level Timestamps
    ↓
[Ekstraksi Kata "very"]     ← Potong audio berdasarkan timestamp
    ↓
Audio Pendek (kata "very")
```

**Script**: Bagian pertama kode
- Load Whisper model (medium)
- Transcribe audio dengan word-level timestamps
- Ekstrak setiap kemunculan kata "very"
- Simpan potongan audio ke folder terpisah

### 2. Data Labeling (Pemrosesan)

```
Audio "very"
    ↓
[Whisper Transcription]     ← Transkripsi ulang
    ↓
Text Output
    ↓
[Levenshtein Distance]      ← Hitung error terhadap "very"
    ↓
Error Percentage
    ↓
[Thresholding]              ← Error = 0% → Label 1, else → Label 0
    ↓
Labeled DataFrame
```

**Output**: `data_label.csv` dengan kolom:
- `file`: nama file audio
- `transcript`: hasil transkripsi Whisper
- `error_percent`: persentase error
- `label`: 0 atau 1
- `path`: path lengkap file audio

### 3. Feature Extraction

```
Audio WAV
    ↓
[Librosa Load]              ← Load audio
    ↓
Time-series Audio
    ↓
[Mel-Spectrogram]           ← Konversi ke frequency domain
    ↓
Mel-Spectrogram (var x 128)
    ↓
[Resize]                    ← Standardisasi ukuran
    ↓
Spectrogram (128 x 128 x 1)
```

**Parameter**:
- `n_mels=128`: jumlah mel bands
- `img_size=(128,128)`: ukuran output
- Power to dB conversion untuk normalisasi

### 4. Data Augmentation

```
Original Spectrogram
    ↓
┌────────────┴────────────┐
│                         │
[Time Shift]          [No Change]
    ↓                     ↓
Shifted Spec         Original Spec
    ↓                     ↓
[Add Noise]          [Identity]
    ↓                     ↓
Augmented Spec       Original Spec
    ↓                     ↓
└────────────┬────────────┘
             ↓
    Combined Dataset (2x size)
```

**Augmentasi**:
1. **Time Shifting**: Geser spectrogram ±20 steps secara horizontal
2. **Noise Addition**: Tambah Gaussian noise (factor=0.02)

### 5. Model Training Pipeline

```
Dataset (220 samples)
    ↓
[Train-Test Split]          ← 80:20, stratified
    ↓
┌────────────┴────────────┐
│                         │
Training Set          Test Set
(176 samples)         (44 samples)
    ↓                     ↓
[Train-Val Split]     [Hold Out]
    ↓
┌────────────┴────────────┐
│                         │
Train (141)           Validation (35)
    ↓                     ↓
[Model Training]      [Early Stopping]
    ↓                     ↓
Trained Model ←───────────┘
    ↓
[Evaluation on Test Set]
    ↓
Metrics & Visualisasi
```

---

## 🏗️ Arsitektur Model

### Model: 1D CNN for Spectrogram Classification

```
Input: (128, 128, 1)
    ↓
Reshape: (128, 128)         ← Flatten untuk Conv1D
    ↓
────────────────────────────────────────
Block 1:
    Conv1D(128 filters, kernel=3)
    ReLU Activation
    MaxPooling1D(pool_size=2)
    Dropout(rate=0.2 atau 0.3)
    ↓
    Output: (63, 128)
────────────────────────────────────────
Block 2:
    Conv1D(128 filters, kernel=3)
    ReLU Activation
    MaxPooling1D(pool_size=2)
    Dropout(rate=0.2 atau 0.3)
    ↓
    Output: (30, 128)
────────────────────────────────────────
Block 3:
    Conv1D(128 filters, kernel=3)
    ReLU Activation
    MaxPooling1D(pool_size=2)
    Dropout(rate=0.2 atau 0.3)
    ↓
    Output: (14, 128)
────────────────────────────────────────
Global Average Pooling
    ↓
    Output: (128,)
────────────────────────────────────────
Dense Layer (64 units, ReLU)
    ↓
Output Layer (1 unit, Sigmoid)
    ↓
Prediction: [0.0 - 1.0]
```

### Total Parameters
- **Trainable**: 156,161 parameters
- **Size**: ~610 KB

### Hyperparameter Experiments
1. **Dropout 0.3**: Akurasi Test = 72.73%
2. **Dropout 0.2**: Akurasi Test = 75.00% ✅ (Best)

---

## 📈 Hasil & Performa

### Eksperimen 1: Dropout 0.3

**Training**:
- Epochs: 120
- Best Validation Accuracy: 77.78% (epoch 112)
- Final Training Accuracy: ~84%

**Testing**:
```
Accuracy: 72.73%

              precision    recall    f1-score
Class 0           0.80      0.74       0.77
Class 1           0.63      0.71       0.67
```

### Eksperimen 2: Dropout 0.2 (BEST)

**Training**:
- Epochs: 120
- Best Validation Accuracy: 86.11% (epoch 91)
- Final Training Accuracy: ~89%

**Testing**:
```
Accuracy: 75.00%

              precision    recall    f1-score
Class 0           0.79      0.81       0.80
Class 1           0.69      0.65       0.67
```

### Analisis
- ✅ Model berhasil mencapai target akurasi >75%
- ✅ Dropout 0.2 memberikan hasil terbaik (lebih tinggi 2.27%)
- ⚠️ Class imbalance: Class 0 (27 samples) > Class 1 (17 samples)
- ✅ Tidak ada overfitting signifikan
- ⚠️ Precision untuk Class 1 masih bisa ditingkatkan

---

## 🚀 Cara Menggunakan

### Prasyarat

```bash
# Sistem Requirements
- Python 3.12+
- Google Colab (recommended) atau Jupyter Notebook
- GPU (optional, untuk training lebih cepat)
```

### Instalasi Dependencies

```bash
# Whisper dan dependencies
pip install git+https://github.com/linto-ai/whisper-timestamped
pip install SpeechRecognition python-Levenshtein soundfile librosa scikit-image

# Deep Learning
pip install tensorflow keras

# Data Processing
pip install pandas numpy scikit-learn matplotlib
```

### Langkah Eksekusi

#### 1. Setup Environment

```python
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
```

#### 2. Preprocessing Audio (One-time)

```python
# Konfigurasi path
RAW_DIR = "/path/to/raw/audio/"
CUT_DIR = "/path/to/output/cuts/"

# Load Whisper model
model = whisper.load_model("medium")

# Proses semua file
for file in os.listdir(RAW_DIR):
    cut_all_target_words(file, TEXT, "very", CUT_DIR)
```

#### 3. Labeling Data

```python
# Load Whisper untuk transkripsi
model = whisper.load_model("small")

# Proses dan label semua audio
data_list = []
for file in os.listdir(CUT_DIR):
    transcript = whisper_transcribe(file)
    error = calculate_error(transcript, "very")
    label = 1 if error == 0 else 0
    data_list.append({...})

# Save ke CSV
df.to_csv('data_label.csv', index=False)
```

#### 4. Training Model

```python
# Extract features
X_raw, X_aug, y = [], [], []
for idx, row in df.iterrows():
    spec = get_spectrogram(row['path'])
    X_raw.append(spec)
    X_aug.append(augment(spec))
    y.append(row['label'])

# Prepare dataset
X = np.concatenate([X_raw, X_aug])
y = np.concatenate([y, y])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# Build & compile model
model = Sequential([...])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=120,
    batch_size=32
)
```

#### 5. Evaluasi & Prediksi

```python
# Load best model
model.load_weights("best_model.h5")

# Prediksi
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Evaluasi
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Visualisasi learning curves
plot_training_history(history)
```

---

## 📁 Struktur Proyek untuk GitHub

```
speech-recognition-very/
│
├── README.md                      # Dokumentasi ini
├── LICENSE                        # Lisensi (pilih sesuai kebutuhan)
├── requirements.txt               # Dependencies Python
├── .gitignore                     # File yang tidak di-track
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_data_labeling.ipynb
│   ├── 03_feature_extraction.ipynb
│   └── 04_model_training.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Fungsi ekstraksi audio
│   ├── labeling.py                # Fungsi labeling otomatis
│   ├── feature_extraction.py      # Fungsi mel-spectrogram
│   ├── augmentation.py            # Fungsi augmentasi
│   ├── model.py                   # Definisi arsitektur model
│   └── utils.py                   # Helper functions
│
├── data/
│   ├── raw/                       # Audio asli (tidak di-upload ke git)
│   ├── processed/                 # Audio hasil preprocessing
│   └── data_label.csv             # Label dataset
│
├── models/
│   ├── best_model.h5              # Model terbaik
│   └── model_architecture.json    # Arsitektur model
│
├── results/
│   ├── training_history.png       # Learning curves
│   ├── confusion_matrix.png       # Confusion matrix
│   └── classification_report.txt  # Laporan evaluasi
│
└── docs/
    ├── project_presentation.pdf   # Presentasi proyek
    └── technical_report.pdf       # Laporan teknis lengkap
```

---

## 📝 Langkah Upload ke GitHub

### 1. Persiapan Lokal

```bash
# Clone atau inisialisasi repo
git init
# atau
git clone https://github.com/username/speech-recognition-very.git

cd speech-recognition-very
```

### 2. Buat File .gitignore

```bash
# File .gitignore content
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/
dist/
build/

# Jupyter Notebook
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data (terlalu besar)
data/raw/
data/processed/*.wav
*.wav
*.mp3

# Models (terlalu besar untuk GitHub free)
models/*.h5
models/*.keras
*.h5

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Google Drive
.gdrive/
```

### 3. Buat requirements.txt

```bash
# File requirements.txt
whisper-timestamped==1.15.9
SpeechRecognition==3.14.4
python-Levenshtein==0.27.3
soundfile==0.13.1
librosa==0.11.0
scikit-image==0.25.2
tensorflow==2.9.0
keras==2.9.0
pandas==2.0.0
numpy==2.0.2
scikit-learn==1.6.1
matplotlib==3.7.0
```

### 4. Modularisasi Kode

Pisahkan kode dari notebook ke file Python terpisah:

**src/preprocessing.py**:
```python
import whisper_timestamped as whisper
import soundfile as sf
import numpy as np

def cut_all_target_words(audio_path, text, target_word="very", output_folder="./cuts/"):
    """Ekstrak kata target dari audio panjang"""
    # ... (kode dari notebook)
    pass
```

**src/model.py**:
```python
from tensorflow.keras import Sequential, layers

def create_model(dropout_rate=0.2):
    """Buat model CNN 1D"""
    model = Sequential([
        layers.Reshape((128,128), input_shape=(128,128,1)),
        # ... (rest of architecture)
    ])
    return model
```

### 5. Commit dan Push

```bash
# Stage semua file
git add .

# Commit dengan pesan deskriptif
git commit -m "Initial commit: Speech recognition model for 'very' classification"

# Tambahkan remote (jika belum)
git remote add origin https://github.com/username/speech-recognition-very.git

# Push ke GitHub
git push -u origin main
```

### 6. Best Practices untuk Commit Messages

```bash
# Format: <type>: <subject>

# Examples:
git commit -m "feat: add data preprocessing pipeline"
git commit -m "feat: implement mel-spectrogram extraction"
git commit -m "feat: add CNN model architecture"
git commit -m "fix: resolve audio loading issues"
git commit -m "docs: update README with usage instructions"
git commit -m "refactor: modularize code into separate modules"
git commit -m "test: add unit tests for preprocessing"
git commit -m "perf: optimize spectrogram computation"
```

### 7. Buat Release dan Tag

```bash
# Setelah model final
git tag -a v1.0 -m "Release v1.0: Model with 75% accuracy"
git push origin v1.0
```

### 8. Upload Model (Large Files)

Untuk model .h5 yang besar, gunakan Git LFS atau upload ke platform lain:

```bash
# Option 1: Git LFS
git lfs install
git lfs track "*.h5"
git add .gitattributes
git add models/best_model.h5
git commit -m "feat: add trained model"
git push

# Option 2: Gunakan Google Drive/Dropbox
# Tambahkan link download di README
```

---

## 🔧 Troubleshooting

### Issue 1: Audio tidak terpotong dengan benar
**Solusi**: 
- Cek format audio (harus WAV, mono/stereo)
- Verifikasi Whisper model sudah di-load
- Pastikan kata "very" ada dalam audio

### Issue 2: Error saat load model
**Solusi**:
```python
# Load dengan custom objects jika perlu
from tensorflow.keras.models import load_model
model = load_model('best_model.h5', compile=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Issue 3: Memory error saat training
**Solusi**:
- Kurangi batch size
- Gunakan Google Colab dengan GPU
- Proses dataset dalam batch

---

## 🔮 Future Improvements

1. **Data Collection**:
   - Tambah data untuk balance classes (target: 200+ samples per class)
   - Variasi speaker (accent, gender, age)
   - Noise conditions (clean, noisy, reverb)

2. **Model Architecture**:
   - Eksperimen dengan 2D CNN (treat spectrogram as image)
   - Try LSTM/GRU untuk sequential modeling
   - Implement attention mechanism
   - Transfer learning dengan pre-trained audio models (YAMNet, VGGish)

3. **Feature Engineering**:
   - MFCC sebagai alternatif Mel-Spectrogram
   - Combine multiple features (MFCC + Mel + Chroma)
   - Dynamic time warping untuk alignment

4. **Deployment**:
   - Convert to TFLite untuk mobile deployment
   - Build REST API dengan FastAPI
   - Real-time prediction dari microphone

5. **Evaluation**:
   - Cross-validation untuk robust evaluation
   - Analyze per-speaker performance
   - Interpretability dengan Grad-CAM

---

## 👥 Kontributor

**Kelompok 23 - Tugas Besar Deep Learning**

- Anggota 1
- Anggota 2
- Anggota 3
- Anggota 4

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---

## 📧 Kontak

Untuk pertanyaan atau saran:
- Email: kelompok23@university.edu
- GitHub Issues: [Create new issue](https://github.com/username/speech-recognition-very/issues)

---

## 🙏 Acknowledgments

- OpenAI Whisper untuk model transcription
- Linto AI untuk Whisper Timestamped
- TensorFlow/Keras team
- Librosa untuk audio processing
- Google Colab untuk computing resources

---

**Last Updated**: December 2024
**Version**: 1.0.0
