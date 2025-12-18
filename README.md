# Speech Recognition Deep Learning Project - Klasifikasi Kata "Very"

## ML Canvas

![ML Canvas](https://github.com/RendiAlexanderH/Analisis-Pengucapan-Kata-Very-dengan-Deep-Learning/blob/main/Salinan%20dari%20ML%20CANVAS_10.png)

## 📋 Deskripsi Proyek

Proyek ini merupakan sistem klasifikasi audio berbasis deep learning untuk mendeteksi kata "very" dalam audio speech. Model ini menggunakan Convolutional Neural Network (CNN) 1D yang dilatih pada mel-spectrogram untuk membedakan antara ucapan kata "very" yang benar (label=1) dan ucapan yang tidak benar atau kata lain (label=0).

### Tujuan Proyek
- **Tujuan Utama**: Membangun model deep learning untuk klasifikasi akurasi pengucapan kata "very"
- **Aplikasi**: Sistem pembelajaran bahasa, speech therapy, quality control untuk speech recognition

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
  - Label 0 (salah): 85 samples
  - Label 1 (benar): 50 samples
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
Visualisasi Evaluasi
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
- Model berhasil mencapai target akurasi >75%
- Dropout 0.2 memberikan hasil terbaik (lebih tinggi 2.27%)
- Class imbalance: Class 0 (27 samples) > Class 1 (17 samples)
- Tidak ada overfitting signifikan
- Precision untuk Class 1 masih bisa ditingkatkan

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

## 👥 Kelompok MLOps

- Elilya Octaviani - 122450009
- Rendi Alexander Hutagalung - 122450057
- Izza Lutfia - 122450090
- Tobias David Manogari - 122450091
