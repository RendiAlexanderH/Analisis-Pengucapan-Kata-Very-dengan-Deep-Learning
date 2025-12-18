import os
import numpy as np
import librosa
import mlflow
import mlflow.tensorflow
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# =============================
# KONFIGURASI
# =============================
DATA_DIR = "Data"
MODEL_DIR = "models"
SAMPLE_RATE = 22050
DURATION = 2.0
N_MELS = 128
EPOCHS = 30
BATCH_SIZE = 16

os.makedirs(MODEL_DIR, exist_ok=True)

mlflow.set_experiment("Analisis_Pengucapan_Very")

# =============================
# EKSTRAKSI FITUR
# =============================
def extract_mel(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    max_len = int(SAMPLE_RATE * DURATION)
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

    return np.expand_dims(mel_db, axis=-1)

# =============================
# LOAD DATA
# =============================
X, y = [], []

print("Loading audio files...")

for label_name in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label_name)
    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if file.endswith(".wav"):
            file_path = os.path.join(label_path, file)
            feature = extract_mel(file_path)
            X.append(feature)
            y.append(label_name)

X = np.array(X)
y = np.array(y)

# Encode label
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print(f"Total data: {len(X)}")
print("Shape X:", X.shape)

# =============================
# SPLIT DATA
# =============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# MODEL CNN
# =============================
model = Sequential([
    Conv2D(32, (3,3), activation="relu",
           input_shape=(N_MELS, X.shape[2], 1)),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =============================
# TRAINING + MLFLOW
# =============================
mlflow.tensorflow.autolog()

with mlflow.start_run():
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    model.save(os.path.join(MODEL_DIR, "model_very.keras"))

print("Training selesai, model tersimpan.")
