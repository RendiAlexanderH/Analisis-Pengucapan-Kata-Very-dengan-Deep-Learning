import os
import numpy as np
import librosa
import mlflow
import mlflow.tensorflow
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout
)

# =============================
# CONFIG
# =============================
DATA_DIR = "Data"          # folder audio
MODEL_DIR = "models"
SAMPLE_RATE = 22050
DURATION = 2.0             # seconds
N_MELS = 128
EPOCHS = 30
BATCH_SIZE = 16

os.makedirs(MODEL_DIR, exist_ok=True)

# =============================
# FEATURE EXTRACTION
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

    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    mel_db = np.expand_dims(mel_db, axis=-1)

    return mel_db

# =============================
# LOAD DATA
# =============================
X = []
y = []

print("Loading audio files...")

for file in os.listdir(DATA_DIR):
    if file.endswith(".wav"):
        file_path = os.path.join(DATA_DIR, file)

        # LABELING SEDERHANA (sesuai kondisi kamu)
        # Semua data satu kelas → binary dummy (1)
        label = 1

        feature = extract_mel(file_path)
        X.append(feature)
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} samples")
print("Input shape:", X.shape)

# =============================
# TRAIN TEST SPLIT
# =============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# MODEL CNN (CLASSIFICATION)
# =============================
model = Sequential([
    Conv2D(32, (3,3), activation="relu",
           input_shape=(N_MELS, X.shape[2], 1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),

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
# MLFLOW AUTOLOG
# =============================
mlflow.tensorflow.autolog()

# =============================
# TRAIN
# =============================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# =============================
# SAVE MODEL (MLFLOW FORMAT)
# =============================
mlflow.tensorflow.save_model(
    model,
    path=os.path.join(MODEL_DIR, "cnn_model")
)

print("Training completed & model saved")
