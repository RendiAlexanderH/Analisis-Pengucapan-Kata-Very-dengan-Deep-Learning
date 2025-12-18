# =====================================================
# TRAINING MODEL PRONUNCIATION "VERY"
# Dataset besar | Folder-based | MLflow
# =====================================================

import os
import numpy as np
import librosa
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
DATA_DIR = "Data/very_dataset"   # SESUAIKAN DENGAN ACTION
CLASSES = ["correct", "incorrect"]

SAMPLE_RATE = 16000
DURATION = 2.0
N_MELS = 128
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

# =========================
# VALIDASI DATASET
# =========================
print("🔍 Checking dataset...")

for cls in CLASSES:
    path = os.path.join(DATA_DIR, cls)
    if not os.path.exists(path):
        raise ValueError(f"Folder tidak ditemukan: {path}")

    files = [f for f in os.listdir(path) if f.endswith(".wav")]
    if len(files) == 0:
        raise ValueError(f"Tidak ada file .wav di {path}")

print("✅ Dataset OK")

# =========================
# HELPER FUNCTIONS
# =========================
def load_audio(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    max_len = int(SAMPLE_RATE * DURATION)

    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

    return np.expand_dims(mel_db, axis=-1)


def collect_filepaths():
    paths, labels = [], []
    for label, cls in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, cls)
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                paths.append(os.path.join(folder, file))
                labels.append(label)
    return paths, labels


# =========================
# LOAD FILE PATHS (NOT AUDIO)
# =========================
file_paths, labels = collect_filepaths()

X_train, X_val, y_train, y_val = train_test_split(
    file_paths,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# =========================
# TF.DATA PIPELINE (RAM AMAN)
# =========================
def tf_loader(path, label):
    audio = tf.numpy_function(
        load_audio,
        [path],
        tf.float32
    )
    audio.set_shape((N_MELS, int(SAMPLE_RATE * DURATION / 512) + 1, 1))
    return audio, label


train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(1000).map(tf_loader).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.map(tf_loader).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# =========================
# MODEL (CNN)
# =========================
def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(N_MELS, None, 1)),

        layers.Conv2D(32, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), activation="relu"),
        layers.GlobalAveragePooling2D(),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(CLASSES), activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =========================
# TRAINING + MLFLOW
# =========================
mlflow.set_experiment("Pronunciation-Very-CNN")

with mlflow.start_run():
    mlflow.log_params({
        "sample_rate": SAMPLE_RATE,
        "duration": DURATION,
        "n_mels": N_MELS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE
    })

    model = build_cnn()
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    val_loss, val_acc = model.evaluate(val_ds)
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("val_accuracy", val_acc)

    mlflow.tensorflow.log_model(
        model,
        artifact_path="cnn_model"
    )

print("🎉 Training selesai & model tersimpan di MLflow")
