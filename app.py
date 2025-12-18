import os
import librosa
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow

# =============================
# CONFIG
# =============================
DATA_DIR = "data/audio"
SAMPLE_RATE = 16000
N_MELS = 128
MAX_LEN = 63      # PENTING → konsisten!
BATCH_SIZE = 16
EPOCHS = 30

# =============================
# LOAD AUDIO → MEL SPECTROGRAM
# =============================
def load_audio_files(folder):
    X = []

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)

            y, sr = librosa.load(path, sr=SAMPLE_RATE)

            mel = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=N_MELS
            )

            mel = librosa.power_to_db(mel, ref=np.max)

            # 🔒 FIX PANJANG
            if mel.shape[1] < MAX_LEN:
                pad = MAX_LEN - mel.shape[1]
                mel = np.pad(mel, ((0,0),(0,pad)))
            else:
                mel = mel[:, :MAX_LEN]

            X.append(mel)

    X = np.array(X)
    X = X[..., np.newaxis]  # (N, 128, 63, 1)
    return X


print("Loading audio files...")
X = load_audio_files(DATA_DIR)
print(f"Loaded {X.shape[0]} samples")

# =============================
# MODEL: CONV AUTOENCODER (FIX SHAPE)
# =============================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(N_MELS, MAX_LEN, 1)),

    tf.keras.layers.Conv2D(16, (3,3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2,2), padding="same"),

    tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2,2), padding="same"),

    tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),

    tf.keras.layers.UpSampling2D((2,2)),
    tf.keras.layers.Conv2D(16, (3,3), activation="relu", padding="same"),

    tf.keras.layers.UpSampling2D((2,2)),
    tf.keras.layers.Conv2D(1, (3,3), activation="sigmoid", padding="same"),
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

# =============================
# MLFLOW
# =============================
mlflow.tensorflow.autolog()

with mlflow.start_run():
    model.fit(
        X, X,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2
    )

    model.save("model")
