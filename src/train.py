import os
import librosa
import numpy as np
import tensorflow as tf
import mlflow.tensorflow

# ======================
# CONFIG
# ======================
DATA_DIR = "Data/audio"
SR = 16000
DURATION = 2.0
N_MELS = 128
EPOCHS = 30
BATCH_SIZE = 16

# ======================
# LOAD DATA
# ======================
def load_audio():
    X = []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]

    if not files:
        raise ValueError(f"No audio files found in {DATA_DIR}")

    for f in files:
        path = os.path.join(DATA_DIR, f)
        y, _ = librosa.load(path, sr=SR)

        max_len = int(SR * DURATION)
        y = y[:max_len]
        y = np.pad(y, (0, max_len - len(y)))

        mel = librosa.feature.melspectrogram(
            y=y, sr=SR, n_mels=N_MELS
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = (mel - mel.min()) / (mel.max() - mel.min())

        X.append(mel)

    X = np.array(X)[..., np.newaxis]
    return X

# ======================
# TRAIN
# ======================
print("Loading audio files...")
X = load_audio()
print(f"Loaded {X.shape[0]} samples")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(N_MELS, X.shape[2], 1)),
    tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(8, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2DTranspose(8, 3, strides=2, padding="same"),
    tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding="same"),
    tf.keras.layers.Conv2D(1, 3, activation="sigmoid", padding="same")
])

model.compile(
    optimizer="adam",
    loss="mse"
)

mlflow.tensorflow.autolog()

model.fit(
    X, X,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

mlflow.tensorflow.log_model(model, "autoencoder_model")

print("Training completed successfully")
