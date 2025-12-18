# src/train.py
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# ================================
# CONFIG
# ================================
DATA_DIR = "Data"  # folder dataset WAV
SR = 16000         # sample rate
N_MELS = 40        # jumlah filter Mel
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ================================
# LOAD DATA
# ================================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return np.mean(mel_db, axis=1)  # rata-rata per mel band

X = []
y = []

for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)
    if os.path.isdir(label_path):
        for file in os.listdir(label_path):
            if file.endswith(".wav"):
                file_path = os.path.join(label_path, file)
                features = extract_features(file_path)
                X.append(features)
                y.append(label)

X = np.array(X)
y = np.array(y)

# ================================
# CEK DATA
# ================================
if len(X) == 0:
    raise ValueError(f"No audio files found in {DATA_DIR}. Please check your dataset folder!")

# ================================
# SPLIT DATA
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ================================
# SCALING
# ================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================
# TRAINING SVM
# ================================
clf = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
clf.fit(X_train, y_train)

# ================================
# EVALUASI
# ================================
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print(classification_report(y_test, y_pred))

# ================================
# SIMPAN MODEL
# ================================
import joblib
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Model dan scaler berhasil disimpan di folder 'models/'")
