import librosa
import numpy as np
import os

def extract_mel(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith(".wav"):
            mel = extract_mel(os.path.join(input_dir, file))
            np.save(os.path.join(output_dir, file.replace(".wav", ".npy")), mel)

if __name__ == "__main__":
    process_folder("data/raw", "data/processed")
