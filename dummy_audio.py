import numpy as np
from scipy.io.wavfile import write
fs = 16000
y = np.zeros(fs)  # 1 detik audio kosong
write("Data/dummy.wav", fs, y.astype(np.int16))
