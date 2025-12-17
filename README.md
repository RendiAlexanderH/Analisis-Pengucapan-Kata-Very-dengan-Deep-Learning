# Dataset Pengucapan Kata "Very"

Dataset ini digunakan untuk penelitian dan tugas besar
**Analisis Pengucapan Kata "Very" Menggunakan Deep Learning**.

---

## 📂 Struktur Folder

data/
├── raw/
│   └── audio_asli.wav
├── very/
│   ├── very_1.wav
│   ├── very_2.wav
│   └── ...
├── data_label.csv
└── README.md

---

## 🎙️ Deskripsi Data

Dataset terdiri dari potongan audio kata **"very"**
yang diekstraksi secara otomatis dari rekaman suara
menggunakan model **Whisper Timestamped**.

Setiap file audio:
- Format: `.wav`
- Durasi: ±0.2–0.5 detik
- Satu file berisi satu pengucapan kata *very*

---

## 🏷️ Label Data

Pelabelan dilakukan otomatis menggunakan
**Levenshtein Distance** antara hasil transkripsi dan kata target `"very"`.

| Label | Keterangan |
|------|-----------|
| 1 | Pengucapan benar |
| 0 | Pengucapan salah |

Detail label tersimpan dalam `data_label.csv`.

---

## ⚙️ Pra-pemrosesan

1. Transkripsi audio dengan Whisper
2. Pemotongan kata target berdasarkan timestamp
3. Ekstraksi Mel Spectrogram (128×128)
4. Data augmentation (time shifting & noise)

---

## 📌 Catatan

- Dataset ini **khusus untuk keperluan akademik**
- File audio besar tidak disertakan langsung di GitHub jika melebihi batas
