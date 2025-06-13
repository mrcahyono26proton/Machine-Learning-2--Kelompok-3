# 🚗🏍️ Klasifikasi Mobil vs Motor dengan Transfer Learning (MobileNetV2)

## 📚 Deskripsi Proyek
Tugas ini merupakan bagian dari mata kuliah pembelajaran mesin, di mana saya melakukan klasifikasi dua jenis objek: **table (meja)** dan **kursi (chair)** menggunakan pendekatan **transfer learning** dengan model pretrained **MobileNetV2** dari PyTorch.

Dataset diambil dari Kaggle dan terdiri dari 200 gambar (100 gambar meja, 100 gambar kursi) yang saya split menjadi data pelatihan dan pengujian dengan rasio 80:20.

---

## 🗂️ Struktur Proyek

📁 car-vs-bike-transfer-learning
├── 📄 README.md
├── 📄 tugas_4_Nur_Ridho.ipynb
└── 📁 dataset
	├── train
		│ ├── table
		│ └── chair
	└── test
		├── table
		└── chair

---

## 🔍 Dataset
- **Sumber**: https://images.cv/dataset/table-image-classification-dataset, https://images.cv/dataset/chair-image-classification-dataset
- **Total digunakan**: 700 gambar
- **Split data**:
  - 400 gambar untuk pelatihan (200 mobil, 200 motor)
  - 300 gambar untuk pengujian (150 mobil, 150 motor)

---

## 🧠 Model & Arsitektur

- Pretrained: **MobileNetV2** (ImageNet)
- Layer akhir diubah untuk klasifikasi 2 kelas
- Frozen semua layer pretrained
- Fine-tuning dilakukan pada classifier baru
- Optimizer: `Adam`
- Loss Function: `CrossEntropyLoss`
- Epoch: 10

---

## 📊 Hasil Evaluasi

- **Akurasi**: ~98% pada data test
- **Visualisasi**:
  - Training loss per epoch
  - Confusion Matrix
  - Classification Report

---

## 🛠️ Instalasi & Cara Menjalankan

1. Clone repositori ini:
   git clone https://github.com/username/car-vs-bike-transfer-learning.git
   cd car-vs-bike-transfer-learning

Install dependency:

pip install -r requirements.txt
Jalankan Notebook:

jupyter notebook tugas_4_mandiri_lengkap.ipynb
Pastikan folder dataset berada di struktur direktori seperti di atas (dataset/train/ dan dataset/test/)

🧑‍💻 Kontributor
Nama: Nur Muhammad Ridho
NIM: 442023611036
Mata Kuliah: Machine Learning 2