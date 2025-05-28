
# Tugas Kelompok CNN - Komparasi Model

## Deskripsi Proyek
Repositori ini merupakan bagian dari tugas mata kuliah yang bertujuan untuk membandingkan performa model CNN dari masing-masing anggota kelompok. Fokus utama proyek ini adalah mengkaji pengaruh strategi seperti augmentasi data, penggunaan pretrained model, dan fine-tuning terhadap performa model klasifikasi citra.

## Anggota Kelompok
- **Ridho Cahyono** (442023611050) – Training model
- **Faiz** (442023611000) – Preprocessing data
- **Firdis Firnadi** (442023611033) – Laporan dan README
- **Zafran Woro** (442023611021) – Testing model
- **Nur Muhammad Ridho** (44202361190) – Visualisasi

## Struktur Repositori
```
.
├── laporan/
│   └── Laporan_Tugas_Kelompok_CNN.pdf
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   ├── testing.py
│   └── visualization.py
├── tugas-4.ipynb
└── README.md
```

## Hasil Komparasi Model
Setiap anggota melakukan eksperimen masing-masing dengan pendekatan yang berbeda. Perbandingan dilakukan berdasarkan:
- **Akurasi**
- **Loss**
- **Confusion Matrix**

Strategi yang diuji mencakup:
- Augmentasi data
- Pretrained model (misal: ResNet)
- Fine-tuning layer akhir

Kesimpulan: model dengan pretrained + fine-tuning memberikan hasil terbaik dibanding model yang dilatih dari awal.

## Cara Menjalankan
1. Clone repositori:
   ```bash
   git clone https://github.com/username/nama-repo.git
   cd nama-repo
   ```

2. Jalankan notebook atau script Python di folder `src/`.

3. Pastikan dependencies seperti TensorFlow, Keras, NumPy, dan Matplotlib telah diinstal.

## License
Proyek ini dibuat untuk keperluan akademik dan tidak untuk penggunaan komersial.
