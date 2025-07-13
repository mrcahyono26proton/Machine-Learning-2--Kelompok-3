# Implementasi Generative Adversarial Network (GAN)

## Deskripsi

Proyek ini merupakan tugas kelompok dalam mata kuliah *Deep Learning* dengan tujuan memahami dan mengimplementasikan arsitektur dasar dari **Generative Adversarial Network (GAN)**. Model dilatih untuk menghasilkan gambar dari dataset sederhana, dengan memanfaatkan dua arsitektur utama: **Generator** dan **Discriminator**.

Dataset yang digunakan: **[[MNIST/FashionMNIST/CIFAR-10](https://www.kaggle.com/datasets/gpreda/chinese-mnist)]*  
Arsitektur yang digunakan: **DCGAN-style GAN**

---

## Tujuan

- Memahami konsep dasar Generative Adversarial Network.
- Mengimplementasikan GAN menggunakan PyTorch.
- Melatih dan mengamati perkembangan model melalui visualisasi gambar hasil generator.
- Melatih kerja sama tim dalam menyusun dan mendokumentasikan proyek deep learning.

---

## Anggota Kelompok

| Nama Lengkap | NIM | Kontribusi |
|--------------|-----|------------|
| Mohammad Ridho Cahyono | Implementasi Generator 
| Muhammad Faiz Nashih | Visualisasi 
| Firdis Firnadi | Dokumentasi |
| Nur Muhammad Ridho Asy Syauqi | Training |
| Zafran Woro | Debugging |

---

## Cara Menjalankan Proyek

1. Clone repositori:
   ```bash
   git clone https://github.com/mrcahyono26proton/Machine-Learning-2--Kelompok-3/tree/main/Week07.git
   cd Week08

2. Install dependensi:
    pip install -r requirements.txt

3. Jalankan notebook:
    Buka gan_kelompok-3.ipynb di Jupyter Notebook atau JupyterLab

    Jalankan sel-selnya secara berurutan

    Hasil gambar generator akan otomatis tersimpan di folder generated_images/.