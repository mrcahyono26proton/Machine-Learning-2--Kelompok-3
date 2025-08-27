# Machine Translation: Membandingkan Arsitektur RNN-Attention dan Transformer

Dibuat oleh Mohammad Ridho Cahyono

## Deskripsi Proyek
Proyek ini merupakan implementasi dan perbandingan dua arsitektur deep learning terkemuka untuk tugas Penerjemahan Mesin (Machine Translation), yaitu menerjemahkan teks dari bahasa Inggris ke bahasa Indonesia. Kami melatih dan mengevaluasi model RNN-Attention sebagai baseline dan model Transformer sebagai arsitektur state-of-the-art untuk memahami perbedaan kinerja dan efisiensinya.

Model dilatih dan dievaluasi menggunakan metrik standar seperti Cross-Entropy Loss, Perplexity (PPL), dan skor BLEU.

## Sumber Dataset
Dataset yang digunakan dalam proyek ini adalah pasangan kalimat bilingual Inggris-Indonesia yang diambil dari proyek Tatoeba.
- Link Dataset: https://www.manythings.org/anki/

## Cara Menjalankan Proyek
Ikuti langkah-langkah di bawah ini untuk menjalankan proyek dari awal.

1. Klon Repositori
Buka terminal atau command prompt dan jalankan perintah berikut untuk mengklon repositori GitHub ini.

        git clone https://github.com/nama-pengguna/nama-repositori.git
        cd nama-repositori

2. Instal Dependensi
Pastikan Anda memiliki Python 3.7+ terinstal. Kemudian, instal semua pustaka yang diperlukan menggunakan pip.

        pip install torch numpy matplotlib nltk

3. Unduh Dataset
Unduh file dataset ind.txt dari tautan yang disediakan di atas dan letakkan di dalam folder data/ind-eng/.

4. Persiapan Data
Jalankan skrip data_preparation.py untuk membersihkan data, membangun kamus (kosakata), dan membagi dataset menjadi set pelatihan, validasi, dan pengujian.

        python data_preparation.py

    Skrip ini akan membuat file-file berikut: en_vocab.json, id_vocab.json, dan processed_data.json.

5. Melatih Model
Ada dua model yang dapat Anda latih:

    - Melatih Model RNN-Attention (Baseline)
    Jalankan skrip main.py untuk melatih model dasar RNN dengan attention.

            python main.py

    - Melatih Model Transformer
    Jalankan skrip train_transformer.py untuk melatih model Transformer.

            python train_transformer.py

6. Hasil
Setelah pelatihan selesai, skrip akan menampilkan metrik kinerja (Loss, PPL, BLEU) pada set pengujian dan beberapa contoh terjemahan. Grafik kurva pelatihan (loss dan PPL) juga akan disimpan dalam file .png di direktori proyek.