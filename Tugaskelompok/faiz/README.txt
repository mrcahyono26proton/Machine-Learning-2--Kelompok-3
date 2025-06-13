
Klasifikasi Gambar: Kupu-Kupu vs Tupai

Proyek ini merupakan implementasi model klasifikasi gambar menggunakan PyTorch
untuk membedakan antara dua kelas: kupu-kupu dan tupai. Model dilatih menggunakan 
dataset citra dan dievaluasi dengan metrik akurasi.

Struktur Proyek:
- kupukupu-tupai.ipynb       -> Notebook utama berisi proses training dan evaluasi
- datasets/                  -> Folder berisi dataset (train, val, test)
- models/                    -> (Opsional) Folder untuk menyimpan model yang sudah dilatih
- README.txt                 -> Dokumentasi proyek ini

Arsitektur Model:
Model yang digunakan adalah CNN sederhana (contohnya ResNet18 dari torchvision), 
yang dilatih untuk klasifikasi dua kelas (biner).

Teknologi yang Digunakan:
- Python 3
- PyTorch
- torchvision
- matplotlib, numpy, PIL
- Jupyter Notebook atau PyCharm

Cara Menjalankan:
1. Clone repositori ini:
   git clone https://github.com/nama-kamu/kupukupu-tupai-classification.git
   cd kupukupu-tupai-classification

2. (Opsional) Buat virtual environment:
   python -m venv venv
   source venv/bin/activate      # Untuk Linux/macOS
   venv\Scripts\activate         # Untuk Windows

3. Install semua dependensi:
   pip install -r requirements.txt

4. Jalankan notebook:
   jupyter notebook kupukupu-tupai.ipynb

Hasil:
Setelah pelatihan, model mencapai akurasi sekitar XX% pada data uji (test set).

Contoh Hasil Klasifikasi:
- Gambar 1: Kupu-kupu → benar
- Gambar 2: Tupai → benar

Catatan:
- Pastikan struktur dataset sesuai dengan format:
    datasets/train/kupu
    datasets/train/tupai
    datasets/val/kupu
    datasets/val/tupai
- Dataset bisa diperoleh dari internet atau dikumpulkan secara mandiri.

Lisensi:
Proyek ini dilisensikan di bawah MIT License.
