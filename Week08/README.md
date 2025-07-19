# Prediksi Suhu Per Jam Menggunakan LSTM

## Kelompok 3
- Mohammad Ridho Cahyono
- Faiz Naashih
- FIrdis Firnadi
- Nur Muhammad Ridho Asy Syauqi
- Zafran Woro

### Deskripsi Singkat
Proyek ini bertujuan untuk membangun dan mengevaluasi model Long Short-Term Memory (LSTM) untuk melakukan prediksi suhu per jam (time-series forecasting). Model dilatih menggunakan dataset historis suhu dan diekspor ke format TensorFlow Lite (TFLite) untuk inferensi yang efisien.

### Dataset
Nama File: MLTempDataset1.csv

Jumlah Data: 7056 baris

Link: [Time Series Room Temperature Data](https://www.kaggle.com/datasets/vitthalmadane/ts-temp-1)

Kolom:
- Datetime: Waktu pencatatan data (per jam).
- Hourly_Temp: Suhu yang tercatat dalam satuan Celcius.
- Kualitas Data: Dataset ini bersih dan tidak memiliki nilai yang hilang (missing values).

### Struktur Proyek
- Week08.ipynb: Notebook utama yang berisi seluruh proses dari analisis hingga inferensi.
- MLTempDataset1.csv: Dataset yang digunakan untuk melatih model.
- model_lstm_suhu.tflite: Model final yang telah dikonversi untuk inferensi.
- saved_model_lstm_suhu/: Direktori model yang disimpan dalam format SavedModel TensorFlow.

### Metodologi
Proses pengembangan model dibagi menjadi beberapa tahap utama:
1. **Eksplorasi Data (EDA)**
   - Grafik Garis Suhu per Jam: Menunjukkan tren dan pola suhu dari waktu ke waktu. Terlihat adanya pola siklus harian dan musiman.
   - Histogram Distribusi Suhu: Menunjukkan sebaran frekuensi suhu dalam dataset, yang terlihat mendekati distribusi normal.
2. **Pra-pemrosesan Data**
   - Normalisasi: Data suhu dinormalisasi ke rentang [0, 1] menggunakan MinMaxScaler. Hal ini penting untuk membantu konvergensi model neural network.
   - Windowing: Data diubah menjadi sekuens (jendela) di mana 24 jam data sebelumnya (window_size = 24) digunakan untuk memprediksi suhu pada jam berikutnya.
   - Split Data: Dataset dibagi menjadi data latih (80%) dan data uji (20%) untuk melatih dan mengevaluasi performa model pada data yang belum pernah dilihat sebelumnya.
3. **Pemodelan**

    Model dibangun menggunakan Keras Sequential API dengan arsitektur sederhana namun efektif:
   - LSTM Layer: Satu lapisan LSTM dengan 64 unit untuk menangkap pola temporal dalam data.
   - Dense Layer: Satu lapisan output Dense dengan 1 unit untuk menghasilkan nilai prediksi suhu.
4. **Training**

    Model dilatih dengan konfigurasi berikut:
   - Optimizer: Adam.
   - Loss Function: Mean Squared Error (MSE).
   - Callbacks: EarlyStopping digunakan untuk menghentikan pelatihan jika tidak ada perbaikan pada validation loss setelah 5 epoch, sehingga mencegah overfitting dan menyimpan bobot model terbaik.
   - Epochs: 20 (pelatihan berhenti pada epoch ke-10 berkat EarlyStopping).
5. **Evaluasi Model**

    Performa model dievaluasi pada data uji menggunakan beberapa metrik standar:
   - Root Mean Squared Error (RMSE): Rata-rata akar kuadrat dari selisih error.
   - Mean Absolute Error (MAE): Rata-rata selisih absolut antara nilai aktual dan prediksi.
   - R² (R-squared): Koefisien determinasi yang mengukur seberapa baik model mereplikasi variasi data.
   - Relative Squared Error (RSE): Normalisasi dari total error kuadrat.
    Grafik perbandingan antara nilai aktual dan prediksi menunjukkan bahwa model mampu mengikuti tren suhu dengan sangat baik.
6. **Inferensi**

Model yang telah dilatih disimpan dalam format TensorFlow Lite (.tflite) untuk penggunaan yang lebih ringan dan cepat. Skrip inferensi menunjukkan cara memprediksi suhu untuk 6 jam ke depan berdasarkan data terakhir yang tersedia.
**Contoh Output Inferensi**:
    
    Inference dari model TFLite (waktu sekarang: 2025-07-19 14:28:28):
    Perkiraan suhu pada 2025-07-19 15:28:28 → 23.51 °C
    Perkiraan suhu pada 2025-07-19 16:28:28 → 23.70 °C
    Perkiraan suhu pada 2025-07-19 17:28:28 → 23.92 °C
    Perkiraan suhu pada 2025-07-19 18:28:28 → 24.10 °C
    Perkiraan suhu pada 2025-07-19 19:28:28 → 24.25 °C
    Perkiraan suhu pada 2025-07-19 20:28:28 → 24.37 °C

### Instalasi & Menjalankan
1. Clone Repositori: Buka terminal atau command prompt dan clone repositori ini ke komputermu.
    `git clone https://github.com/mrcahyono26proton/Machine-Learning-2--Kelompok-3.git`
2. Masuk ke Direktori: Pindah ke direktori proyek yang baru saja di-clone.
    `cd Machine-Learning-2--Kelompok-3/Week08`
3. Instal Dependensi: Instal semua library yang dibutuhkan menggunakan pip.
    `pip install pandas numpy matplotlib seaborn scikit-learn tensorflow`
4. Jalankan Notebook: Buka dan jalankan file Week08.ipynb menggunakan Jupyter Notebook atau Jupyter Lab.
    `jupyter notebook Week08.ipynb`

### Struktur File
    .
    ├── MLTempDataset1.csv
    ├── Week08.ipynb
    ├── model_lstm_suhu.tflite
    └── saved_model_lstm_suhu/
        ├── assets/
        ├── fingerprints.pb
        ├── keras_metadata.pb
        └── saved_model.pb
