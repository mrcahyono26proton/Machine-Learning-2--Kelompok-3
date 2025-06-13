
TUGAS MANDIRI 4 - Convolutional Neural Network (CNN)

Kelompok:
- Ridho Cahyono (442023611050) - Training Model
- Faiz (442023611000) - Preprocessing Data
- Firdis Firnadi (442023611033) - Laporan & README
- Zafran Woro (442023611021) - Testing Model
- Nur Muhammad Ridho (44202361190) - Visualisasi

Deskripsi:
Notebook ini berisi implementasi Convolutional Neural Network (CNN) untuk klasifikasi citra menggunakan dataset yang telah disediakan. Proyek ini merupakan bagian dari tugas mandiri ke-4 untuk mata kuliah terkait kecerdasan buatan atau deep learning.

Tahapan dalam notebook:
1. **Import Library**: Mengimpor semua library penting seperti TensorFlow, Keras, NumPy, Matplotlib, dll.
2. **Load Dataset**: Menggunakan dataset citra dari Keras (misalnya CIFAR-10 atau MNIST) dan melakukan normalisasi data.
3. **Preprocessing Data**: Meliputi reshaping, one-hot encoding label, dan membagi dataset menjadi data latih dan uji.
4. **Arsitektur Model CNN**: Membangun model CNN dengan beberapa layer Conv2D, MaxPooling2D, dan Dense.
5. **Training Model**: Melatih model dengan data latih, menggunakan callback seperti EarlyStopping jika diperlukan.
6. **Evaluasi Model**: Mengevaluasi model terhadap data uji dan mencetak akurasi serta loss.
7. **Visualisasi**: Menampilkan grafik akurasi dan loss training vs validation, serta contoh prediksi dari model.

Cara Menjalankan:
1. Jalankan semua sel secara berurutan di Google Colab atau Jupyter Notebook.
2. Pastikan koneksi internet stabil jika menggunakan dataset dari Keras.
3. Hasil akhir akan menunjukkan akurasi model dan visualisasi hasil prediksi.

Catatan:
- Model dapat dikembangkan lebih lanjut dengan menambahkan regularisasi atau augmentasi data untuk menghindari overfitting.
- Evaluasi tambahan dapat dilakukan menggunakan confusion matrix dan classification report dari `sklearn`.
