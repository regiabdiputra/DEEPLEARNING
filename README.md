# 🤖 Hybrid Attention LSTM: Klasifikasi Spam Email Mahasiswa ITERA

## 👩‍💻 Anggota Kelompok
1. Natasya Ega Lina Marbun - 122450024
2. Eksanty F Sugma Islamiaty - 122450001
3. Muhammad Regi Abdi Puta Amanta - 122450031

## 🚀 Projek *Deep Learning*

Proyek ini menyajikan implementasi model **Hybrid Attention Long Short-Term Memory (LSTM)**, sebuah pendekatan *deep learning* canggih untuk klasifikasi spam pada lingkungan akademik, khususnya email mahasiswa Institut Teknologi Sumatera (ITERA). Peningkatan email spam dapat mengganggu efektivitas komunikasi dan berpotensi menimbulkan risiko keamanan digital.

Model Hybrid ini dirancang untuk mengatasi keterbatasan metode tradisional dengan menggabungkan dua jenis fitur secara komprehensif: fitur sekuensial (teks) dan fitur struktural (numerik/metadata).

## ✨ Metodologi dan Arsitektur Utama

Model yang diusulkan adalah arsitektur dual-input yang memanfaatkan kemampuan LSTM dan *Temporal Attention Mechanism* untuk pemrosesan teks, serta lapisan **Dense** untuk fitur numerik.

### Arsitektur Model: Hybrid Attention LSTM

| Cabang | Tujuan | Komponen Utama |
| :--- | :--- | :--- |
| **Cabang Teks** | Memproses urutan kata dan menangkap dependensi sekuensial. | `Embedding` (256 Dimensi) → `LSTM` (32 Unit) → `Attention` |
| **Cabang Numerik** | Memproses 7 fitur metadata (misalnya, jumlah URL, panjang pesan, domain). | `Dense` (32 Unit) |
| **Klasifikasi** | Menggabungkan hasil dan menghasilkan probabilitas biner (Spam/Ham). | `Concatenate` → `Dense` → `Dropout` (rate 0.5) → `Sigmoid` |

**Total Parameter yang Dapat Dilatih (Trainable Params):** 1.420.737.

### 💡 Keunggulan Mekanisme Attention

Kami menggunakan *temporal attention mechanism* untuk memungkinkan model memberikan bobot yang berbeda pada setiap kata dalam urutan teks. Ini memastikan model dapat memprioritaskan kata-kata kunci yang paling relevan—bahkan jika kata tersebut jarang muncul—sehingga meningkatkan kinerja klasifikasi.

## 📊 Hasil Kinerja Model

Model Hybrid Attention LSTM menunjukkan performa klasifikasi yang sangat baik pada data uji:

| Metrik | Hasil |
| :--- | :--- |
| **AUC Score** | **0.9574** (Sangat Baik) |
| **Akurasi Uji (Test Accuracy)** | **0.8833** |
| **F1-Score (Weighted Avg)** | 0.8836 |
| **Loss Uji (Test Loss)** | 0.2815 |

Model ini juga menunjukkan kemampuan generalisasi yang baik, ditunjukkan dari kurva *Training* dan *Validation* yang menunjukkan peningkatan akurasi konsisten hingga *early stopping* tercapai, tanpa mengalami *overfitting* signifikan.

## 🔍 Interpretasi Model (Attention Heatmap)

Visualisasi bobot perhatian (attention weights) membuktikan bahwa model berhasil mengidentifikasi pola kunci:

*   **Pada Kelas Spam:** Bobot perhatian yang lebih tinggi diberikan pada kata-kata yang mengandung unsur desakan atau informasi seperti **“upcoming”**, **“activities”**, **“due”**, dan **“assignment”**.
*   **Pada Kelas Ham:** Bobot perhatian lebih tinggi pada kata-kata yang berkaitan dengan pengumpulan dan penugasan akademik, seperti **“submitted”**, **“assignment”**, **“tugas”**, dan **“submission”**.

## 📁 Struktur Data dan Pra-pemrosesan

### Data
Data yang digunakan adalah 600 email primer mahasiswa ITERA yang terbagi menjadi 347 email spam dan 253 email ham.

### Fitur
Model ini memanfaatkan gabungan fitur Teks dan 7 Fitur Numerik, yang mencakup atribut bawaan data dan atribut tambahan yang diekstrak:
*   `num_urls`
*   `num_exclaim`
*   `has_attachment`
*   `body_len`
*   `from_domain_enc`
*   **`num_special_chars`** (Ditambahkan)
*   **`avg_word_len`** (Ditambahkan)

### Tahap Pra-pemrosesan Kunci
1.  **Penggabungan:** Kolom `subject` dan `body` digabungkan untuk konteks yang lengkap.
2.  **Pembersihan Teks:** Menghapus URL, angka, karakter khusus, dan menormalisasi huruf kecil.
3.  **Stopword Removal & Stemming:** Menggunakan *Stopword Removal* gabungan (Indonesia & Inggris) dan *stemming* Sastrawi untuk menormalkan bentuk kata.
4.  **Tokenisasi & Padding:** Teks diubah menjadi urutan 100 token bilangan bulat (`maxlen=100`).
5.  **Scaling Numerik:** Fitur numerik dinormalisasi menggunakan `MinMaxScaler` agar berada pada rentang 0 hingga 1.

## 💻 Cara Akses dan Menjalankan Proyek

Anda dapat mengakses *source code* dan notebook Colab yang digunakan untuk eksperimen ini:

[**Tautan Akses Streamlit**](https://spam004.streamlit.app/)

[**Tautan Akses Colab**](https://colab.research.google.com/drive/1uopNZNEwqBmecofLgWGi1Mr4KKwa1P4D?usp=sharing)
