## Pengecek Fakta Berita Berbasis AI

Aplikasi ini dirancang untuk menganalisis keaslian sebuah berita yang disediakan melalui link URL. Sistem ini menggunakan lima metode AI yang berbeda untuk memberikan prediksi, yang kemudian digabungkan untuk menghasilkan keputusan akhir yang lebih andal.

## Fitur Utama

-   **Analisis Berita via Link:** Mengekstrak konten artikel secara otomatis dari link URL yang diberikan pengguna.
-   **Multi-Model Analysis:** Menggunakan 5 metode AI untuk analisis yang komprehensif:
    1.  **BERT**
    2.  **RoBERTa**
    3.  **ELECTRA**
    4.  **XLNet**
    5.  **Bagging (Ensemble Voting)**
-   **Hasil Terperinci:** Menampilkan prediksi (Fakta/Hoax) dan skor kepercayaan dari setiap model secara individual.
-   **Keputusan Akhir Ensemble:** Memberikan kesimpulan akhir berdasarkan voting mayoritas dari semua model untuk meningkatkan akurasi dan keandalan.
-   **Dasbor Kinerja:** Menyajikan visualisasi perbandingan F1-score dan akurasi dari setiap model dalam bentuk tabel dan diagram.

---

## Arsitektur & Alur Kerja Proyek

Proyek ini dibangun dengan alur kerja yang terstruktur, mulai dari persiapan data mentah hingga penyajian hasil analisis melalui aplikasi web.

**Penjelasan Alur Kerja:**

1.  **Persiapan Data:** Berbagai dataset mentah dari berbagai kategori digabungkan, dibersihkan, dan diproses oleh skrip `src/preprocess.py` menjadi satu set data terpadu yang siap digunakan.
2.  **Pelatihan Model:** Data terpadu dibagi menjadi set data latih, validasi, dan uji. Data latih dan validasi digunakan untuk melakukan *fine-tuning* pada empat model Transformer (BERT, RoBERTa, ELECTRA, XLNet). Setiap model yang telah dilatih disimpan ke dalam folder `models/`.
3.  **Evaluasi & Ensemble:** Menggunakan data uji, skrip `src/evaluate.py` memuat keempat model terlatih untuk mengevaluasi kinerjanya. Prediksi dari keempat model digabungkan menggunakan metode *majority voting* untuk menghasilkan prediksi **Bagging (Ensemble)**. Laporan kinerja, tabel perbandingan, dan visualisasi dihasilkan pada tahap ini.
4.  **Aplikasi Web:** Sebuah server backend (Flask) memuat semua model terlatih ke dalam memori. Antarmuka pengguna (frontend) menerima input link berita dari pengguna, mengirimkannya ke backend untuk dianalisis, dan kemudian menampilkan hasil prediksi dari setiap model serta hasil akhir dari ensemble.

---

## Visualisasi Data & Hasil

Skrip `evaluate.py` secara otomatis menghasilkan beberapa visualisasi untuk membantu memahami dataset dan kinerja model. Semua gambar disimpan di dalam folder `results/`, yang dapat Anda periksa setelah menjalankan evaluasi. Contoh visualisasi yang dihasilkan meliputi:

* **Distribusi Kelas:** Menunjukkan perbandingan jumlah berita Fakta dan Hoax.
* **Distribusi Panjang Teks:** Histogram yang menunjukkan sebaran jumlah kata dalam berita.
* **Perbandingan Metrik Kinerja:** Diagram batang yang membandingkan metrik kunci dari kelima metode AI.
* **Confusion Matrix:** Dihasilkan untuk setiap model untuk melihat secara detail performa klasifikasinya.

## Struktur Proyek

```
NEWS-FACK-CHECK/
├── app.py                  # Backend server Flask
├── datasets/               # Folder untuk semua dataset (mentah)
├── frontend/
│   └── index.html          # Antarmuka pengguna (UI) aplikasi
├── models/                 # (Folder ini diabaikan oleh Git, dibuat saat training)
├── results/                # Hasil evaluasi, diagram, dan laporan
├── src/                    # Folder untuk semua skrip Python
│   ├── preprocess.py
│   ├── train_bert.py
│   ├── ... (skrip training lainnya) ...
│   └── evaluate.py
├── .gitignore              # Mengabaikan file/folder yang tidak perlu di-upload
├── requirements.txt        # Daftar pustaka Python yang diperlukan
└── README.md               # File ini
```

## Arsitektur Model AI

Proyek ini memanfaatkan beberapa model Transformer pre-trained yang telah di-fine-tune pada dataset berita berbahasa Indonesia.

| Metode              | Model Pre-trained yang Digunakan                   | Keterangan                                      |
| ------------------ | -------------------------------------------------- | ----------------------------------------------- |
| **BERT** | `indobenchmark/indobert-base-p2`                   | Model BERT yang dioptimalkan untuk Bahasa Indonesia.  |
| **RoBERTa** | `cahya/roberta-base-indonesian-522M`             | Varian RoBERTa untuk Bahasa Indonesia.            |
| **ELECTRA** | `google/electra-base-discriminator`                | Model multilingual yang efisien dan berkinerja tinggi. |
| **XLNet** | `xlnet-base-cased`                                 | Model multilingual dengan arsitektur autoregressive. |
| **Bagging** | Ensemble Voting                                    | Menggabungkan prediksi dari 4 model di atas.      |

---

## Instalasi & Pengaturan

Untuk menjalankan proyek ini di mesin lokal Anda, ikuti langkah-langkah berikut.

### Prasyarat

-   Python 3.8 atau versi lebih baru.
-   `pip` dan `venv` untuk manajemen pustaka.
-   Git untuk mengkloning repositori.
-   **GPU NVIDIA dengan CUDA:** Sangat direkomendasikan untuk mempercepat proses pelatihan model.

### Langkah-langkah Instalasi

1.  **Clone Repositori**
    ```bash
    git clone https://github.com/DarrenDeo/News-Fact-Check.git
    cd News-Fact-Check
    ```

2.  **Buat dan Aktifkan Virtual Environment**
    ```bash
    # Buat venv
    python -m venv venv

    # Aktifkan venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Instal Semua Pustaka yang Diperlukan**
    ```bash
    pip install -r requirements.txt
    ```

## Cara Penggunaan

Proyek ini memiliki alur kerja dari persiapan data hingga menjalankan aplikasi web.

### Langkah 1: Siapkan Dataset Mentah

-   Karena dataset berukuran besar, Anda perlu mengunduhnya secara manual dari sumber yang disebutkan dan menempatkannya di dalam folder `datasets/` dengan struktur folder yang sesuai (misalnya, `datasets/politics/`, `datasets/sports/`, dll.).

### Langkah 2: Pra-pemrosesan Data

-   Skrip ini akan menggabungkan dataset mentah Anda, membersihkan teks, dan membaginya menjadi set data latih, validasi, dan uji.
-   Pastikan konfigurasi di dalam `src/preprocess.py` sudah sesuai dengan file Anda.
-   Jalankan dari direktori `src/`:
    ```bash
    python preprocess.py
    ```

### Langkah 3: Latih Model AI

-   Jalankan setiap skrip pelatihan satu per satu dari direktori `src/`. Proses ini akan memakan waktu dan membutuhkan GPU.
    ```bash
    python train_bert.py
    python train_roberta.py
    python train_electra.py
    python train_xlnet.py
    ```
-   Model yang berhasil dilatih akan disimpan di dalam folder `models/`.

### Langkah 4: Evaluasi Semua Model

-   Setelah semua model dilatih, jalankan skrip evaluasi.
-   Jalankan dari direktori `src/`:
    ```bash
    python evaluate.py
    ```
-   Hasil evaluasi akan disimpan di dalam folder `results/`.

### Langkah 5: Jalankan Aplikasi Web

-   Arahkan terminal Anda ke **direktori root** proyek.
-   Jalankan server Flask:
    ```bash
    python app.py
    ```
-   Tunggu hingga server berjalan dan semua model dimuat.
-   Buka browser Anda dan kunjungi alamat: **`http://127.0.0.1:5000`**.
-   Anda sekarang dapat memasukkan link berita untuk dianalisis.

---
