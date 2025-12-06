# News Fact Check — Observability + Docker

Aplikasi **Pengecek Fakta Berita Berbasis AI** untuk menganalisis keaslian berita dari **link URL** menggunakan beberapa model Transformer dan **ensemble voting**. Branch **`observability-docker`** menambahkan **containerization** dan **stack observability** agar aplikasi bisa dijalankan konsisten lintas OS sekaligus mudah dipantau performanya.

Versi ini disiapkan untuk kebutuhan demo/presentasi dan assignment yang menuntut:

* Containerization
* Cross-platform testing
* Monitoring & dashboarding

---

## Fitur

### A. Fitur Aplikasi

* Analisis berita melalui URL.
* Multi-model inference:

  * BERT
  * RoBERTa
  * ELECTRA
  * XLNet
* **Ensemble majority voting** untuk prediksi akhir.
* Menampilkan hasil per model + confidence.

### B. Fitur DevOps

* **Dockerfile** untuk build image aplikasi.
* **Docker Compose** untuk menjalankan keseluruhan stack.
* **Persistensi model** menggunakan Docker volume.
* **Setup model dari GitHub Releases** untuk stabilitas, konsistensi versi, dan mengurangi ketergantungan download runtime.

### C. Fitur Observability

* **Prometheus**: scraping dan penyimpanan metrik time-series.
* **Grafana**: visualisasi dashboard.
* **cAdvisor**: metrik container.
* **Node Exporter**: metrik sistem host (paling optimal di Linux/WSL).

---

## Arsitektur Singkat

Satu network Docker Compose berisi service:

* `app` (Flask + Gunicorn)
* `prometheus`
* `grafana`
* `cadvisor`
* `node_exporter`

Prometheus melakukan scrape:

* `app:5000/metrics`
* `cadvisor:8080/metrics`
* `node_exporter:9100/metrics`

Grafana menggunakan Prometheus sebagai data source.

---

## Struktur Proyek

```
News-Fact-Check/
├── app.py
├── frontend/
│   └── index.html
├── src/
│   └── ... (pipeline data & training)
├── monitoring/
│   ├── prometheus.yml
│   └── alert_rules.yml
├── Dockerfile
├── docker-compose.yml
├── setup.sh
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Prasyarat

* Docker Desktop (Windows/macOS) atau Docker Engine (Linux).
* Docker Compose v2.
* Koneksi internet untuk menarik image dan **mengunduh model release (sekali di awal)**.

---

## Quick Start

1. Clone repo dan checkout branch:

```bash
git clone https://github.com/DarrenDeo/News-Fact-Check.git
cd News-Fact-Check
git checkout observability-docker
```

2. Build dan jalankan semua service:

```bash
docker compose up -d --build
```

3. Akses service:

| Service    | URL                                            |
| ---------- | ---------------------------------------------- |
| App        | [http://localhost:5000](http://localhost:5000) |
| Prometheus | [http://localhost:9090](http://localhost:9090) |
| Grafana    | [http://localhost:3000](http://localhost:3000) |
| cAdvisor   | [http://localhost:8080](http://localhost:8080) |

---

## Cara Kerja Model Setup dari GitHub Releases

Saat container `app` start, skrip `setup.sh` akan:

1. Mengecek apakah folder model sudah lengkap di volume.
2. Jika belum ada atau belum lengkap, file model diunduh dari **GitHub Releases** lalu diekstrak.
3. Jika sudah lengkap, proses download akan di-skip.

Konfigurasi default yang umum dipakai:

* `MODEL_RELEASE_TAG=v1.0.0-models`
* `MODEL_ASSET=models.tar.gz`
* `MODELS_DIR=/data/models`

Kamu bisa override via environment variables jika diperlukan:

```bash
MODEL_RELEASE_TAG=v1.0.0-models MODEL_ASSET=models.tar.gz docker compose up -d --build
```

---

## Volume & Persistensi

Compose menyiapkan volume untuk menjaga model tidak hilang saat restart:

* volume models (untuk `/data/models`)
* volume data/cache (jika dikonfigurasi)

Reset total:

```bash
docker compose down -v
```

---

## Verifikasi Aplikasi

1. Buka `http://localhost:5000`.
2. Masukkan link berita.
3. Pastikan:

   * hasil tiap model tampil
   * hasil ensemble tampil

> Catatan: first run bisa lebih lama karena proses download & extract model.

---

## Verifikasi Monitoring

### 1) Pastikan semua container aktif

```bash
docker compose ps
```

### 2) Cek target Prometheus

Buka `http://localhost:9090` → **Status → Targets**.

Target yang idealnya **UP**:

* `news-fact-check-app`
* `cadvisor`
* `node_exporter`

> Di Windows, `node_exporter` bisa saja tidak sekomplet Linux. Untuk demo metrik host, WSL Ubuntu biasanya lebih mulus.

### 3) Query cepat untuk validasi

Di Prometheus Graph atau Grafana Explore:

* Status target:

  ```
  up
  ```

* CPU container:

  ```
  rate(container_cpu_usage_seconds_total[1m])
  ```

* Memory container:

  ```
  container_memory_working_set_bytes
  ```

---

## Grafana Setup

1. Buka `http://localhost:3000`.

2. Login default (jika belum diubah):

   * user: `admin`
   * password: `admin`

3. Tambahkan Prometheus Data Source:

   * Connections → Data sources → Prometheus
   * URL: `http://prometheus:9090`
   * Save & Test

---

## Dashboard yang Direkomendasikan

### A. Container Metrics (cAdvisor)

Panel yang umum dan relevan untuk tugas:

* Container CPU Usage
* Container Memory Working Set
* Top Memory Hungry Containers
* Container Count

### B. App Metrics

Jika app sudah expose `/metrics`:

* Request rate
* Error rate (4xx/5xx)
* Latency (p95/p99) jika histogram tersedia

Jika panel App Metrics masih kosong:

* pastikan endpoint `http://app:5000/metrics` bisa di-scrape oleh Prometheus
* cek job `news-fact-check-app` di `monitoring/prometheus.yml`

---

## Cross-Platform Testing

Stack ini ditujukan agar **image dan compose yang sama** dapat dijalankan di:

* Windows (Docker Desktop)
* macOS
* Linux (termasuk WSL Ubuntu)

Untuk laporan tugas, urutan dokumentasi yang disarankan:

1. Screenshot `docker compose ps`.
2. Screenshot halaman app `localhost:5000` dengan hasil prediksi.
3. Screenshot Prometheus Targets (App/cAdvisor/node_exporter).
4. Screenshot Grafana dashboard Container Metrics.
5. (Opsional) Screenshot Explore dengan query `up`.

---

## Troubleshooting Cepat

### A) `localhost:5000` belum muncul

* Cek logs app:

  ```bash
  docker compose logs -f app
  ```
* Biasanya karena `setup.sh` masih download/extract model.

### B) Prometheus target app DOWN

* Pastikan target di `monitoring/prometheus.yml` menggunakan service name:

  * `app:5000`

### C) Build terasa lambat

* Pastikan `.dockerignore` mengabaikan folder besar seperti:

  * `datasets/`
  * `models/`
  * `venv/` atau `.venv/`

---

## Catatan

* Model yang tersedia di GitHub Releases adalah artifact siap deploy untuk kebutuhan demonstrasi.
* Pipeline training dan evaluasi tetap tersedia di folder `src/` jika ingin retraining.
