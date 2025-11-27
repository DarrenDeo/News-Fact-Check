# News Fact Check

Aplikasi pengecek fakta berita berbasis AI. Masukkan **URL berita**, sistem akan mengekstrak konten artikel, melakukan inferensi dengan empat model Transformer (BERT, RoBERTa, ELECTRA, XLNet), lalu menggabungkannya dengan **Bagging (Ensemble Voting)**. Hasil berisi label **Fakta/Hoax** dan **confidence** per model serta hasil akhir ensemble. Antarmuka web sederhana (HTML/Tailwind/Chart.js) disertakan.

---

## Fitur

* **Analisis via URL**: scraping judul & isi artikel, pembersihan teks, dan klasifikasi.
* **Multi-model + Ensemble**: BERT, RoBERTa, ELECTRA, XLNet + Bagging (mayoritas).
* **Visualisasi**: Tabel F1/Accuracy dan chart perbandingan.
* **Siap jalan lintas platform**: Windows, macOS, Linux/WSL melalui container runtime (Docker/Podman).
* **Otomatis unduh model**: model AI diambil dari GitHub Releases saat container pertama kali dijalankan.

---

## Arsitektur Singkat

```
Frontend (index.html) ──► Flask API (/predict)
      ▲                        │
      └── GET /                │ 4× model (bert/roberta/electra/xlnet)
                                └─ Ensemble (majority voting)
```

* **Backend**: `app.py` (Flask) memuat model dari `/app/models/*`.
* **Frontend**: `frontend/index.html` (Tailwind + Chart.js).
* **Model**: diunduh otomatis sebagai `models.tar.gz` dari GitHub Releases → diekstrak ke `/app/models`.

---

## Prasyarat

* **Docker Desktop** (Windows/macOS) atau **Docker Engine** (Linux/WSL Ubuntu).

Internet dibutuhkan saat **run pertama** untuk mengunduh model.

---

## Cara Pakai (Paling Cepat)

> Image publik tersedia di GHCR:
> `ghcr.io/darrendeo/news-fact-check:latest`

### 1) Jalankan langsung

```bash
docker pull ghcr.io/darrendeo/news-fact-check:latest

# Run pertama akan mengunduh & mengekstrak model ke /app/models
docker run --rm -p 5000:5000 ghcr.io/darrendeo/news-fact-check:latest
```

Buka browser: **[http://localhost:5000](http://localhost:5000)**

### 2) Simpan model agar tidak unduh ulang (disarankan)

**Named volume**:

```bash
docker volume create newsf_models
docker run --rm -p 5000:5000 -v newsf_models:/app/models \
  ghcr.io/darrendeo/news-fact-check:latest
```

**Bind mount** (agar terlihat di folder lokal):

```bash
mkdir -p models
docker run --rm -p 5000:5000 -v "$(pwd)/models:/app/models" \
  ghcr.io/darrendeo/news-fact-check:latest
```

> **Catatan**: jika package GHCR private, lakukan `docker login ghcr.io` dengan token GitHub yang memiliki `read:packages`.

---


## Build & Run dari Kode Sumber (opsional)

Jika ingin membangun image sendiri:

```bash
# dari root repo
docker build -t news-fact-check .
docker run --rm -p 5000:5000 -v newsf_models:/app/models news-fact-check
```

Atau tanpa container (dev lokal):

```bash
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# pastikan folder models/ berisi:
# models/{bert,roberta,electra,xlnet} dengan file tokenizer & model masing-masing
python app.py
# buka http://127.0.0.1:5000
```

---

## Endpoint

* `GET /` → halaman UI.
* `POST /predict`
  Body:

  ```json
  { "url": "https://alamat/berita" }
  ```

  Respons ringkas:

  ```json
  {
    "BERT": {"prediction": "Fakta", "confidence": "96.12%"},
    "RoBERTa": {"prediction": "Fakta", "confidence": "95.33%"},
    "ELECTRA": {"prediction": "Hoax",  "confidence": "91.05%"},
    "XLNet": {"prediction": "Fakta", "confidence": "93.40%"},
    "Bagging (Ensemble)": {"prediction": "Fakta", "confidence": "75.00%"}
  }
  ```

---

## Struktur Proyek (inti)

```
.
├── app.py               # Flask API (scrape, clean, infer 4 model + ensemble)
├── frontend/
│   └── index.html       # UI (Tailwind + Chart.js)
├── entrypoint.sh        # cek /app/models, unduh & ekstrak models.tar.gz, start app
├── requirements.txt     # torch, transformers, flask, requests, bs4, dll.
├── Dockerfile
└── .github/workflows/
    ├── docker-ci.yml    # CI: compile check + docker build
    └── docker-cd.yml    # CD: build & push ke GHCR (latest/branch/sha)
```

> **Model** tidak di-commit. Diunduh otomatis saat run pertama.
> Pastikan struktur hasil ekstraksi:
> `/app/models/{bert,roberta,electra,xlnet}`

---

## CI/CD

* **CI** (`.github/workflows/docker-ci.yml`):
  Trigger pada push/PR ke branch yang ditentukan (mis. `Dockerize-Version`) atau manual.
  Langkah: checkout → Python compile check → **docker build**.

* **CD** (`.github/workflows/docker-cd.yml`):
  Login ke **ghcr.io** memakai `GITHUB_TOKEN`, build image, tag (`latest`, `<branch>`, `<sha>`), **push** ke GHCR:
  `ghcr.io/darrendeo/news-fact-check`.

---

## Troubleshooting

* **Model selalu diunduh ulang**
  Jalankan container dengan volume/mount ke `/app/models` (lihat “Simpan model” di atas).

* **Struktur model salah (mis. muncul `/app/models/models/bert`)**
  Kosongkan volume lama, jalankan ulang agar ekstraksi menghasilkan:
  `/app/models/bert`, `/app/models/roberta`, dst.

* **`/predict` error 500 / hasil kosong**

  * Cek log container: pastikan semua subfolder model ada & terbaca.
  * Pastikan URL berita bisa diakses publik (sebagian situs memblok scraper).
  * Coba ulangi dengan volume baru (untuk memastikan model tidak korup saat ekstraksi).

* **WSL: `permission denied /var/run/docker.sock`**

  * `sudo docker ps` untuk tes.
  * Tambahkan user ke grup docker: `sudo usermod -aG docker $USER` lalu **restart terminal**.
  * Aktifkan **WSL Integration** di Docker Desktop untuk distro yang dipakai.

* **Menjalankan `docker` dari distro `docker-desktop`**
  Gunakan **PowerShell**/**CMD** atau WSL distro kamu (Ubuntu), bukan shell `docker-desktop`.

* **GitHub Actions: `No space left on device`**
  Pipeline sudah disederhanakan (install deps hanya saat `docker build`). Hindari langkah ganda yang menginstal paket berat di host runner.

---


## Lisensi

MIT. Silakan gunakan dan modifikasi dengan tetap menyertakan lisensi.
