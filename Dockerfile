# 1. Gunakan base image Python yang stabil
FROM python:3.11-slim

# 2. Set direktori kerja di dalam container
WORKDIR /app

# 3. Instal dependensi sistem yang diperlukan (git dan git-lfs) SEBAGAI ROOT
RUN apt-get update && apt-get install -y git git-lfs && git-lfs install

# 4. Salin file requirements terlebih dahulu untuk caching
COPY requirements.txt .

# 5. Instal semua pustaka Python yang diperlukan
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# 6. Salin semua file proyek lainnya
COPY . .

# 7. Jadikan skrip setup bisa dieksekusi
RUN chmod +x setup.sh

# 8. Beri tahu Docker port mana yang akan didengarkan
EXPOSE 7860

# 9. Perintah untuk menjalankan aplikasi
# Jalankan setup.sh untuk mengunduh model, LALU jalankan server Gunicorn.
# Semua akan berjalan sebagai root, yang akan menyelesaikan masalah izin.
CMD ["/bin/bash", "-c", "./setup.sh && gunicorn --bind 0.0.0.0:7860 --timeout 600 app:app"]
