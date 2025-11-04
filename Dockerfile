# Image resmi python 3.9 dengan slim yang berarti lebih ringan.
FROM python:3.9-slim

# Perintah ini untuk membuat folder '/app/' di container. dan menjadi direktori kerja utama.
WORKDIR /app

# Salin file requirements.txt ke dalam direktori kerja ('/app') container.
COPY requirements.txt .

# menjalankan perintah pip install untuk menginstal semua dependensi yang tercantum dalam requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Salin file app.py ke direktori kerja container.
COPY app.py .

# Salin seluruh folder 'frontend' dan isinya ke dalam container.
COPY ./frontend ./frontend

# Salin seluruh folder 'models' yang berisi semua model AI.
COPY ./models ./models

# memberitahu Docker bahwa container akan 'mendengarkan' di port 5000 saat berjalan.
EXPOSE 5000

# perintah yang akan dieksekusi secara otomatis saat container dimulai.
CMD ["python", "app.py"]