#!/bin/bash

# Script ini akan berjalan sekali saat Space dibangun.
# Tujuannya adalah mengunduh semua model yang diperlukan ke persistent storage.

echo "=================================================="
echo "Memulai proses setup: Mengunduh semua model AI..."
echo "=================================================="

# Direktori di dalam persistent storage untuk menyimpan model
# Path ini akan menjadi /data/models di dalam container Docker Anda
MODEL_STORAGE_PATH="/data/models"

# Buat direktori jika belum ada
mkdir -p $MODEL_STORAGE_PATH

# Gunakan huggingface-cli untuk mengunduh setiap model
# Perintah ini akan mengunduh model dan menyimpannya di path yang kita tentukan

echo "Mengunduh BERT..."
huggingface-cli download indobenchmark/indobert-base-p2 --local-dir $MODEL_STORAGE_PATH/bert --local-dir-use-symlinks False

echo "Mengunduh RoBERTa..."
huggingface-cli download cahya/roberta-base-indonesian-522M --local-dir $MODEL_STORAGE_PATH/roberta --local-dir-use-symlinks False

echo "Mengunduh ELECTRA..."
huggingface-cli download google/electra-base-discriminator --local-dir $MODEL_STORAGE_PATH/electra --local-dir-use-symlinks False

echo "Mengunduh XLNet..."
huggingface-cli download xlnet-base-cased --local-dir $MODEL_STORAGE_PATH/xlnet --local-dir-use-symlinks False

echo "=================================================="
echo "Semua model telah berhasil diunduh ke $MODEL_STORAGE_PATH"
echo "=================================================="

