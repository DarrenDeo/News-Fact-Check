#!/usr/bin/env bash
set -e

# HARUS cocok dengan MODELS_DIR di app.py (relative "models" dari /app)
MODEL_DIR="/app/models"
MODEL_TAR_URL="https://github.com/DarrenDeo/News-Fact-Check/releases/download/v1.0.0-models/models.tar.gz"
TMP_TAR="/tmp/models.tar.gz"

echo "[entrypoint] Cek folder models di $MODEL_DIR ..."

if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
  echo "[entrypoint] $MODEL_DIR kosong atau belum ada. Download dari GitHub Releases..."
  mkdir -p "$MODEL_DIR"

  echo "[entrypoint] Download $MODEL_TAR_URL ..."
  curl -L "$MODEL_TAR_URL" -o "$TMP_TAR"

  echo "[entrypoint] Isi arsip (preview 10 baris):"
  tar -tzf "$TMP_TAR" | head

  echo "[entrypoint] Extract models.tar.gz ke $MODEL_DIR ..."
  # Arsip kamu berisi "<something>/models/bert/..." → buang 2 level: "<something>" dan "models"
  tar -xzf "$TMP_TAR" -C "$MODEL_DIR" --strip-components=2

  rm "$TMP_TAR"
  echo "[entrypoint] Download & extract selesai."

  echo "[entrypoint] Listing isi $MODEL_DIR:"
  ls -R "$MODEL_DIR"
else
  echo "[entrypoint] $MODEL_DIR sudah ada dan tidak kosong. Skip download."
fi

echo "[entrypoint] Menjalankan app.py ..."
exec python app.py
