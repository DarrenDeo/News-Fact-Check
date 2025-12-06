#!/usr/bin/env bash
set -euo pipefail

echo "=================================================="
echo "Setup model dari GitHub Releases"
echo "=================================================="

REPO="DarrenDeo/News-Fact-Check"

MODEL_RELEASE_TAG="${MODEL_RELEASE_TAG:-v1.0.0-models}"
MODEL_ASSET="${MODEL_ASSET:-models.tar.gz}"

MODELS_DIR="${MODELS_DIR:-/data/models}"

TMP_TAR="/tmp/${MODEL_ASSET}"
URL="https://github.com/${REPO}/releases/download/${MODEL_RELEASE_TAG}/${MODEL_ASSET}"

REQUIRED_DIRS=("bert" "roberta" "electra" "xlnet")

is_models_complete() {
  for d in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "${MODELS_DIR}/${d}" ]; then
      return 1
    fi
  done
  return 0
}

echo "[setup] MODELS_DIR = ${MODELS_DIR}"
echo "[setup] Release URL = ${URL}"

mkdir -p "${MODELS_DIR}"

if is_models_complete; then
  echo "[setup] Model sudah lengkap. Skip download."
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "[setup] ERROR: curl tidak ditemukan di image."
  echo "[setup] Tambahkan instalasi curl di Dockerfile."
  exit 1
fi

echo "[setup] Model belum lengkap / belum ada. Downloading..."
curl -fL "${URL}" -o "${TMP_TAR}"

echo "[setup] Download selesai. Mulai extract..."

# Coba beberapa kemungkinan struktur path di tar:
# - ./models/bert/...  -> strip 2
# - models/bert/...    -> strip 1
# - bert/...           -> strip 0
EXTRACT_SUCCESS=0

for STRIP in 2 1 0; do
  echo "[setup] Trying extract with --strip-components=${STRIP}"

  rm -rf "${MODELS_DIR:?}/"*
  tar -xzf "${TMP_TAR}" -C "${MODELS_DIR}" --strip-components="${STRIP}" || true

  if is_models_complete; then
    echo "[setup] ✅ Extract OK dengan strip=${STRIP}"
    EXTRACT_SUCCESS=1
    break
  fi
done

rm -f "${TMP_TAR}"

echo "[setup] Final listing:"
ls -la "${MODELS_DIR}"

if [ "${EXTRACT_SUCCESS}" -eq 1 ]; then
  echo "[setup] ✅ Model lengkap dan siap dipakai."
else
  echo "[setup] ❌ Model masih belum lengkap. Cek struktur asset release kamu."
  exit 1
fi

echo "=================================================="
echo "Setup selesai"
echo "=================================================="
