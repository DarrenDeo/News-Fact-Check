# src/download_models.py
import os
from huggingface_hub import snapshot_download

# Skrip ini akan dipanggil oleh setup.sh untuk mengunduh semua model
# menggunakan pustaka Python, yang seringkali lebih stabil.

# Path ke persistent storage di Hugging Face Spaces
MODEL_STORAGE_PATH = "/data/models"

# Daftar model yang akan diunduh
MODELS_TO_DOWNLOAD = {
    "bert": "indobenchmark/indobert-base-p2",
    "roberta": "cahya/roberta-base-indonesian-522M",
    "electra": "google/electra-base-discriminator",
    "xlnet": "xlnet-base-cased"
}

def main():
    print("==================================================")
    print("Memulai proses download model dengan skrip Python...")
    print("==================================================")
    
    for model_key, model_id in MODELS_TO_DOWNLOAD.items():
        print(f"\n---> Mengunduh {model_key.upper()} ({model_id})")
        
        # Tentukan direktori tujuan untuk model ini
        local_dir_path = os.path.join(MODEL_STORAGE_PATH, model_key)
        
        try:
            # Unduh semua file dari repositori model ke direktori lokal
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir_path,
                local_dir_use_symlinks=False, # Penting untuk lingkungan Docker
                resume_download=True # Akan melanjutkan unduhan jika terputus
            )
            print(f"---> {model_key.upper()} berhasil diunduh ke {local_dir_path}")
        except Exception as e:
            print(f"[ERROR] Gagal mengunduh {model_key.upper()}: {e}")
            # Kita bisa memilih untuk melanjutkan atau berhenti. Mari kita lanjutkan.
            pass

    print("\n==================================================")
    print("Proses download model selesai.")
    print("==================================================")

if __name__ == "__main__":
    main()
