# src/download_models.py
import os
import shutil
from huggingface_hub import snapshot_download

# --- Atur Direktori Cache ---
cache_dir = "/data/.cache"
os.environ['HF_HOME'] = cache_dir
os.makedirs(cache_dir, exist_ok=True)
print(f"Hugging Face home/cache directory set to: {os.environ['HF_HOME']}")

# Path utama untuk menyimpan model final
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
    print("Memulai proses download model dengan verifikasi...")
    print("==================================================")
    
    for model_key, model_id in MODELS_TO_DOWNLOAD.items():
        print(f"\n---> Memeriksa {model_key.upper()} ({model_id})")
        
        local_dir_path = os.path.join(MODEL_STORAGE_PATH, model_key)
        
        # PERBAIKAN: Verifikasi kelengkapan model.
        # Cara sederhana adalah memeriksa keberadaan file bobot model.
        is_complete = False
        if os.path.exists(local_dir_path):
            if os.path.exists(os.path.join(local_dir_path, "model.safetensors")) or \
               os.path.exists(os.path.join(local_dir_path, "pytorch_model.bin")):
                is_complete = True

        if is_complete:
             print(f"---> Model {model_key.upper()} sudah lengkap. Melewati unduhan.")
             continue
        elif os.path.exists(local_dir_path) and not is_complete:
            print(f"---> Model {model_key.upper()} tidak lengkap. Menghapus folder lama untuk mengunduh ulang...")
            shutil.rmtree(local_dir_path)
            
        try:
            print(f"---> Mengunduh {model_key.upper()}...")
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir_path,
                local_dir_use_symlinks=False,
                resume_download=True 
            )
            print(f"---> {model_key.upper()} berhasil diunduh ke {local_dir_path}")
        except Exception as e:
            print(f"[ERROR] Gagal mengunduh {model_key.upper()}: {e}")
            pass

    print("\n==================================================")
    print("Proses download model selesai.")
    print("==================================================")

if __name__ == "__main__":
    main()
