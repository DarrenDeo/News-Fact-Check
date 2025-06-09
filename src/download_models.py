# src/download_models.py
import os
from huggingface_hub import snapshot_download

# --- Perbaikan Kunci: Atur HF_HOME secara eksplisit ---
# Ini memberitahu pustaka huggingface untuk menggunakan /data/.cache sebagai
# folder cache, yang berada di dalam persistent storage yang bisa kita tulis.
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
    print("Memulai proses download model dengan skrip Python...")
    print("==================================================")
    
    for model_key, model_id in MODELS_TO_DOWNLOAD.items():
        print(f"\n---> Mengunduh {model_key.upper()} ({model_id})")
        
        # Tentukan direktori tujuan untuk model ini
        local_dir_path = os.path.join(MODEL_STORAGE_PATH, model_key)
        
        # Periksa apakah model sudah ada untuk menghemat waktu saat restart
        # Kita periksa keberadaan config.json sebagai penanda
        if os.path.exists(os.path.join(local_dir_path, "config.json")):
             print(f"---> Model {model_key.upper()} sudah ada. Melewati unduhan.")
             continue

        try:
            # Fungsi snapshot_download akan secara otomatis menggunakan
            # variabel lingkungan HF_HOME yang telah kita atur untuk cache.
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir_path,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"---> {model_key.upper()} berhasil diunduh ke {local_dir_path}")
        except Exception as e:
            print(f"[ERROR] Gagal mengunduh {model_key.upper()}: {e}")
            # Kita biarkan skrip berlanjut jika satu model gagal,
            # agar tidak menghentikan seluruh proses build.
            pass

    print("\n==================================================")
    print("Proses download model selesai.")
    print("==================================================")

if __name__ == "__main__":
    main()
