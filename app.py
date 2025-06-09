from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os
import re
from scipy.stats import mode
import requests
from bs4 import BeautifulSoup
import traceback # Impor untuk melacak error detail

# --- 1. Inisialisasi Aplikasi Flask ---
app = Flask(__name__, static_folder='frontend')

# --- 2. Konfigurasi dan Pemuatan Model ---
MODELS_DIR = "/data/models" 
MODEL_CONFIG = {
    "BERT": os.path.join(MODELS_DIR, "bert"),
    "RoBERTa": os.path.join(MODELS_DIR, "roberta"),
    "ELECTRA": os.path.join(MODELS_DIR, "electra"),
    "XLNet": os.path.join(MODELS_DIR, "xlnet")
}

models_cache = {}
# Di server Hugging Face (CPU), kita akan selalu menggunakan CPU.
device = torch.device("cpu")
print(f"Perangkat komputasi diatur ke: {device}")

def scrape_news_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else ""
        article_body = ""
        main_content = soup.find('article') or soup.find('div', class_=re.compile(r'content|body|main|post|article|detail', re.I)) or soup.find('main')
        if main_content:
            paragraphs = main_content.find_all('p')
            article_body = " ".join([p.get_text(strip=True) for p in paragraphs])
        if not article_body or len(article_body) < 150:
            all_paragraphs = soup.find_all('p')
            article_body = " ".join([p.get_text(strip=True) for p in all_paragraphs])
        full_text = f"{title}. {article_body}"
        if not full_text.strip() or full_text.strip() == ".": return None, "Gagal mengekstrak konten artikel."
        return full_text, None
    except Exception as e: return None, f"Gagal mengakses atau memproses link: {e}"

def clean_text_for_prediction(text_input):
    if not isinstance(text_input, str): return ""
    text = re.sub(r'\[\s*salah\s*\]|\(\s*salah\s*\)|<.*?>|http\S+|www\S+|https\S+|\@\w+|\#\w+|[^\x00-\x7F]+|[^a-zA-Z0-9\s\.\,\?\!]', ' ', text_input, flags=re.IGNORECASE|re.MULTILINE).strip()
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_all_models():
    """Memuat semua model dan tokenizer ke memori dan secara eksplisit memindahkannya ke CPU."""
    print("*" * 50)
    print("Memuat semua model AI dari persistent storage...")
    for model_name, model_path in MODEL_CONFIG.items():
        print(f"\n[LOAD] Mencoba memuat model: {model_name}")
        if os.path.exists(model_path):
            try:
                print(f"  [LOAD] Path ditemukan: {model_path}. Memuat tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                print(f"  [LOAD] Tokenizer {model_name} dimuat. Memuat model...")
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                print(f"  [LOAD] Model {model_name} dimuat. Memindahkan ke CPU dan set ke eval mode...")
                model.to(device)
                model.eval()
                models_cache[model_name] = (model, tokenizer)
                print(f"  [SUCCESS] {model_name} berhasil dikonfigurasi.")
            except Exception as e: 
                print(f"  [ERROR] Gagal saat memuat model {model_name}: {e}")
                traceback.print_exc()
        else:
            print(f"  [WARNING] Direktori model untuk {model_name} tidak ditemukan di {model_path}")
    print("\nProses pemuatan semua model selesai.")
    print(f"Total model yang berhasil dimuat: {len(models_cache)}")
    print("*" * 50)


@app.route('/predict', methods=['POST'])
def predict():
    print("\n[PREDICT] Menerima permintaan di /predict")
    try:
        data = request.get_json()
        url_input = data.get('url', '')
        print(f"[PREDICT] URL yang diterima: {url_input}")
        if not url_input or not url_input.strip(): return jsonify({"error": "URL tidak boleh kosong"}), 400

        print("[PREDICT] Memulai proses scraping...")
        text_from_url, error_message = scrape_news_from_url(url_input)
        if error_message: return jsonify({"error": error_message}), 400
        print("[PREDICT] Scraping berhasil.")
        
        cleaned_text = clean_text_for_prediction(text_from_url)
        print("[PREDICT] Teks berhasil dibersihkan.")
        
        all_predictions = {}
        individual_preds_list = []

        for model_name, (model, tokenizer) in models_cache.items():
            print(f"  [PREDICT] Melakukan prediksi dengan {model_name}...")
            try:
                inputs = tokenizer.encode_plus(cleaned_text, add_special_tokens=True, max_length=256, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence, predicted_class_idx = torch.max(probs, dim=1)
                predicted_class = "Hoax" if predicted_class_idx.item() == 1 else "Fakta"
                individual_preds_list.append(predicted_class_idx.item())
                all_predictions[model_name] = {"prediction": predicted_class, "confidence": f"{confidence.item():.2%}"}
                print(f"  [PREDICT] Prediksi {model_name} berhasil: {predicted_class}")
            except Exception as e:
                print(f"  [ERROR] Prediksi dengan {model_name} gagal: {e}")
                all_predictions[model_name] = {"prediction": "Error", "confidence": "N/A"}

        if individual_preds_list:
            print("[PREDICT] Melakukan ensemble voting...")
            ensemble_vote_result = mode(np.array(individual_preds_list))
            final_prediction_idx = ensemble_vote_result.mode[0] if isinstance(ensemble_vote_result.mode, np.ndarray) else ensemble_vote_result.mode
            final_prediction = "Hoax" if final_prediction_idx == 1 else "Fakta"
            agreement = np.mean([p == final_prediction_idx for p in individual_preds_list])
            all_predictions["Bagging (Ensemble)"] = {"prediction": final_prediction, "confidence": f"{agreement:.2%}"}
            print("[PREDICT] Ensemble voting selesai.")

        print("[PREDICT] Mengirimkan hasil ke frontend.")
        return jsonify(all_predictions)
    except Exception as e:
        print(f"[FATAL ERROR] Terjadi error tak terduga di rute /predict:")
        traceback.print_exc()
        return jsonify({"error": "Kesalahan internal server."}), 500

@app.route('/')
def serve_index(): return send_from_directory('frontend', 'index.html')

# --- Pemuatan Model Dipanggil di Sini ---
# Ini akan dieksekusi saat Gunicorn mengimpor file app.py
load_all_models()

if __name__ == '__main__':
    # Blok ini sekarang hanya untuk menjalankan server secara lokal, bukan di Hugging Face
    app.run(host="0.0.0.0", port=7860, debug=True)
