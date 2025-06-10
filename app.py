# app.py
from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os
import re
from scipy.stats import mode
import requests
from bs4 import BeautifulSoup

# --- 1. Inisialisasi Aplikasi Flask ---
app = Flask(__name__, static_folder='frontend')

# --- 2. Konfigurasi dan Pemuatan Model ---
MODELS_DIR = "models"
MODEL_CONFIG = {
    "BERT": os.path.join(MODELS_DIR, "bert"),
    "RoBERTa": os.path.join(MODELS_DIR, "roberta"),
    "ELECTRA": os.path.join(MODELS_DIR, "electra"),
    "XLNet": os.path.join(MODELS_DIR, "xlnet")
}

models_cache = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Fungsi Web Scraping ---
def scrape_news_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else ""
        
        article_body = ""
        main_content = soup.find('article') or \
                       soup.find('div', class_=re.compile(r'content|body|main|post|article|detail', re.I)) or \
                       soup.find('main')

        if main_content:
            paragraphs = main_content.find_all('p')
            article_body = " ".join([p.get_text(strip=True) for p in paragraphs])
        
        if not article_body or len(article_body) < 150:
            print("  INFO (Scraper): Konten utama tidak signifikan, mencoba mengambil semua paragraf.")
            all_paragraphs = soup.find_all('p')
            article_body = " ".join([p.get_text(strip=True) for p in all_paragraphs])

        full_text = f"{title}. {article_body}"
        
        if not full_text.strip() or full_text.strip() == ".":
            return None, "Gagal mengekstrak konten artikel yang signifikan dari link tersebut."

        return full_text, None
    except requests.exceptions.RequestException as e:
        return None, f"Gagal mengakses link: {e}"
    except Exception as e:
        return None, f"Terjadi error saat scraping: {e}"

def clean_text_for_prediction(text_input):
    if not isinstance(text_input, str): return ""
    text = re.sub(r'\[\s*salah\s*\]|\(\s*salah\s*\)', '', text_input, flags=re.IGNORECASE).strip()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#\w+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\?\!]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_all_models():
    print("*" * 50)
    print("Memuat semua model AI ke memori...")
    for model_name, model_path in MODEL_CONFIG.items():
        if os.path.exists(model_path):
            print(f"  > Memuat {model_name} dari {model_path}...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                model.to(device)
                model.eval()
                models_cache[model_name] = (model, tokenizer)
                print(f"  > {model_name} berhasil dimuat.")
            except Exception as e:
                print(f"  ERROR saat memuat model {model_name}: {e}")
        else:
            print(f"  PERINGATAN: Direktori model untuk {model_name} tidak ditemukan di {model_path}")
    print("Semua model yang tersedia telah dimuat.")
    print("*" * 50)


# --- 3. Perbarui Rute API untuk Prediksi dengan Penanganan Error ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        url_input = data.get('url', '')

        if not url_input or not url_input.strip():
            return jsonify({"error": "URL tidak boleh kosong"}), 400

        print(f"Menerima permintaan untuk URL: {url_input}")
        text_from_url, error_message = scrape_news_from_url(url_input)

        if error_message:
            print(f"Error scraping: {error_message}")
            return jsonify({"error": error_message}), 400
        
        cleaned_text = clean_text_for_prediction(text_from_url)
        
        all_predictions = {}
        individual_preds_list = []

        for model_name, (model, tokenizer) in models_cache.items():
            try:
                inputs = tokenizer.encode_plus(
                    cleaned_text,
                    add_special_tokens=True,
                    max_length=256,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)

                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)
                
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence, predicted_class_idx = torch.max(probs, dim=1)
                
                predicted_class = "Hoax" if predicted_class_idx.item() == 1 else "Fakta"
                individual_preds_list.append(predicted_class_idx.item())

                all_predictions[model_name] = {
                    "prediction": predicted_class,
                    "confidence": f"{confidence.item():.2%}"
                }
            except Exception as e:
                print(f"Error saat prediksi dengan {model_name}: {e}")
                all_predictions[model_name] = {"prediction": "Error", "confidence": "N/A"}

        if individual_preds_list:
            ensemble_vote_result = mode(np.array(individual_preds_list))
            # PERBAIKAN: Cara yang lebih aman untuk mendapatkan hasil dari `mode`
            final_prediction_idx = ensemble_vote_result.mode[0] if isinstance(ensemble_vote_result.mode, np.ndarray) else ensemble_vote_result.mode
            
            final_prediction = "Hoax" if final_prediction_idx == 1 else "Fakta"
            
            agreement = np.mean([p == final_prediction_idx for p in individual_preds_list])
            
            all_predictions["Bagging (Ensemble)"] = {
                "prediction": final_prediction,
                "confidence": f"{agreement:.2%}"
            }

        return jsonify(all_predictions)

    except Exception as e:
        print(f"[FATAL ERROR] Terjadi error tak terduga di rute /predict: {e}")
        return jsonify({"error": f"Terjadi kesalahan internal pada server. Silakan periksa log terminal backend untuk detail."}), 500


# --- 4. Rute untuk Menyajikan Halaman Web (Frontend) ---
@app.route('/')
def serve_index():
    return send_from_directory('frontend', 'index.html')

# --- 5. Jalankan Aplikasi ---
if __name__ == '__main__':
    load_all_models()
    app.run(host="0.0.0.0", port=5000, debug=False)

