# src/evaluate.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm.auto import tqdm
from scipy.stats import mode

# --- Konfigurasi ---
PROCESSED_DATA_DIR = "../datasets/processed"
MODELS_DIR = "../models"
RESULTS_DIR = "../results"
MAX_LEN = 256
BATCH_SIZE = 16

os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_CONFIG = {
    "bert": "BERT",
    "roberta": "RoBERTa",
    "electra": "ELECTRA",
    "xlnet": "XLNet"
}

# --- Kelas Dataset Kustom ---
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts; self.labels = labels; self.tokenizer = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, item):
        text = str(self.texts[item]); label = self.labels[item]
        encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'labels': torch.tensor(label, dtype=torch.long)}

# --- Fungsi untuk Mendapatkan Prediksi ---
def get_predictions(model, data_loader, device):
    model = model.eval().to(device); predictions = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Predicting", leave=False):
            input_ids = batch["input_ids"].to(device); attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
    return np.array(predictions)

# --- Fungsi Visualisasi Baru ---
def plot_class_distribution(df, output_path):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=df, palette='pastel')
    plt.title('Distribusi Kelas pada Data Uji', fontsize=16)
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('Jumlah', fontsize=12)
    plt.xticks(ticks=[0, 1], labels=['Fakta (0)', 'Hoax (1)'])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Diagram distribusi kelas disimpan di: {output_path}")

def plot_text_length_distribution(df, output_path):
    plt.figure(figsize=(10, 6))
    text_lengths = df['text'].str.split().str.len()
    sns.histplot(text_lengths, bins=50, kde=True, color='skyblue')
    plt.title('Distribusi Panjang Teks pada Data Uji', fontsize=16)
    plt.xlabel('Jumlah Kata', fontsize=12)
    plt.ylabel('Frekuensi', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Diagram distribusi panjang teks disimpan di: {output_path}")

def plot_full_metric_comparison(summary_df, output_path):
    # 'Melt' dataframe untuk plotting dengan seaborn
    df_melted = pd.melt(summary_df, id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-score'],
                        var_name='Metric', value_name='Score')

    plt.figure(figsize=(14, 8))
    barplot = sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted, palette='viridis')
    plt.title('Perbandingan Kinerja Metrik Antar Model', fontsize=16, pad=20)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Skor', fontsize=12)
    plt.xticks(rotation=15, ha="right")
    plt.ylim(0.9, 1.0) # Fokus pada perbedaan skor yang kecil
    plt.legend(title='Metrik')
    
    # Tambahkan label nilai di atas bar
    for container in barplot.containers:
        barplot.bar_label(container, fmt='%.4f', fontsize=9, padding=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Diagram perbandingan metrik lengkap disimpan di: {output_path}")

# --- Fungsi Utama Evaluasi ---
def main_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    # 1. Muat Data Uji dan Buat Visualisasi Data
    test_data_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")
    if not os.path.exists(test_data_path):
        print(f"ERROR: File data uji tidak ditemukan di {test_data_path}"); return
    test_df = pd.read_csv(test_data_path)
    test_df.dropna(subset=['text', 'label'], inplace=True); true_labels = test_df['label'].astype(int).to_numpy(); test_texts = test_df['text'].tolist()
    if len(test_texts) == 0: print("Data uji kosong. Hentikan evaluasi."); return

    print("\n--- Membuat Visualisasi Data Uji ---")
    plot_class_distribution(test_df, os.path.join(RESULTS_DIR, "class_distribution.png"))
    plot_text_length_distribution(test_df, os.path.join(RESULTS_DIR, "text_length_distribution.png"))

    # 2. Dapatkan Prediksi dari Setiap Model Individual
    all_predictions = {}
    for model_folder, model_name in MODEL_CONFIG.items():
        print(f"\n--- Memproses Model: {model_name} ---")
        model_path = os.path.join(MODELS_DIR, model_folder)
        if not os.path.exists(model_path):
            print(f"  PERINGATAN: Direktori model {model_path} tidak ditemukan. Melewati."); continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            test_dataset = NewsDataset(test_texts, true_labels, tokenizer, MAX_LEN)
            test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0)
            predictions = get_predictions(model, test_data_loader, device)
            all_predictions[model_name] = predictions
            print(f"  Prediksi untuk model {model_name} selesai.")
        except Exception as e: print(f"  ERROR saat memproses model {model_name}: {e}")
    if not all_predictions: print("KRITIS: Tidak ada model yang berhasil diproses. Hentikan evaluasi."); return

    # 3. Buat Prediksi Ensemble (Bagging) dengan Voting
    stacked_predictions = np.array(list(all_predictions.values())); ensemble_preds, _ = mode(stacked_predictions, axis=0, keepdims=False)
    all_predictions["Bagging (Ensemble)"] = ensemble_preds; print("\n--- Prediksi Ensemble (Bagging) dengan voting selesai. ---")

    # 4. Hitung Metrik dan Buat Output untuk Semua Model
    results_summary = []
    for model_name, predictions in all_predictions.items():
        print(f"\n--- Hasil Evaluasi untuk: {model_name} ---")
        report_dict = classification_report(true_labels, predictions, target_names=['Fakta (0)', 'Hoax (1)'], output_dict=True, zero_division=0)
        report_text = classification_report(true_labels, predictions, target_names=['Fakta (0)', 'Hoax (1)'], zero_division=0)
        print("\nClassification Report:"); print(report_text)
        with open(os.path.join(RESULTS_DIR, f"{model_name.replace(' ', '_')}_classification_report.txt"), "w") as f: f.write(report_text)
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fakta (0)', 'Hoax (1)'], yticklabels=['Fakta (0)', 'Hoax (1)'])
        plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title(f'Confusion Matrix - {model_name}')
        cm_path = os.path.join(RESULTS_DIR, f"{model_name.replace(' ', '_')}_confusion_matrix.png"); plt.savefig(cm_path, dpi=300); plt.close()
        print(f"  Confusion matrix disimpan di {cm_path}")
        results_summary.append({
            "Model": model_name, "Accuracy": report_dict['accuracy'], "Precision": report_dict['weighted avg']['precision'],
            "Recall": report_dict['weighted avg']['recall'], "F1-score": report_dict['weighted avg']['f1-score']
        })

    # 5. Buat dan Simpan Tabel Perbandingan & Diagram Perbandingan Akhir
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_df = summary_df.sort_values(by="F1-score", ascending=False).reset_index(drop=True)
        print("\n--- Ringkasan Perbandingan Final Antar Model ---")
        print(summary_df.round(4).to_string())
        summary_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison_summary.csv"), index=False, float_format='%.4f')
        summary_df.to_excel(os.path.join(RESULTS_DIR, "model_comparison_summary.xlsx"), index=False, float_format='%.4f')
        print(f"\nRingkasan perbandingan disimpan di '{RESULTS_DIR}/'")
        # Panggil fungsi visualisasi perbandingan metrik yang baru
        plot_full_metric_comparison(summary_df, os.path.join(RESULTS_DIR, "full_metric_comparison_plot.png"))

if __name__ == '__main__':
    main_evaluation()
