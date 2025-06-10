import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm
import os
import numpy as np


MODEL_NAME = "cahya/roberta-base-indonesian-522M" # Pilihan model RoBERTa Indonesia
MAX_LEN = 256
BATCH_SIZE = 8 # Sesuaikan berdasarkan VRAM GPU Anda (sama seperti BERT)
EPOCHS = 3
LEARNING_RATE = 2e-5
OUTPUT_DIR = "../models/roberta" # Direktori penyimpanan untuk model RoBERTa
PROCESSED_DATA_DIR = "../datasets/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Kelas Dataset Kustom (Sama seperti pada train_bert.py) ---
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False, # RoBERTa tidak menggunakan token_type_ids
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- Fungsi untuk Memuat Data (Sama seperti pada train_bert.py) ---
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.dropna(subset=['text', 'label'], inplace=True)
        df['label'] = df['label'].astype(int)
        print(f"Data berhasil dimuat dari {file_path}. Jumlah baris: {len(df)}")
        if df.empty:
            print(f"PERINGATAN: Tidak ada data yang tersisa di {file_path} setelah menghapus NaN.")
        return df['text'].tolist(), df['label'].tolist()
    except FileNotFoundError:
        print(f"ERROR: File tidak ditemukan di {file_path}")
        return [], []
    except Exception as e:
        print(f"ERROR saat memuat data dari {file_path}: {e}")
        return [], []

# --- Fungsi Pelatihan per Epoch (Sama seperti pada train_bert.py) ---
def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
    model = model.train()
    total_train_loss = 0
    total_train_correct = 0
    progress_bar = tqdm(data_loader, desc="Training Epoch", leave=False)
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        total_train_loss += loss.item()
        _, preds = torch.max(logits, dim=1)
        total_train_correct += torch.sum(preds == labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if (batch_idx + 1) % 10 == 0:
            current_loss = total_train_loss / (batch_idx + 1)
            num_processed_samples = (batch_idx + 1) * data_loader.batch_size
            if batch_idx + 1 == len(data_loader): 
                num_processed_samples = n_examples - (len(data_loader) -1) * data_loader.batch_size if n_examples > (len(data_loader) -1) * data_loader.batch_size else data_loader.batch_size
            current_acc = total_train_correct.double() / num_processed_samples
            current_acc = min(current_acc, 1.0)
            progress_bar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.4f}'})
    avg_train_loss = total_train_loss / len(data_loader)
    avg_train_acc = total_train_correct.double() / n_examples
    return avg_train_acc, avg_train_loss

# --- Fungsi Evaluasi (Sama seperti pada train_bert.py) ---
def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    total_eval_loss = 0
    total_eval_correct = 0
    all_preds_list = []
    all_labels_list = []
    with torch.no_grad():
        progress_bar_eval = tqdm(data_loader, desc="Evaluating", leave=False)
        for batch in progress_bar_eval:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            total_eval_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            total_eval_correct += torch.sum(preds == labels)
            all_preds_list.extend(preds.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())
    avg_eval_loss = total_eval_loss / len(data_loader)
    avg_eval_acc = total_eval_correct.double() / n_examples
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels_list, all_preds_list, average='weighted', zero_division=0
    )
    return avg_eval_acc, avg_eval_loss, precision, recall, f1

# --- Fungsi Utama Pelatihan ---
def main():
    print("Memulai pelatihan model RoBERTa...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    print(f"Memuat tokenizer dari: {MODEL_NAME}")
    # Gunakan RobertaTokenizerFast
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
    
    print("Memuat data latih dan validasi...")
    train_texts, train_labels = load_data(os.path.join(PROCESSED_DATA_DIR, "train.csv"))
    val_texts, val_labels = load_data(os.path.join(PROCESSED_DATA_DIR, "val.csv"))

    if not train_texts or not val_texts:
        print("KRITIS: Data latih atau validasi kosong. Hentikan pelatihan.")
        return

    print(f"Jumlah data latih: {len(train_texts)}, Jumlah data validasi: {len(val_texts)}")

    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    print(f"Memuat model pre-trained: {MODEL_NAME}")
    # Gunakan RobertaForSequenceClassification
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    total_steps = len(train_data_loader) * EPOCHS
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    best_val_f1 = 0.0

    print(f"\n--- Memulai Loop Pelatihan untuk {EPOCHS} Epoch ---")
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        print('-' * 30)
        actual_train_samples = len(train_dataset)
        train_acc, train_loss = train_epoch(
            model, train_data_loader, optimizer, device, scheduler, actual_train_samples
        )
        print(f'Loss Latih: {train_loss:.4f} | Akurasi Latih: {train_acc:.4f}')

        val_acc, val_loss, val_precision, val_recall, val_f1 = eval_model(
            model, val_data_loader, device, len(val_dataset)
        )
        print(f'Loss Validasi: {val_loss:.4f} | Akurasi Validasi: {val_acc:.4f}')
        print(f'Presisi Validasi: {val_precision:.4f} | Recall Validasi: {val_recall:.4f} | F1 Validasi: {val_f1:.4f}')

        if val_f1 > best_val_f1:
            print(f"F1-score validasi meningkat dari {best_val_f1:.4f} ke {val_f1:.4f}. Menyimpan model...")
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            best_val_f1 = val_f1
            with open(os.path.join(OUTPUT_DIR, "best_metrics.txt"), "w") as f:
                f.write(f"Epoch: {epoch + 1}\n")
                f.write(f"Best Validation F1-score: {best_val_f1:.4f}\n")
                f.write(f"Validation Accuracy: {val_acc:.4f}\n")
                f.write(f"Validation Precision: {val_precision:.4f}\n")
                f.write(f"Validation Recall: {val_recall:.4f}\n")
                f.write(f"Validation Loss: {val_loss:.4f}\n")

    print("\nPelatihan selesai.")
    print(f"Model terbaik disimpan di direktori: {OUTPUT_DIR}")
    print(f"F1-score validasi terbaik yang dicapai: {best_val_f1:.4f}")

if __name__ == '__main__':
    main()
