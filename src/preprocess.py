import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Konfigurasi Path Berdasarkan Struktur ---
BASE_PROJECT_DIR = ".."
DATASETS_ROOT_DIR = os.path.join(BASE_PROJECT_DIR, "datasets")
MERGED_DATA_DIR = os.path.join(DATASETS_ROOT_DIR, "merged_for_processing")
PROCESSED_OUTPUT_DIR = os.path.join(DATASETS_ROOT_DIR, "processed")

os.makedirs(MERGED_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_OUTPUT_DIR, exist_ok=True)

# --- 1. FUNGSI UNTUK MENGGABUNGKAN DATASET SPESIFIK KATEGORI ---

def merge_sports_datasets(
    raw_sports_dir=os.path.join(DATASETS_ROOT_DIR, "sports"),
    out_dir=MERGED_DATA_DIR,
    out_filename="sports_merged.csv"
):
    fake_path = os.path.join(raw_sports_dir, "fake.csv")
    real_path = os.path.join(raw_sports_dir, "real.csv")
    out_path  = os.path.join(out_dir, out_filename)

    if not (os.path.exists(fake_path) and os.path.exists(real_path)):
        print(f"PERINGATAN: File '{fake_path}' atau '{real_path}' tidak ditemukan. Melewati penggabungan dataset sports.")
        return None

    print(f"Menggabungkan dataset sports dari '{raw_sports_dir}' menjadi '{out_path}'...")
    try:
        df_fake = pd.read_csv(fake_path, on_bad_lines="skip")
        text_col_fake = 'tweet' if 'tweet' in df_fake.columns else 'text'
        if text_col_fake not in df_fake.columns:
             alt_cols = ['content', 'article', 'title', 'news', 'description', 'text_content']
             text_col_fake = next((col for col in alt_cols if col in df_fake.columns), None)
             if not text_col_fake: raise ValueError(f"Kolom teks utama tidak ditemukan di {fake_path}")
             print(f"  INFO (sports fake): Menggunakan kolom '{text_col_fake}' sebagai teks.")

        df_fake = df_fake[[text_col_fake]].copy()
        df_fake.rename(columns={text_col_fake: "text"}, inplace=True)
        df_fake["label"] = 1

        df_real = pd.read_csv(real_path, on_bad_lines="skip")
        text_col_real = 'tweet' if 'tweet' in df_real.columns else 'text'
        if text_col_real not in df_real.columns:
            alt_cols = ['content', 'article', 'title', 'news', 'description', 'text_content']
            text_col_real = next((col for col in alt_cols if col in df_real.columns), None)
            if not text_col_real: raise ValueError(f"Kolom teks utama tidak ditemukan di {real_path}")
            print(f"  INFO (sports real): Menggunakan kolom '{text_col_real}' sebagai teks.")

        df_real = df_real[[text_col_real]].copy()
        df_real.rename(columns={text_col_real: "text"}, inplace=True)
        df_real["label"] = 0

        df_combined = pd.concat([df_fake, df_real], ignore_index=True)
        df_combined.to_csv(out_path, index=False)
        print(f"'{out_filename}' (sports) berhasil disimpan di: {out_dir}\n")
        return out_path
    except Exception as e:
        print(f"ERROR saat menggabungkan dataset sports: {e}")
        return None

def merge_politics_datasets(
    raw_politics_dir=os.path.join(DATASETS_ROOT_DIR, "politics"),
    out_dir=MERGED_DATA_DIR,
    out_filename="politics_merged.csv"
):
    # Menggunakan nama kolom aktual dari output Anda sebelumnya
    xlsx_files_info = [
        {"name": "dataset_cnn_10k_cleaned.xlsx", "text_col": "text_new", "label_col": "hoax"},
        {"name": "dataset_kompas_4k_cleaned.xlsx", "text_col": "text_new", "label_col": "hoax"},
        {"name": "dataset_tempo_6k_cleaned.xlsx", "text_col": "text_new", "label_col": "hoax"},
        {"name": "dataset_turnbackhoax_10_cleaned.xlsx", "text_col": "Clean Narasi", "label_col": "hoax"}
    ]

    source_files_paths = [os.path.join(raw_politics_dir, info["name"]) for info in xlsx_files_info]
    out_path = os.path.join(out_dir, out_filename)

    existing_files_info = []
    for i, path in enumerate(source_files_paths):
        if os.path.exists(path):
            existing_files_info.append({"path": path, **xlsx_files_info[i]})
        else:
            print(f"PERINGATAN: File politik '{path}' (dikonfigurasi sebagai '{xlsx_files_info[i]['name']}') tidak ditemukan. Akan dilewati.")

    if not existing_files_info:
        print("PERINGATAN: Tidak ada file sumber politik yang ditemukan. Melewati penggabungan dataset politik.")
        return None

    print(f"Menggabungkan file politik dari '{raw_politics_dir}' menjadi '{out_path}'...")
    dfs = []
    try:
        for file_info in existing_files_info:
            print(f"  Membaca file politik: {file_info['path']}")
            df = pd.read_excel(file_info["path"])
            text_c, label_c = file_info["text_col"], file_info["label_col"]

            if text_c not in df.columns or label_c not in df.columns:
                print(f"  PERINGATAN PENTING: Kolom '{text_c}' atau '{label_c}' TIDAK DITEMUKAN di {file_info['path']}. File ini akan dilewati.")
                print(f"    Kolom yang tersedia di file tersebut: {df.columns.tolist()}")
                continue
            
            df_sub = df[[text_c, label_c]].copy()
            df_sub = df_sub.rename(columns={text_c: "text", label_c: "label"})
            dfs.append(df_sub)

        if not dfs:
            print("Tidak ada data politik yang berhasil dibaca untuk digabungkan. Periksa konfigurasi nama kolom di xlsx_files_info dan nilai labelnya.")
            return None
            
        df_politics = pd.concat(dfs, ignore_index=True)
        if 'label' in df_politics.columns:
            try:
                if not pd.api.types.is_integer_dtype(df_politics['label']):
                    df_politics.dropna(subset=['label'], inplace=True)
                    # Mencoba mengekstrak angka dari string seperti '0.0' atau '1.0'
                    df_politics['label'] = df_politics['label'].astype(str).str.extract(r'(\d+)').astype(float).astype(int)
            except Exception as e_convert:
                print(f"  PERINGATAN (politics merge): Gagal mengonversi kolom label ke integer setelah merge: {e_convert}. Periksa nilai-nilai di kolom label asli.")

        df_politics.to_csv(out_path, index=False)
        print(f"'{out_filename}' (politics) berhasil disimpan di: {out_dir}\n")
        return out_path
    except Exception as e:
        print(f"ERROR saat menggabungkan dataset politik: {e}")
        return None

def merge_gossip_datasets(
    raw_gossip_dir=os.path.join(DATASETS_ROOT_DIR, "gossip"),
    out_dir=MERGED_DATA_DIR,
    out_filename="gossip_merged.csv"
):
    files_to_merge = {
        "gossipcop_fake.csv": 1,
        "gossipcop_real.csv": 0,
        "politifact_fake.csv": 1,
        "politifact_real.csv": 0
    }
    out_path = os.path.join(out_dir, out_filename)
    dfs = []

    print(f"Menggabungkan dataset gossip dari '{raw_gossip_dir}' menjadi '{out_path}'...")
    all_files_exist = True
    for filename in files_to_merge.keys():
        if not os.path.exists(os.path.join(raw_gossip_dir, filename)):
            print(f"PERINGATAN: File gossip '{filename}' tidak ditemukan di '{raw_gossip_dir}'.")
            all_files_exist = False
    
    if not all_files_exist:
        print("Melewati penggabungan dataset gossip karena ada file yang hilang.")
        return None

    try:
        for filename, label_val in files_to_merge.items():
            file_path = os.path.join(raw_gossip_dir, filename)
            df_part = pd.read_csv(file_path, on_bad_lines="skip")
            
            text_col_gossip = None
            if 'title' in df_part.columns: text_col_gossip = 'title'
            elif 'text' in df_part.columns: text_col_gossip = 'text'
            elif 'content' in df_part.columns: text_col_gossip = 'content'

            if not text_col_gossip:
                print(f"PERINGATAN: Kolom teks ('title', 'text', atau 'content') tidak ditemukan di {filename}. Melewati file ini.")
                continue
            
            df_part = df_part[[text_col_gossip]].copy()
            df_part.rename(columns={text_col_gossip: "text"}, inplace=True)
            df_part["label"] = label_val
            dfs.append(df_part)

        if not dfs:
            print("Tidak ada data gossip yang berhasil dibaca untuk digabungkan.")
            return None

        df_combined = pd.concat(dfs, ignore_index=True)
        df_combined.to_csv(out_path, index=False)
        print(f"'{out_filename}' (gossip) berhasil disimpan di: {out_dir}\n")
        return out_path
    except Exception as e:
        print(f"ERROR saat menggabungkan dataset gossip: {e}")
        return None

# --- 2. KONFIGURASI PATH DAN KOLOM UNTUK MAIN PREPROCESSING ---
DATASET_INPUT_FILES = {
    "politics": None,
    "sports": None,
    "gossip": None,
    "medical": os.path.join(DATASETS_ROOT_DIR, "medical", "ACOVMD.csv"),
    "general": os.path.join(DATASETS_ROOT_DIR, "general", "indonesian_hoax_news.csv")
}

# Perubahan untuk GENERAL: menggunakan 'title' untuk teks dan juga untuk EKSTRAKSI label
DATASET_INPUT_COLUMNS = {
    "politics": {"text_col": "text", "label_col": "label"},
    "sports":   {"text_col": "text", "label_col": "label"},
    "gossip":   {"text_col": "text", "label_col": "label"},
    "medical":  {"text_col": "Tweet", "label_col": "Label"},
    "general":  {"text_col": "title", "label_col": "title"} # label_col diubah ke "title"
}

# --- 3. FUNGSI Pembersihan & Pemetaan Label ---
def clean_text_universal(text_input):
    if not isinstance(text_input, str): return ""
    # Menghapus tag [SALAH], (SALAH), dll. dari teks SEBELUM pembersihan lain, agar tidak mempengaruhi label
    text = re.sub(r'\[\s*salah\s*\]|\(\s*salah\s*\)', '', text_input, flags=re.IGNORECASE).strip()
    
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#\w+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\?\!]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def map_label_universal(label_value, category_name):
    # label_value untuk general sekarang adalah isi dari kolom 'title'
    label_str_original_case = str(label_value).strip() # Simpan case asli untuk ekstraksi tag
    label_str_lower = label_str_original_case.lower()

    if category_name == "politics":
        if label_str_lower == "0" or label_str_lower == "0.0": return 0
        if label_str_lower == "1" or label_str_lower == "1.0": return 1

    elif category_name in ["sports", "gossip"]:
        if label_str_lower == "0": return 0
        if label_str_lower == "1": return 1

    elif category_name == "medical":
        if label_str_lower == "true": return 1
        if label_str_lower == "false": return 0

    elif category_name == "general":
        # Ekstrak label dari kolom title (yang sekarang ada di label_value)
        if isinstance(label_value, str):
            # Gunakan label_str_original_case untuk memeriksa pola agar tidak terpengaruh oleh .lower() pada tag
            if re.search(r'\[\s*SALAH\s*\]|\(\s*SALAH\s*\)', label_str_original_case, flags=re.IGNORECASE):
                return 1 # Hoax
            else:
                # Asumsi: jika tidak ada tag [SALAH] atau (SALAH), maka itu fakta.
                return 0 # Fakta
        return None # Jika label_value (title) bukan string

    return None

# --- 4. FUNGSI UTAMA PRA-PEMROSESAN ---
def main_preprocess_all_categories(input_files_config, input_columns_config, output_dir_processed):
    all_processed_data = []
    print("\nMemulai tahap pra-pemrosesan utama untuk semua kategori...")

    for category, input_file_path in tqdm(input_files_config.items(), desc="Kategori Diproses"):
        if not input_file_path or not os.path.exists(input_file_path):
            print(f"  PERINGATAN: File input untuk kategori '{category}' tidak tersedia ('{input_file_path}'). Melewati.")
            continue

        col_config = input_columns_config.get(category)
        if not col_config:
            print(f"  PERINGATAN: Konfigurasi kolom input untuk '{category}' tidak ditemukan. Melewati.")
            continue

        text_col  = col_config.get("text_col")
        label_col_for_mapping = col_config.get("label_col") # Ini akan jadi 'title' untuk general

        if not text_col:
            print(f"  PERINGATAN: 'text_col' tidak terdefinisi untuk '{category}'. Melewati.")
            continue
        
        encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        df_cat = None

        try:
            if input_file_path.endswith(".csv"):
                for enc in encodings_to_try:
                    try:
                        df_cat = pd.read_csv(input_file_path, on_bad_lines="skip", encoding=enc)
                        print(f"  INFO: File '{os.path.basename(input_file_path)}' ({category}) berhasil dibaca dengan encoding '{enc}'.")
                        break 
                    except UnicodeDecodeError:
                        if enc == encodings_to_try[-1]:
                             print(f"  PERINGATAN: Gagal membaca '{os.path.basename(input_file_path)}' ({category}) dengan semua encoding yang dicoba.")
                             df_cat = None
                    except Exception as read_err:
                        print(f"  ERROR pandas saat membaca '{os.path.basename(input_file_path)}' ({category}) dengan encoding '{enc}': {read_err}")
                        df_cat = None; break
                if df_cat is None and not input_file_path.endswith(".xlsx"):
                    print(f"  ERROR KRITIS: CSV '{os.path.basename(input_file_path)}' ({category}) tidak dapat dibaca. Melewati.")
                    continue
            elif input_file_path.endswith(".xlsx"):
                 df_cat = pd.read_excel(input_file_path)
                 print(f"  INFO: File '{os.path.basename(input_file_path)}' ({category}) berhasil dibaca (Excel).")
            else:
                print(f"  Format file '{input_file_path}' ({category}) tidak didukung.")
                continue
        except Exception as e:
            print(f"  ERROR UMUM saat membaca file input '{input_file_path}' ({category}): {e}")
            continue
        
        if df_cat is None: 
            continue

        print(f"    Memproses kategori '{category}' ({len(df_cat)} baris). Kolom ditemukan: {df_cat.columns.tolist()}")

        if text_col not in df_cat.columns:
            print(f"    PERINGATAN BESAR: Kolom teks '{text_col}' tidak ditemukan di '{os.path.basename(input_file_path)}' ({category}). Melewati kategori ini.")
            continue
        # Untuk label_col_for_mapping, kita tahu itu akan 'title' untuk general, yang pasti ada.
        # Untuk kategori lain, label_col akan merujuk ke kolom label asli.
        if label_col_for_mapping not in df_cat.columns:
            print(f"    PERINGATAN BESAR: Kolom yang digunakan untuk mapping label '{label_col_for_mapping}' tidak ditemukan di '{os.path.basename(input_file_path)}' ({category}). Baris tanpa label akan dilewati.")


        valid_rows_cat = 0
        for idx, row in tqdm(df_cat.iterrows(), total=df_cat.shape[0], desc=f"      Baris {category}", leave=False):
            text_data_for_cleaning = str(row.get(text_col, "")) # Ambil teks dari text_col untuk dibersihkan
            
            # Ambil nilai yang akan digunakan untuk menentukan label
            # Untuk 'general', ini akan menjadi isi dari 'title'. Untuk lainnya, isi dari kolom 'label' asli.
            value_for_label_mapping = str(row.get(label_col_for_mapping, ""))

            if pd.isna(text_data_for_cleaning) or not text_data_for_cleaning.strip():
                continue
            if pd.isna(value_for_label_mapping) and category != "general": # Untuk general, title (sbg value_for_label_mapping) harus ada
                 continue


            cleaned_text = clean_text_universal(text_data_for_cleaning)
            mapped_label_val = map_label_universal(value_for_label_mapping, category)

            if cleaned_text and mapped_label_val is not None:
                all_processed_data.append({
                    "text": cleaned_text,
                    "label": mapped_label_val,
                    "category": category
                })
                valid_rows_cat += 1
        print(f"        → Baris valid (dengan label terpetakan) yang diproses untuk '{category}': {valid_rows_cat}")

    if not all_processed_data:
        print("KRITIS: Tidak ada data yang berhasil diproses dari semua kategori. Periksa konfigurasi dan file input.")
        return
        
    final_df = pd.DataFrame(all_processed_data)
    print(f"\nTotal data gabungan akhir (dengan label valid): {len(final_df)} baris")
    
    final_df.dropna(subset=['label', 'text'], inplace=True)
    final_df = final_df[final_df['text'].str.strip().astype(bool)]
    if not final_df.empty:
        final_df['label'] = final_df['label'].astype(int)
    else:
        print("KRITIS: DataFrame kosong setelah pembersihan akhir. Tidak ada data untuk disimpan.")
        return

    print("\nDistribusi label final:")
    print(final_df["label"].value_counts(dropna=False))
    print("\nDistribusi kategori final:")
    print(final_df["category"].value_counts(dropna=False))
    
    if len(final_df) == 0:
        print("KRITIS: Tidak ada data tersisa setelah pembersihan akhir. Tidak bisa melanjutkan.")
        return

    if len(final_df) > 0 and len(final_df["label"].unique()) > 1 and final_df["label"].value_counts().min() >= 2:
        train_df, temp_df = train_test_split(final_df, test_size=0.30, random_state=42, stratify=final_df["label"])
        if len(temp_df) > 0 and len(temp_df["label"].unique()) > 1 and temp_df["label"].value_counts().min() >= 2:
            val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"])
        elif len(temp_df) > 0 :
             val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
        else: val_df,test_df = pd.DataFrame(columns=final_df.columns), pd.DataFrame(columns=final_df.columns)
    elif len(final_df) > 0 :
        train_df, temp_df = train_test_split(final_df, test_size=0.30, random_state=42)
        if len(temp_df) > 0: val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
        else: val_df,test_df = pd.DataFrame(columns=final_df.columns), pd.DataFrame(columns=final_df.columns)
    else: return

    print(f"\nUkuran Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    train_df.to_csv(os.path.join(output_dir_processed, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir_processed, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir_processed, "test.csv"), index=False)
    print(f"\nData train/val/test disimpan di '{output_dir_processed}'")


    # MAIN EXECUTION
if __name__ == "__main__":
    print("Memulai skrip pra-pemrosesan terpadu...")
    DATASET_INPUT_FILES["sports"] = merge_sports_datasets()
    DATASET_INPUT_FILES["politics"] = merge_politics_datasets()
    DATASET_INPUT_FILES["gossip"] = merge_gossip_datasets()

    print("\n--- Status File Input Sebelum Pra-Pemrosesan Utama ---")
    for cat, path in DATASET_INPUT_FILES.items():
        status = "ADA" if path and os.path.exists(path) else "TIDAK ADA / GAGAL DIBUAT"
        print(f"Kategori: {cat.ljust(10)} | Path: {str(path).ljust(70)} | Status: {status}")
    print("----------------------------------------------------")
    main_preprocess_all_categories(DATASET_INPUT_FILES, DATASET_INPUT_COLUMNS, PROCESSED_OUTPUT_DIR)
    print("\nSkrip pra-pemrosesan selesai.")

