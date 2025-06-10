# generate_diagram.py
# Skrip ini menggunakan pustaka Graphviz untuk membuat diagram alur kerja proyek.
# Pastikan Graphviz terinstal:
# 1. Instal pustaka Python: pip install graphviz
# 2. Instal aplikasi Graphviz di sistem Anda: https://graphviz.org/download/
#    (Penting: Tambahkan path 'bin' dari Graphviz ke environment variable PATH Anda)

from graphviz import Digraph

# Inisialisasi diagram baru
dot = Digraph('ProjectWorkflow', comment='Alur Kerja Proyek Pengecek Fakta AI')
dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue', fontname='Helvetica')
dot.attr('edge', fontname='Helvetica', fontsize='10')

# --- Tahap 1: Persiapan Data ---
with dot.subgraph(name='cluster_0') as c:
    c.attr(label='Tahap 1: Persiapan & Pra-pemrosesan Data', style='rounded', color='grey')
    c.node('raw_data', 'Dataset Mentah\n(Politik, Sports, Gossip, Medis, General)', shape='folder', fillcolor='lightyellow')
    c.node('preprocess', 'Jalankan src/preprocess.py\n(Merge, Clean, Map Labels, Split)')
    c.node('processed_data', 'Data Siap Pakai\n(train.csv, val.csv, test.csv)', shape='cylinder', fillcolor='lightyellow')
    
    c.edge('raw_data', 'preprocess')
    c.edge('preprocess', 'processed_data')

# --- Tahap 2: Pelatihan Model ---
with dot.subgraph(name='cluster_1') as c:
    c.attr(label='Tahap 2: Pelatihan Model Individual', style='rounded', color='grey')
    c.node('train_val_data', 'Data Latih & Validasi', shape='cylinder', fillcolor='lightyellow')
    
    c.node('train_bert', 'Fine-tune BERT')
    c.node('train_roberta', 'Fine-tune RoBERTa')
    c.node('train_electra', 'Fine-tune ELECTRA')
    c.node('train_xlnet', 'Fine-tune XLNet')
    
    c.node('bert_model', 'Model BERT', shape='box3d', fillcolor='lightgreen')
    c.node('roberta_model', 'Model RoBERTa', shape='box3d', fillcolor='lightgreen')
    c.node('electra_model', 'Model ELECTRA', shape='box3d', fillcolor='lightgreen')
    c.node('xlnet_model', 'Model XLNet', shape='box3d', fillcolor='lightgreen')
    
    c.edge('train_val_data', 'train_bert')
    c.edge('train_val_data', 'train_roberta')
    c.edge('train_val_data', 'train_electra')
    c.edge('train_val_data', 'train_xlnet')
    
    c.edge('train_bert', 'bert_model')
    c.edge('train_roberta', 'roberta_model')
    c.edge('train_electra', 'electra_model')
    c.edge('train_xlnet', 'xlnet_model')

# --- Tahap 3: Evaluasi & Ensemble ---
with dot.subgraph(name='cluster_2') as c:
    c.attr(label='Tahap 3: Evaluasi & Ensemble', style='rounded', color='grey')
    c.node('test_data', 'Data Uji', shape='cylinder', fillcolor='lightyellow')
    c.node('evaluate', 'Jalankan src/evaluate.py')
    c.node('ensemble', 'Ensemble Voting (Bagging)')
    c.node('results', 'Laporan Kinerja & Visualisasi\n(CSV, XLSX, PNG)', shape='document', fillcolor='lightgrey')
    
    c.edge('test_data', 'evaluate')
    c.edge('bert_model', 'evaluate', style='dashed')
    c.edge('roberta_model', 'evaluate', style='dashed')
    c.edge('electra_model', 'evaluate', style='dashed')
    c.edge('xlnet_model', 'evaluate', style='dashed')
    
    c.edge('evaluate', 'ensemble')
    c.edge('ensemble', 'results')

# --- Tahap 4: Aplikasi Web Interaktif ---
with dot.subgraph(name='cluster_3') as c:
    c.attr(label='Tahap 4: Aplikasi Web & Inferensi Real-time', style='rounded', color='grey')
    c.node('user', 'Pengguna', shape='ellipse', fillcolor='moccasin')
    c.node('frontend', 'Frontend (index.html)', shape='component')
    c.node('backend', 'Backend Server (app.py)\nMemuat Semua Model', shape='component', width='3')
    c.node('prediction', 'Hasil Prediksi Final\n(JSON)', shape='document')
    
    c.edge('user', 'frontend', label='Input Link Berita')
    c.edge('frontend', 'backend', label='Kirim URL via API')
    c.edge('backend', 'prediction', label='Scrape & Prediksi')
    c.edge('prediction', 'frontend', label='Kirim Hasil')
    c.edge('frontend', 'user', label='Tampilkan Hasil Analisis')

# Hubungkan tahap-tahap
dot.edge('processed_data', 'train_val_data', style='invis') # Layouting helper
dot.edge('processed_data', 'test_data', style='invis')
dot.edge('bert_model', 'backend', style='dashed', constraint='false') # Layouting helper

# Render diagram ke file PNG
output_filename = 'arsitektur_dan_alur_kerja'
dot.render(output_filename, format='png', cleanup=True)

print(f"Diagram berhasil dibuat dan disimpan sebagai '{output_filename}.png'")
