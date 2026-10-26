# Sentiment Analysis — App Reviews

> **🔗 Live Demo:** [https://kusjay-space.github.io/sentiment-analysis](https://kusjay-space.github.io/sentiment-analysis)

> **🚧 Work in progress** — hasil akan diupdate setelah training. Semua angka di dokumen ini diisi dari `results/metrics.json` hasil run aktual.

Pipeline NLP untuk klasifikasi sentimen review aplikasi menjadi **positif, netral, atau negatif**.

## Overview

Project ini adalah submission untuk AI Engineer Intern assessment di Timedoor. Menggunakan dataset Amazon Reviews EN dan fine-tune DistilBERT untuk mengklasifikasi review.

## Dataset

Sumber: [HuggingFace — `SetFit/amazon_reviews_multi_en`](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en) (salah satu platform yang direkomendasikan brief, selain Kaggle dan Roboflow).

**Ukuran split**

| Split | Jumlah | Keterangan |
|---|---|---|
| Train | **30.000** | subsample stratified 12k NEG / 6k NEU / 12k POS, seed 42 |
| Validation | 5.000 | seleksi model & hyperparameter |
| Test | 5.000 | disentuh **tepat sekali** untuk angka final |

**Mapping label** — rating bintang digabung menjadi tiga kelas sentimen:

| Bintang asli | Kelas |
|---|---|
| ★☆☆☆☆ · ★★☆☆☆ | NEGATIVE |
| ★★★☆☆ | NEUTRAL |
| ★★★★☆ · ★★★★★ | POSITIVE |

**Distribusi kelas pada data training** — hanya bintang 3 yang menjadi NEUTRAL, sedangkan NEGATIVE dan POSITIVE masing-masing mendapat dua bintang. Karena itu NEUTRAL jadi kelas minoritas (20%):

| Kelas | Jumlah | Proporsi |
|---|---|---|
| NEGATIVE | 12.000 | 40% |
| NEUTRAL | 6.000 | 20% |
| POSITIVE | 12.000 | 40% |
| **Total** | **30.000** | 100% |

Karena NEUTRAL minoritas, baseline memakai `class_weight='balanced'` dan F1 dilaporkan per kelas.

**Karakteristik teks:** bahasa Inggris · median 23 kata (p95 = 88, maks 429) · 98.3% review ≤ 128 kata sehingga `max_len=128` memadai · kolom `id`, `text`, `label`, `label_text`.

## Model

| Model | Split | Akurasi | F1-macro |
|---|---|---|---|
| TF-IDF + LogisticRegression (tuned) | validation | 70.2% | 0.674 |
| DistilBERT (fine-tuned) | **test** | **75.1%** | **0.687** |

Kedua model dievaluasi di split yang sama (train subsample → seleksi di validation → angka final di test, disentuh sekali). DistilBERT mengalahkan baseline **+4.9% accuracy**.

**F1 per kelas (DistilBERT, test):** NEGATIVE 0.806, NEUTRAL 0.403, POSITIVE 0.851. NEUTRAL tetap kelas terlemah — bintang 3 secara inheren ambigu (lihat analisis di bawah). Inference 10 sampel unseen: **8/10 benar** (`results/predictions.csv`).

### Kenapa baseline mentok di ~70%

Sweep hyperparameter TF-IDF+LogReg (lihat `src/baseline_tuned.py`) menunjukkan ceiling ≈70% pada data training 30k — bigram/trigram, char n-gram, C besar, dan 510k features semuanya < 71%. Bottlenecknya adalah kelas NEUTRAL (F1 0.47): bintang 3 secara inheren ambigu dan TF-IDF tidak menangkap konteks. Ini motivasi utama fine-tune DistilBERT.

### Hasil DistilBERT vs ekspektasi

Target awal 85% tidak tercapai (real: 75.1% test). Penyebab: (1) training hanya 2 epoch di subset 30k — model kontekstual butuh lebih banyak data/epoch untuk menangkap nuansa NEUTRAL; (2) NEUTRAL (bintang 3) memang ambigu secara linguistik — bahkan DistilBERT hanya F1 0.40 di kelas ini. Namun DistilBERT tetap unggul **+4.9% accuracy** atas baseline dan jauh lebih baik di NEGATIVE/POSITIVE (0.81/0.85). Perbaikan lanjutan: tambah data training, 3-4 epoch, atau focal loss untuk kelas minoritas.

### Validasi di Data Nyata (Real-World Test)

Test set akademik bisa menipu: model bisa menghafal pola dataset. Untuk itu model diuji pada **review asli hasil scraping Play Store** (`google-play-scraper`), di luar distribusi training. Ground truth = star rating, mapping sama dengan training.

| Test | Data | n | Accuracy | F1-macro | Confidence < 0.5 |
|---|---|---|---|---|---|
| Test set akademik | Amazon Reviews (test split) | 5.000 | 75.1% | 0.687 | — |
| **Real-world English** | Play Store US (Spotify, Instagram) | 67 | **74.6%** | 0.593 | 10% |
| **Real-world Indonesian** | Play Store ID (Gojek, Tokopedia) | 80 | 57.5% | 0.364 | **69%** |

**Interpretasi:**

1. **English generalizes.** Selisih 75.1% (akademik) vs 74.6% (nyata) hanya **0.5 poin** — bukti model tidak overfit dan benar-benar belajar sinyal sentimen, bukan menghafal dataset.

2. **Indonesia di luar domain.** Model dilatih hanya pada teks Inggris; tokenizer memecah kata Indonesia menjadi 2–4 subword tanpa makna (contoh: `mengecewakan` → `meng`/`##ece`/`##wa`/`##kan`). Akibatnya akurasi turun ke 57.5%, dan model cenderung bias ke NEGATIVE (60 dari 80 prediksi).

3. **Model menandai ketidakpastiannya.** Di data Indonesia, **69% prediksi punya confidence < 0.5**, versus hanya 10% di data English. Model tidak salah tebak dengan yakin — ia melaporkan rendahnya keyakinan, yang merupakan perilaku yang diinginkan untuk sistem produksi (bisa di-flag untuk review manusia).

4. **Noise di data nyata.** Sebagian review Play Store US ternyata berbahasa Portugis, dan sebagian label bintang tidak konsisten dengan teksnya (mis. teks negatif tapi diberi 5 bintang). Ini menekankan bahwa ceiling akurasi pada data nyata lebih rendah dari benchmark akademik.

**Implikasi untuk produksi:** untuk feedback pengguna Indonesia, diperlukan model berbahasa Indonesia (IndoBERT / XLM-R) yang dilatih pada korpus Indonesia — di luar scope assessment ini (lihat PRD §3 Out of Scope). Bukti lengkap per sampel: [`results/real_world_test.json`](results/real_world_test.json).

## Quick Start

```bash
# Clone
git clone https://github.com/kusjay-space/sentiment-analysis.git
cd sentiment-analysis

# Buat virtual environment
python -m venv .venv && source .venv/bin/activate

# Install
pip install -r requirements.txt

# Download model checkpoint (~256 MB) dari Releases
gh release download v1.0 -p model_final.zip
unzip model_final.zip -d model/

# Training (baseline)
python -m src.baseline

# Training (DistilBERT) — butuh GPU, gunakan Colab
python -m src.train

# Inference pada test set (menghasilkan metrics.json, predictions.csv, confusion matrix)
python -m src.inference
```

## Cara Pakai untuk Data Review Sendiri

> **Penting:** jalankan dengan Python dari virtualenv proyek, bukan `python` sistem.
> Cara paling aman: aktifkan venv dulu (`source .venv/bin/activate`) atau pakai wrapper `./run.sh`.
> Kalau muncul `ModuleNotFoundError: No module named 'torch'`, itu tandanya venv belum aktif.

```bash
# Cara 1 (disarankan) — wrapper, otomatis pakai venv
./run.sh predict "This product is amazing"
./run.sh analyze reviews.csv

# Cara 2 — aktifkan venv dulu
source .venv/bin/activate
python -m src.predict "This product is amazing"
./run.sh analyze reviews.csv

# Cara 3 — panggil venv python langsung
.venv/bin/python -m src.predict "This product is amazing"
.venv/bin/python -m src.analyze reviews.csv
```

Ada dua mode, tergantung kebutuhan.

### A. Satu teks saja (cepat)

```bash
./run.sh predict "This product is amazing, works perfectly"
```

Output:

```
  → POSITIVE (conf 0.992)
    NEG 0.002 | NEU 0.005 | POS 0.992
```

Tanpa argumen, masuk mode interaktif (ketik teks → Enter → hasil, `quit` untuk keluar):

```bash
./run.sh predict
```

### B. Banyak review dari file CSV/Excel (batch)

Siapkan file dengan **minimal satu kolom berisi teks review**. Contoh `reviews.csv`:

```csv
id,review,rating
1,This app is fantastic. Works perfectly.,5
2,Stopped working after the update. Crashes constantly.,1
3,It's okay. Does the job but nothing special.,3
```

Kolom `rating` **opsional** — kalau ada (skala 1–5 atau 0–4), tool otomatis menghitung akurasi terhadap rating tersebut.

Jalankan:

```bash
./run.sh analyze reviews.csv
```

Output di terminal:

```
Reading reviews.csv ...
  3 rows · 3 columns
  text column : review
  rating column: rating

Running inference on 3 reviews (device: cpu) ...

==============================================================
HASIL ANALISIS
==============================================================
  Review dianalisis      : 3
  Rata-rata confidence   : 0.863
  Perlu review manual    : 0 (confidence < 0.5)

  Distribusi sentimen:
    POSITIVE       2  ( 66.7%)  █████████████████████████████████
    NEUTRAL        0  (  0.0%)
    NEGATIVE       1  ( 33.3%)  ████████████████

  Akurasi vs rating asli : 100.00% (3 baris punya rating)
--------------------------------------------------------------
  Output  : results/analyzed_reviews.csv
  Ringkasan: results/analyze_summary.json
--------------------------------------------------------------
```

**File output** (`results/analyzed_reviews.csv`) berisi kolom asli + kolom baru:

| Kolom baru | Isi |
|---|---|
| `sentiment` | `POSITIVE` / `NEUTRAL` / `NEGATIVE` |
| `confidence` | 0–1, keyakinan model pada prediksinya |
| `prob_negative` | probabilitas NEGATIVE |
| `prob_neutral` | probabilitas NEUTRAL |
| `prob_positive` | probabilitas POSITIVE |
| `needs_review` | `True` bila confidence < 0.5 → sebaiknya dicek manusia |

**Opsi tambahan:**

```bash
# Tentukan sendiri kolom teksnya (kalau nama kolom tidak terdeteksi otomatis)
./run.sh analyze reviews.csv --text-col komentar

# Tentukan lokasi output
./run.sh analyze reviews.csv -o hasil/analisis.csv

# Excel sebagai input DAN output
./run.sh analyze reviews.xlsx -o hasil.xlsx

# Cek dulu kolom apa yang terdeteksi, tanpa menjalankan model
./run.sh analyze reviews.csv --dry-run

# Ubah batch size (naikkan kalau RAM/VRAM besar)
./run.sh analyze reviews.csv --batch-size 64
```

**Nama kolom yang dikenali otomatis:** `text`, `review`, `review_text`, `content`, `comment`, `komentar`, `ulasan`, `body`, `message`, `feedback`, `deskripsi`. Untuk rating: `rating`, `score`, `stars`, `star`, `bintang`, `label`. Kalau tidak ada yang cocok, tool memilih kolom teks dengan rata-rata panjang terbesar; kalau masih gagal, gunakan `--text-col`.

**Format yang didukung:** `.csv` (UTF-8/UTF-8-BOM/latin-1, delimiter `,` `;` tab `|` terdeteksi otomatis), `.tsv`, `.xlsx`, `.xls`.

**Kecepatan:** ~6 review/detik di CPU 2 core (147 review = 24 detik). Dengan GPU jauh lebih cepat.

**Batas bahasa:** model ini **hanya dilatih pada teks Inggris**. Review berbahasa Indonesia akan tetap menghasilkan prediksi, tetapi akurasinya rendah (~57%) dan sering ber-confidence rendah (lihat `needs_review`). Lihat bagian *Validasi di Data Nyata* di atas.

## Results

Hasil inference 10 sampel unseen dari test split (8/10 benar). Angka final dari `results/metrics.json`:

| Text | True | Predicted | Confidence |
|---|---|---|---|
| Stopped working after a couple of weeks... | NEGATIVE | NEGATIVE | 0.971 |
| I have not yet got my order!!! | NEGATIVE | NEGATIVE | 0.515 |
| I had gotten these for a family member... | NEUTRAL | NEGATIVE ❌ | 0.707 |
| The presentation is great, but the flavor is ordinary... | NEUTRAL | NEUTRAL | 0.588 |
| Surprisingly ineffective product... | NEGATIVE | NEGATIVE | 0.661 |
| too loosen. cable is easy to dropping | NEGATIVE | NEUTRAL ❌ | 0.586 |
| A lot of the capsules were empty... | NEGATIVE | NEGATIVE | 0.808 |
| My niece love these balls!... | POSITIVE | POSITIVE | 0.981 |
| Ordered this for a Santa present... came broken! | NEGATIVE | NEGATIVE | 0.909 |
| Great product. Very satisfied... | POSITIVE | POSITIVE | 0.991 |

Model checkpoint (~256 MB) tersedia di [GitHub Releases](https://github.com/kusjay-space/sentiment-analysis/releases).

## Development Timeline

Timeline 22–30 Oktober 2026 (deadline submit 30 Okt, 12.00 WITA) ada di [TASKS.md](TASKS.md).

## Repo Structure

```
├── notebooks/     # Jupyter notebooks
├── src/           # Python scripts
├── results/       # metrics, predictions, plots
├── examples/      # contoh file input (reviews_sample.csv)
├── run.sh         # wrapper: jalankan tanpa activate venv
├── docs/          # GitHub Pages
├── model/         # checkpoint (via Releases)
└── LICENSE        # MIT
```

## License

MIT
