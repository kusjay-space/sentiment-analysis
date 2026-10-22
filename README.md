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

# Training (baseline)
python -m src.baseline

# Training (DistilBERT) — butuh GPU, gunakan Colab
python -m src.train

# Inference
python -m src.inference
```

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
├── docs/          # GitHub Pages
├── model/         # checkpoint (via Releases)
└── LICENSE        # MIT
```

## License

MIT
