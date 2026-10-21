# Sentiment Analysis — App Reviews

> **🔗 Live Demo:** [https://kusjay-space.github.io/sentiment-analysis](https://kusjay-space.github.io/sentiment-analysis)

> **🚧 Work in progress** — hasil akan diupdate setelah training. Semua angka di dokumen ini diisi dari `results/metrics.json` hasil run aktual.

Pipeline NLP untuk klasifikasi sentimen review aplikasi menjadi **positif, netral, atau negatif**.

## Overview

Project ini adalah submission untuk AI Engineer Intern assessment di Timedoor. Menggunakan dataset Amazon Reviews EN dan fine-tune DistilBERT untuk mengklasifikasi review.

## Dataset

- **Sumber:** [Amazon Reviews Multi EN](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en)
- **Train:** 200.000 rows
- **Validation:** 5.000 rows
- **Test:** 5.000 rows
- **Mapping:** bintang 1-2 → NEGATIVE, 3 → NEUTRAL, 4-5 → POSITIVE

## Model

| Model | Split | Akurasi | F1-macro |
|---|---|---|---|
| TF-IDF + LogisticRegression (tuned) | validation | 70.2% | 0.674 |
| DistilBERT (fine-tuned) | **test** | **75.1%** | **0.687** |

Kedua model dievaluasi di split yang sama (train subsample → seleksi di validation → angka final di test, disentuh sekali). DistilBERT mengalahkan baseline **+4.9% accuracy**.

**F1 per kelas (DistilBERT, test):** NEGATIVE 0.806, NEUTRAL 0.403, POSITIVE 0.851. NEUTRAL tetap kelas terlemah — bintang 3 secara inheren ambigu (lihat analisis di bawah). Inference 10 sampel unseen: **8/10 benar** (`results/predictions.csv`).

### Kenapa baseline mentok di ~70%

Sweep hyperparameter TF-IDF+LogReg (lihat `src/baseline_tuned.py`) menunjukkan ceiling ≈70% pada constraint 60k train subsample — bigram/trigram, char n-gram, C besar, dan 510k features semuanya < 71%. Bottlenecknya adalah kelas NEUTRAL (F1 0.47): bintang 3 secara inheren ambigu dan TF-IDF tidak menangkap konteks. Ini motivasi utama fine-tune DistilBERT.

### Hasil DistilBERT vs ekspektasi

Target awal 85% tidak tercapai (real: 75.1% test). Penyebab: (1) training hanya 2 epoch di subset 30k — model kontekstual butuh lebih banyak data/epoch untuk menangkap nuansa NEUTRAL; (2) NEUTRAL (bintang 3) memang ambigu secara linguistik — bahkan DistilBERT hanya F1 0.40 di kelas ini. Namun DistilBERT tetap unggul **+4.9% accuracy** atas baseline dan jauh lebih baik di NEGATIVE/POSITIVE (0.81/0.85). Perbaikan lanjutan: train full 200k, 3-4 epoch, atau focal loss untuk kelas minoritas.

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
