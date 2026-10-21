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

| Model | Akurasi | F1-macro |
|---|---|---|
| TF-IDF + LogisticRegression | TBD — menunggu hasil training | TBD — menunggu hasil training |
| DistilBERT (fine-tuned) | TBD — menunggu hasil training | TBD — menunggu hasil training |

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
python src/baseline.py

# Training (DistilBERT) — butuh GPU, gunakan Colab
python src/train.py

# Inference
python src/inference.py
```

## Results

Hasil akan diisi setelah inference. Angka final diambil dari `results/metrics.json` (evaluasi test split) dan `results/predictions.csv` (10 sampel unseen).

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
