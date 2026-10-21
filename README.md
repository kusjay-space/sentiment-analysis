# Sentiment Analysis — App Reviews

> **🔗 Live Demo:** [https://kusjay-space.github.io/sentiment-analysis](https://kusjay-space.github.io/sentiment-analysis)

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
| TF-IDF + LogisticRegression | 85.2% | 0.84 |
| DistilBERT (fine-tuned) | 91.7% | 0.91 |

## Quick Start

```bash
# Clone
git clone https://github.com/kusjay-space/sentiment-analysis.git
cd sentiment-analysis

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

| Text | True | Predicted | Confidence |
|---|---|---|---|
| "Great app, very fast" | POSITIVE | POSITIVE | 0.994 |
| "Crashes every time" | NEGATIVE | NEGATIVE | 0.981 |
| "It's okay I guess" | POSITIVE | NEGATIVE | 0.612 |

## Repo Structure

```
├── notebooks/     # Jupyter notebooks
├── src/           # Python scripts
├── results/       # metrics, predictions, plots
├── docs/          # GitHub Pages
└── model/         # checkpoint (via Releases)
```

## License

MIT
