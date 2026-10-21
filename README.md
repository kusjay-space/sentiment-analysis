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

| Model | Akurasi (validation) | F1-macro |
|---|---|---|
| TF-IDF + LogisticRegression (tuned) | **70.2%** | **0.674** |
| DistilBERT (fine-tuned) | TBD — training di Colab | TBD |

Baseline dievaluasi di validation split (5.000 rows). F1 per kelas: NEGATIVE 0.748, NEUTRAL 0.468, POSITIVE 0.805. NEUTRAL adalah kelas terlemah karena imbalance 2:1:2 (lihat `notebooks/01_eda.ipynb`). DistilBERT akan dievaluasi di test split sekali, setelah training selesai.

### Kenapa baseline mentok di ~70%

Sweep hyperparameter TF-IDF+LogReg (lihat `src/baseline_tuned.py`) menunjukkan ceiling ≈70% pada constraint 60k train subsample — bigram/trigram, char n-gram, C besar, dan 510k features semuanya < 71%. Bottlenecknya adalah kelas NEUTRAL (F1 0.47): bintang 3 secara inheren ambigu dan TF-IDF tidak menangkap konteks. Ini motivasi utama fine-tune DistilBERT — model kontekstual diharapkan menembus 85%.

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
