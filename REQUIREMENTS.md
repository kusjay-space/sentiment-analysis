# REQUIREMENTS — Sentiment Analysis Magang

## Fungsional

| ID | Requirement | Prioritas |
|---|---|---|
| FR-01 | Load dataset Amazon Reviews EN dari HuggingFace | Must |
| FR-02 | Mapping bintang 1-5 ke 3 kelas sentimen | Must |
| FR-03 | Split train/val/test (sudah disediakan dataset) | Must |
| FR-04 | EDA: distribusi label, panjang teks, contoh sampel | Must |
| FR-05 | Baseline TF-IDF + LogisticRegression (`class_weight=balanced`) | Must |
| FR-06 | Fine-tune DistilBERT | Must |
| FR-07 | Evaluasi: accuracy, F1, confusion matrix | Must |
| FR-08 | Inference 10 sampel unseen → CSV | Must |
| FR-09 | Simpan model checkpoint | Must |
| FR-10 | GitHub Pages statis | Must |
| FR-11 | README lengkap dengan link Pages | Must |
| FR-12 | requirements.txt | Must |

## Non-Fungsional

| ID | Requirement | Target |
|---|---|---|
| NFR-01 | Akurasi baseline | ≥ 75% |
| NFR-02 | Akurasi DistilBERT | Target ≥ 85%; **realisasi 75.1% (test)** — tidak tercapai, penyebab dianalisis di README §Hasil DistilBERT vs ekspektasi |
| NFR-03 | Training time | ≤ 30 menit di Colab |
| NFR-04 | Repo size | < 100 MB (model di Releases) |
| NFR-05 | README | bahasa Indonesia, jelas, step-by-step |
| NFR-06 | Pages | mobile-friendly, load < 3 detik |
| NFR-07 | Reproducibility | seed=42 global, requirements.txt dipin `==`, fresh-clone test lolos |

## Teknis

| Komponen | Versi | Catatan |
|---|---|---|
| Python | ≥ 3.10 | Colab default 3.12 |
| transformers | ≥ 5.0 | HuggingFace |
| torch | ≥ 2.0 | PyTorch |
| sklearn | ≥ 1.6 | baseline |
| pandas | ≥ 2.0 | data handling |
| datasets | ≥ 3.0 | HuggingFace datasets |
| matplotlib | ≥ 3.8 | confusion matrix |
| seaborn | ≥ 0.13 | visualisasi |
| Preprocessing DistilBERT | — | teks raw, tokenizer menangani preprocessing |
| Preprocessing TF-IDF | — | lowercase via vectorizer + strip HTML |

## Hardware

| Item | Kebutuhan |
|---|---|
| Training | Google Colab T4 (gratis) |
| Laptop | RAM ≥ 4 GB, browser, internet |
| Storage | ~500 MB (model + data) |
| GPU lokal | tidak perlu |

## Data

| Item | Detail |
|---|---|
| Sumber | `SetFit/amazon_reviews_multi_en` |
| Train | 30.000 rows untuk DistilBERT (subsample stratified: 12k NEG / 6k NEU / 12k POS, seed 42). Baseline TF-IDF memakai 60.000 rows (20k/kelas) |
| Validation | 5.000 rows |
| Test | 5.000 rows |
| Label | Bintang 1–5 → mapping ke NEGATIVE/NEUTRAL/POSITIVE. Bintang 1–2 = NEG, bintang 3 = NEU, bintang 4–5 = POS (kode dataset: label 0–1 → NEG, 2 → NEU, 3–4 → POS). Hanya bintang 3 yang menjadi NEUTRAL, sehingga NEUTRAL minoritas (20%) |
| Bahasa | Inggris |
| Imbalance | NEG 12k / NEU 6k / POS 12k (rasio 2:1:2). NEUTRAL kelas minoritas (20%) karena hanya bintang 3 yang masuk NEUTRAL |
| Preprocessing | TF-IDF: lowercase + strip HTML; DistilBERT: teks raw (tokenizer menangani), max 128 token |

## Model

| Model | Input | Output | Ukuran |
|---|---|---|---|
| TF-IDF + LogReg | teks raw | kelas + probabilitas | <1 MB |
| DistilBERT | token IDs | logits 3 kelas | ~250 MB |

## Output

| File | Format | Isi |
|---|---|---|
| predictions.csv | CSV | text, true_label, predicted, confidence |
| metrics.json | JSON | accuracy, f1_macro, f1_weighted |
| confusion_matrix.png | PNG | 3x3 matrix |
| model/ | folder | checkpoint DistilBERT |
