# REQUIREMENTS — Sentiment Analysis Magang

## Fungsional

| ID | Requirement | Prioritas |
|---|---|---|
| FR-01 | Load dataset Amazon Reviews EN dari HuggingFace | Must |
| FR-02 | Mapping bintang 1-5 ke 3 kelas sentimen | Must |
| FR-03 | Split train/val/test (sudah disediakan dataset) | Must |
| FR-04 | EDA: distribusi label, panjang teks, contoh sampel | Must |
| FR-05 | Baseline TF-IDF + LogisticRegression | Must |
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
| NFR-01 | Akurasi baseline | ≥ 80% |
| NFR-02 | Akurasi DistilBERT | ≥ 85% |
| NFR-03 | Training time | ≤ 30 menit di Colab |
| NFR-04 | Repo size | < 100 MB (model di Releases) |
| NFR-05 | README | bahasa Indonesia, jelas, step-by-step |
| NFR-06 | Pages | mobile-friendly, load < 3 detik |

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
| Train | 200.000 rows |
| Validation | 5.000 rows |
| Test | 5.000 rows |
| Label | 0-4 → mapping ke NEGATIVE/NEUTRAL/POSITIVE |
| Bahasa | Inggris |
| Preprocessing | lowercase, hapus HTML tag, max 256 token |

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
