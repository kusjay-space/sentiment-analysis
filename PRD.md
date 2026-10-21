# PRD — Sentiment Analysis App Reviews

## 1. Tujuan

Bangun pipeline NLP yang mengklasifikasi review teks menjadi **positif, netral, atau negatif**. Sesuai brief Timedoor: *"product team wants to automatically tag incoming user feedback."*

Output akhir: repo GitHub publik + halaman GitHub Pages statis yang menjelaskan metodologi dan menampilkan hasil prediksi.

## 2. Konteks

- **Posisi:** AI Engineer Intern, Timedoor
- **Deadline:** 30 Oktober 2026, 12.00 WITA
- **Tingkat:** Pemula (no prior ML experience)
- **Evaluasi:** ML Fundamentals 40%, Code Quality 20%, Communication 20%, Engineering 20%

## 3. Scope

### In Scope
- Dataset Amazon Reviews EN (HuggingFace `SetFit/amazon_reviews_multi_en`)
- Mapping bintang 1-5 ke 3 kelas: negatif (1-2), netral (3), positif (4-5)
- Baseline: TF-IDF + LogisticRegression
- Model utama: fine-tune DistilBERT (`distilbert/distilbert-base-uncased`)
- Evaluasi: accuracy, F1-macro, confusion matrix
- Inference: 10 sampel unseen → `predictions.csv`
- GitHub Pages statis: metodologi + hasil
- README dengan link Pages di baris atas

### Out of Scope
- Web app / API / live inference
- Multi-bahasa
- Model deployment (Docker, cloud)
- Real-time inference
- Model quantization / ONNX

## 4. Dataset

| Properti | Nilai |
|---|---|
| Sumber | HuggingFace `SetFit/amazon_reviews_multi_en` |
| Train | 200.000 rows |
| Validation | 5.000 rows |
| Test | 5.000 rows |
| Label asli | bintang 0-4 (0=1 star, 4=5 stars) |
| Mapping | 0-1 → NEGATIVE, 2 → NEUTRAL, 3-4 → POSITIVE |
| Bahasa | Inggris |
| Format | `id`, `text`, `label`, `label_text` |

## 5. Model

| Model | Peran | Library | Train |
|---|---|---|---|
| TF-IDF + LogisticRegression | baseline | sklearn | 5 detik CPU |
| DistilBERT base | utama | transformers + PyTorch | 5-10 menit Colab T4 |

Hyperparameter DistilBERT:
- Epoch: 2-3
- Batch size: 16
- Learning rate: 2e-5
- Max length: 256 tokens
- Optimizer: AdamW

## 6. Deliverables

| # | Item | Path |
|---|---|---|
| 1 | Repo GitHub publik | `kusjay-space/sentiment-analysis-magang` |
| 2 | README.md | root, link Pages di baris 1 |
| 3 | Notebook training | `notebooks/train.ipynb` |
| 4 | Script training | `src/train.py` |
| 5 | Script inference | `src/inference.py` |
| 6 | Hasil prediksi | `results/predictions.csv` |
| 7 | Confusion matrix | `results/confusion_matrix.png` |
| 8 | Metrik | `results/metrics.json` |
| 9 | GitHub Pages | `docs/index.html` |
| 10 | requirements.txt | root |

## 7. Success Criteria

| Kriteria | Target |
|---|---|
| Akurasi baseline | ≥ 80% |
| Akurasi DistilBERT | ≥ 85% |
| 10 sampel unseen | 10/10 diprediksi, minimal 8 benar |
| Repo publik | bisa di-clone tanpa error |
| README | jelas, ada link Pages, cara install |
| Pages | live, tampil 10 hasil + metodologi |
| Email | terkirim ke kotama@timedoor.net cc hrd@timedoor.net |

## 8. Risiko

| Risiko | Mitigasi |
|---|---|
| Colab sesi mati | checkpoint ke Google Drive tiap epoch |
| Akurasi rendah | jujur di README, analisis penyebab |
| Model 250MB > limit GitHub | upload ke Releases, link di README |
| Waktu mepet | baseline dulu, DistilBERT bisa di-skip kalau kepepet |

## 9. Timeline (11 hari)

| Hari | Aktivitas | Output |
|---|---|---|
| 1 | Setup repo, load dataset, EDA | notebook EDA |
| 2 | Baseline TF-IDF + LogReg | akurasi baseline |
| 3-4 | Fine-tune DistilBERT di Colab | model checkpoint |
| 5 | Evaluasi + inference 10 sampel | metrics.json + predictions.csv |
| 6 | Buat GitHub Pages | docs/index.html |
| 7 | README + requirements.txt | repo lengkap |
| 8 | Review, perbaiki | repo final |
| 9-10 | Buffer eksperimen | — |
| 11 | Kirim email | submitted |
