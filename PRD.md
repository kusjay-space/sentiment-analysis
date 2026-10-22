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
| Train | 200.000 rows (dipakai 30.000 subsample stratified: 12k NEG / 6k NEU / 12k POS) |
| Validation | 5.000 rows |
| Test | 5.000 rows |
| Label asli | Bintang 1–5 (dikodekan sebagai `label` 0–4 di dataset: label 0 = bintang 1, label 4 = bintang 5) |
| Mapping | Bintang 1–2 → NEGATIVE, bintang 3 → NEUTRAL, bintang 4–5 → POSITIVE (kode: label 0–1 → NEG, 2 → NEU, 3–4 → POS) |
| Bahasa | Inggris |
| Format | `id`, `text`, `label`, `label_text` |

## 5. Model

| Model | Peran | Library | Train |
|---|---|---|---|
| TF-IDF + LogisticRegression | baseline | sklearn | 5 detik CPU |
| DistilBERT base | utama | transformers + PyTorch | ~30 menit untuk subset 30k, 2 epoch, fp16 di T4 |

Hyperparameter DistilBERT:
- Epoch: 2
- Batch size: 32
- Learning rate: 2e-5
- Max length: 128 tokens
- FP16: ya (Colab T4)
- Optimizer: AdamW

## 6. Deliverables

| # | Item | Path |
|---|---|---|
| 1 | Repo GitHub publik | `kusjay-space/sentiment-analysis` |
| 2 | README.md | root, link Pages di baris 1 |
| 3 | Notebook training | `notebooks/train.ipynb` |
| 4 | Script training | `src/train.py` |
| 5 | Script inference | `src/inference.py` |
| 6 | Hasil prediksi | `results/predictions.csv` |
| 7 | Confusion matrix | `results/confusion_matrix.png` |
| 8 | Metrik | `results/metrics.json` |
| 9 | GitHub Pages | `docs/index.html` |
| 10 | requirements.txt | root |
| 11 | LICENSE | root |

## 7. Success Criteria

| Kriteria | Target |
|---|---|
| Akurasi baseline | ≥ 75% (realisasi: 70.2% validation — di bawah target, dianalisis di README) |
| Akurasi DistilBERT | ≥ 85% (realisasi: **75.1% test** — tidak tercapai, penyebab dianalisis di README) |
| Integritas angka | Angka yang dilaporkan harus berasal dari metrics.json yang diproduksi run aktual — tidak ada placeholder |
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
| Kuota Colab habis | subset 30k + fp16 + batch 32, checkpoint tiap epoch ke Drive |

## 9. Timeline (22–30 Oktober 2026)

| Tanggal | Aktivitas | Output |
|---|---|---|
| 22 Okt | Revisi plan, setup repo GitHub + venv, EDA | plan final + notebook EDA |
| 23 Okt | Baseline TF-IDF + LogReg, evaluasi di validation split | metrik baseline |
| 24 Okt | DistilBERT run 1 (subset 30k stratified, max_len 128, batch 32, fp16, 2 epoch) | checkpoint + angka kalibrasi target |
| 25 Okt | Run 2-3 perbaikan, evaluasi final di test split (sekali), freeze artefak | metrics.json + predictions.csv + confusion_matrix.png |
| 26 Okt | Pages + README final pakai angka asli, release model, LICENSE | docs/index.html + repo lengkap |
| 27 Okt | Review end-to-end (fresh clone test) | repo terverifikasi |
| 28 Okt | Bug fix + polish | repo final |
| 29 Okt | Iterasi training opsional (jika review lolos + sisa kuota Colab) | — |
| 30 Okt pagi | Submit email sebelum 12.00 WITA ke kotama@timedoor.net cc hrd@timedoor.net | submitted |
