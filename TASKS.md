# TASKS — Sentiment Analysis Magang

Timeline: 22–30 Oktober 2026. Deadline submit: 30 Okt 2026, 12.00 WITA.

Prinsip: tidak ada angka hasil yang dilaporkan sebelum diproduksi run aktual. Validation split untuk seleksi model, test split untuk angka final (disentuh sekali di akhir).

## [22 Okt] Revisi Plan & Setup

- [ ] Revisi dokumen plan (PRD, REQUIREMENTS, SPEC, TASKS, README, Pages)
- [ ] Set git identity sebelum push pertama: `git config user.name` + `git config user.email`
- [ ] Buat repo GitHub `kusjay-space/sentiment-analysis` + push plan awal
- [ ] Setup venv: `python -m venv .venv && source .venv/bin/activate`
- [ ] Install dependencies + pin requirements.txt dengan `==` setelah install sukses
- [ ] Clone/cek struktur folder lokal
- [ ] Load dataset `SetFit/amazon_reviews_multi_en`
- [ ] EDA: distribusi label, panjang teks, 5 contoh sampel, catat imbalance (NEG 80k / NEU 40k / POS 80k)
- [ ] Commit: "setup repo + EDA"

## [23 Okt] Baseline TF-IDF + LogReg

- [ ] Preprocessing TF-IDF: lowercase via vectorizer + strip HTML
- [ ] TF-IDF vectorizer + `class_weight=balanced` pada LogisticRegression
- [ ] Evaluasi baseline di VALIDATION split (accuracy, F1-macro, confusion matrix)
- [ ] Simpan metrics baseline ke `results/metrics.json` (tandai split = validation)
- [ ] Commit: "baseline TF-IDF + LogReg"

## [24 Okt] DistilBERT Run 1 → Kalibrasi Target

- [ ] Setup Colab notebook (GPU T4, fp16)
- [ ] Subsample stratified 30k (12k NEG / 6k NEU / 12k POS), seed=42
- [ ] Tokenisasi max_len 128 (teks raw, tokenizer menangani preprocessing)
- [ ] Training run 1: batch 32, fp16, 2 epoch, eval_strategy=epoch, load_best_model_at_end=True
- [ ] Evaluasi di VALIDATION split
- [ ] Catat angka run 1 → kalibrasi target realistis (3-class DistilBERT realistis 86–89%)
- [ ] Simpan checkpoint tiap epoch ke Drive
- [ ] Commit: "DistilBERT run 1"

## [25 Okt] Run 2-3 + Evaluasi Final di TEST + Freeze

- [ ] Run 2-3: perbaikan dari insight run 1 (lr, epoch, max_len, dsb)
- [ ] Seleksi model terbaik berdasarkan VALIDATION split
- [ ] Evaluasi FINAL di TEST split — tepat sekali, disentuh satu kali
- [ ] Freeze artefak: `results/metrics.json`, `results/predictions.csv`, `results/confusion_matrix.png`
- [ ] Commit: "fine-tune DistilBERT + evaluasi final test"

## [26 Okt] Pages + README + Release

- [ ] GitHub Pages `docs/index.html` final pakai angka asli dari metrics.json
- [ ] README final: tabel Model + Results diisi angka asli, bukan TBD/placeholder
- [ ] Upload model checkpoint ke GitHub Releases
- [ ] Tambah file LICENSE
- [ ] Commit: "docs + release model"

## [27 Okt] Review End-to-End (Fresh Clone Test)

- [ ] Fresh clone test: clone repo ke folder baru, ikut Quick Start dari nol
- [ ] `pip install -r requirements.txt` di fresh venv sukses (reproducibility check)
- [ ] Cek Pages live + render mobile
- [ ] Cek semua link README valid
- [ ] Commit: "review fixes"

## [28 Okt] Bug Fix + Polish

- [ ] Perbaiki bug hasil review end-to-end
- [ ] Polish README + halaman Pages
- [ ] Cek konsistensi angka di README, Pages, dan metrics.json (harus sama)
- [ ] Commit: "bug fix + polish"

## [29 Okt] Iterasi Training Opsional

Hanya dikerjakan jika review end-to-end lolos DAN masih ada sisa kuota Colab.

- [ ] Iterasi training opsional (hyperparameter tambahan, augmentasi, dsb)
- [ ] Jika ada peningkatan, update angka di semua dokumen dari metrics.json terbaru
- [ ] Commit: "opsional: iterasi training"

## [30 Okt pagi] Submit

Deadline: sebelum 12.00 WITA.

- [ ] Final check semua deliverable
- [ ] Kirim email ke kotama@timedoor.net, cc hrd@timedoor.net
- [ ] Isi email: link repo GitHub + link GitHub Pages
- [ ] Konfirmasi email terkirim
