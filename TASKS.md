# TASKS — Sentiment Analysis Magang

Timeline: 22–30 Oktober 2026. Deadline submit: 30 Okt 2026, 12.00 WITA.

Prinsip: tidak ada angka hasil yang dilaporkan sebelum diproduksi run aktual. Validation split untuk seleksi model, test split untuk angka final (disentuh sekali di akhir).

## [22 Okt] Revisi Plan & Setup

- [x] Revisi dokumen plan (PRD, REQUIREMENTS, SPEC, TASKS, README, Pages)
- [x] Set git identity sebelum push pertama: `git config user.name` + `git config user.email`
- [x] Buat repo GitHub `kusjay-space/sentiment-analysis` + push plan awal
- [x] Setup venv: `python -m venv .venv && source .venv/bin/activate`
- [x] Install dependencies + pin requirements.txt dengan `==` setelah install sukses
- [x] Clone/cek struktur folder lokal
- [x] Load dataset `SetFit/amazon_reviews_multi_en`
- [x] EDA: distribusi label, panjang teks, 5 contoh sampel, catat imbalance (NEG 80k / NEU 40k / POS 80k)
- [x] Commit: "setup repo + EDA"

## [23 Okt] Baseline TF-IDF + LogReg

- [x] Preprocessing TF-IDF: lowercase via vectorizer + strip HTML
- [x] TF-IDF vectorizer + `class_weight=balanced` pada LogisticRegression
- [x] Evaluasi baseline di VALIDATION split (accuracy, F1-macro, confusion matrix)
- [x] Simpan metrics baseline ke `results/metrics.json` (tandai split = validation)
- [x] Commit: "baseline TF-IDF + LogReg"

## [24 Okt] DistilBERT Run 1 → Kalibrasi Target

- [x] Setup Colab notebook (GPU T4, fp16)
- [x] Subsample stratified 30k (12k NEG / 6k NEU / 12k POS), seed=42
- [x] Tokenisasi max_len 128 (teks raw, tokenizer menangani preprocessing)
- [x] Training run 1: batch 32, fp16, 2 epoch, eval_strategy=epoch, load_best_model_at_end=True
- [x] Evaluasi di VALIDATION split
- [x] Catat angka run 1 → kalibrasi target realistis (3-class DistilBERT realistis 86–89%)
- [x] Simpan checkpoint tiap epoch ke Drive
- [x] Commit: "DistilBERT run 1"

## [25 Okt] Run 2-3 + Evaluasi Final di TEST + Freeze

- [x] Run 2-3: perbaikan dari insight run 1 (lr, epoch, max_len, dsb)
- [x] Seleksi model terbaik berdasarkan VALIDATION split
- [x] Evaluasi FINAL di TEST split — tepat sekali, disentuh satu kali
- [x] Freeze artefak: `results/metrics.json`, `results/predictions.csv`, `results/confusion_matrix.png`
- [x] Commit: "fine-tune DistilBERT + evaluasi final test"

## [26 Okt] Pages + README + Release

- [x] GitHub Pages `docs/index.html` final pakai angka asli dari metrics.json
- [x] README final: tabel Model + Results diisi angka asli, bukan TBD/placeholder
- [x] Upload model checkpoint ke GitHub Releases
- [x] Tambah file LICENSE
- [x] Commit: "docs + release model"

## [27 Okt] Review End-to-End (Fresh Clone Test)

- [x] Fresh clone test: clone repo ke folder baru, ikut Quick Start dari nol
- [x] `pip install -r requirements.txt` di fresh venv sukses (reproducibility check)
- [x] Cek Pages live + render mobile
- [x] Cek semua link README valid
- [x] Commit: "review fixes"

## [28 Okt] Bug Fix + Polish

- [ ] Perbaiki bug hasil review end-to-end
- [ ] Polish README + halaman Pages
- [ ] Cek konsistensi angka di README, Pages, dan metrics.json (harus sama)
- [ ] Commit: "bug fix + polish"

## [29 Okt] Iterasi Training Opsional

Hanya dikerjakan jika review end-to-end lolos DAN masih ada sisa kuota Colab.

- [x] Iterasi training opsional: DistilBERT v2 (full 200k, 3 epoch, class weight NEUTRAL×2) — **dicoba, tidak dipakai**. Hasil validation: NEUTRAL F1 0.398→0.531 (+0.13) tetapi accuracy turun 74.9%→71.2% karena NEGATIVE tergerus. Trade-off tidak menguntungkan; v1 dipertahankan sebagai model submission.
- [x] Validasi data nyata: scraping 67 review Play Store US + 80 review Play Store ID → `results/real_world_test.json`. EN 74.6% (generalisasi terbukti), ID 57.5% (di luar domain).

## [30 Okt pagi] Submit

Deadline: sebelum 12.00 WITA.

- [ ] Final check semua deliverable
- [ ] Kirim email ke kotama@timedoor.net, cc hrd@timedoor.net
- [ ] Isi email: link repo GitHub + link GitHub Pages
- [ ] Konfirmasi email terkirim
