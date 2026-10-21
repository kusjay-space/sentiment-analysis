# SPEC — Sentiment Analysis Magang

## Arsitektur Sistem

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Hugging   │────▶│  Preprocess │────▶│   Baseline  │
│   Face DS   │     │   (clean)   │     │  TF-IDF+LR  │
└─────────────┘     └─────────────┘     └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │  Tokenize   │
                    │  (256 max)  │
                    └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │  DistilBERT │
                    │  fine-tune  │
                    └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │   Evaluate  │
                    │  (val/test) │
                    └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │  Inference  │
                    │  10 samples │
                    └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │  GitHub     │
                    │  Pages      │
                    └─────────────┘
```

## Struktur Repo

```
sentiment-analysis-magang/
├── README.md
├── PRD.md
├── REQUIREMENTS.md
├── TASKS.md
├── SPEC.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_finetune.ipynb
│   └── 04_inference.ipynb
├── src/
│   ├── preprocess.py
│   ├── baseline.py
│   ├── train.py
│   └── inference.py
├── results/
│   ├── metrics.json
│   ├── predictions.csv
│   └── confusion_matrix.png
├── model/
│   └── (checkpoint via Git LFS atau Releases)
└── docs/
    └── index.html
```

## Data Flow

### Training

```
Amazon Reviews EN (200k train)
    │
    ▼
Mapping: label 0-1 → NEG, 2 → NEU, 3-4 → POS
    │
    ▼
Preprocess: lowercase, strip HTML, truncate 256
    │
    ▼
Tokenize: DistilBERT tokenizer
    │
    ▼
Train: 2-3 epoch, batch 16, lr 2e-5
    │
    ▼
Save: checkpoint + tokenizer
```

### Inference

```
Test set (5k rows)
    │
    ▼
Ambil 10 sampel acak (seed=42)
    │
    ▼
Tokenize → Model → Softmax
    │
    ▼
Output: {text, true, pred, confidence}
    │
    ▼
Save: predictions.csv
```

## API / Interface

Tidak ada API. Static GitHub Pages only.

## Database

Tidak ada database. File-based only.

## Security

- No secrets in repo
- No API keys
- No personal data
- Model checkpoint di Releases, bukan repo

## Performance

| Metric | Target |
|---|---|
| Training time | ≤ 30 min (Colab T4) |
| Inference time | ≤ 1 detik/sample |
| Model size | ≤ 300 MB |
| Pages load | ≤ 3 detik |

## Dependencies

```
transformers>=5.0
torch>=2.0
datasets>=3.0
pandas>=2.0
scikit-learn>=1.6
matplotlib>=3.8
seaborn>=0.13
tqdm>=4.66
```

## Error Handling

| Error | Handling |
|---|---|
| Colab OOM | reduce batch size to 8 |
| Colab disconnect | checkpoint to Drive |
| Dataset download fail | retry, use cache |
| Model save fail | save to /tmp, upload manual |
| Inference error | log, skip sample, continue |

## Testing

| Test | Cara |
|---|---|
| Data loading | assert len(ds) > 0 |
| Mapping | assert label in {0,1,2} |
| Tokenization | assert max_len ≤ 256 |
| Model output | assert logits.shape == (batch, 3) |
| CSV format | assert columns match spec |
| Pages render | open in browser, visual check |

## Monitoring

Tidak ada monitoring (static site).

## Deployment

| Step | Cara |
|---|---|
| 1. Push code | git push origin main |
| 2. Upload model | GitHub Releases |
| 3. Enable Pages | Settings → Pages → docs/ |
| 4. Verify | buka URL, cek render |
