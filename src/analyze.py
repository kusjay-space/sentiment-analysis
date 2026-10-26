"""Batch sentiment analysis on YOUR OWN review file.

Reads a CSV/TSV/Excel file of reviews, predicts sentiment for every row, and
writes an enriched output file plus a summary report.

Usage:
    # Basic: analyse reviews.csv, output to results/analyzed.csv
    python -m src.analyze data_saya.csv

    # Choose the text column (default: auto-detect / first text-like column)
    python -m src.analyze data_saya.csv --text-col review

    # Choose where to write output
    python -m src.analyze data_saya.csv -o hasil/analisis.csv

    # Excel input/output
    python -m src.analyze reviews.xlsx -o hasil.xlsx

    # See what columns were detected without running the model
    python -m src.analyze data_saya.csv --dry-run

Input requirements:
    * A file with at least one column containing review text.
    * Optional columns (used if present): 'rating' / 'score' / 'stars' for
      automatic accuracy checking, 'id' for row tracking.
    * Encoding: UTF-8 recommended. Delimiter auto-detected for CSV.

Output:
    * An enriched file with added columns: sentiment, confidence, prob_negative,
      prob_neutral, prob_positive, needs_review (True when confidence < 0.5).
    * A printed summary: class distribution, mean confidence, low-confidence
      count, and accuracy vs rating (if a rating column exists).
    * results/analyze_summary.json with the same summary as machine-readable data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model" / "final"
RESULTS_DIR = REPO_ROOT / "results"

ID2LABEL = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# Column names we try, in priority order, when the user does not pass --text-col.
TEXT_COL_CANDIDATES = [
    "text", "review", "review_text", "content", "comment", "komentar",
    "ulasan", "body", "message", "feedback", "deskripsi",
]
RATING_COL_CANDIDATES = ["rating", "score", "stars", "star", "bintang", "label"]

MAX_LEN = 128
BATCH_SIZE = 32
LOW_CONFIDENCE_THRESHOLD = 0.5

_model = None
_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Model loading / inference
# --------------------------------------------------------------------------- #
def _load():
    """Lazy-load model + tokenizer once."""
    global _model, _tokenizer
    if _model is None:
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_DIR}.\n"
                "Download it first:\n"
                "  gh release download v1.0 -p model_final.zip\n"
                "  unzip model_final.zip -d model/\n"
                "Or see model/README.md"
            )
        _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        _model = DistilBertForSequenceClassification.from_pretrained(
            str(MODEL_DIR)
        ).to(_device)
        _model.eval()


def predict_batch(texts: List[str], batch_size: int = BATCH_SIZE) -> List[Dict]:
    """Predict sentiment for a list of texts. Returns one dict per text."""
    _load()
    out: List[Dict] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = _tokenizer(
            chunk,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        ).to(_device)
        with torch.no_grad():
            logits = _model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).cpu()
        for row in probs:
            pred_id = int(row.argmax())
            out.append(
                {
                    "sentiment": ID2LABEL[pred_id],
                    "confidence": round(float(row[pred_id]), 4),
                    "prob_negative": round(float(row[0]), 4),
                    "prob_neutral": round(float(row[1]), 4),
                    "prob_positive": round(float(row[2]), 4),
                }
            )
    return out


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def read_table(path: Path) -> pd.DataFrame:
    """Read CSV/TSV/Excel into a DataFrame, auto-detecting separator."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")

    # CSV: try encodings in order. Sniff the delimiter, but only among real
    # candidates -- pandas' sniffer can otherwise pick a letter (e.g. 't' from
    # the header word "text") when the file has a single column.
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            raw = path.read_text(encoding=enc)
        except UnicodeDecodeError as exc:
            last_err = exc
            continue

        first_line = raw.split("\n", 1)[0]
        delimiter = ","
        for cand in (",", ";", "\t", "|"):
            if cand in first_line:
                delimiter = cand
                break
        try:
            df = pd.read_csv(path, sep=delimiter, encoding=enc)
        except Exception:
            # Last resort: let pandas guess, restricted to sane delimiters.
            df = pd.read_csv(
                path, sep=None, engine="python", encoding=enc,
                # no restriction available here; comma fallback below
            )
        # A single unnamed column usually means the delimiter guess was wrong.
        if len(df.columns) == 1 and "," in raw.split("\n", 1)[0]:
            df = pd.read_csv(path, sep=",", encoding=enc)
        return df

    raise ValueError(
        f"Could not decode {path} ({last_err}). Convert it to UTF-8 and retry."
    )


def detect_text_column(df: pd.DataFrame) -> str:
    """Pick the most likely review-text column."""
    lowered = {str(c).lower().strip(): c for c in df.columns}
    for candidate in TEXT_COL_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    # Fall back to the object column with the longest average string length.
    best, best_len = None, -1.0
    for col in df.columns:
        if df[col].dtype == object:
            avg = df[col].dropna().astype(str).str.len().mean()
            if avg > best_len:
                best, best_len = col, avg
    if best is None:
        raise ValueError(
            "No text column found. Pass it explicitly with --text-col NAME.\n"
            f"Columns available: {list(df.columns)}"
        )
    return best


def detect_rating_column(df: pd.DataFrame, exclude: str) -> Optional[str]:
    """Pick a rating column for optional accuracy checking."""
    lowered = {str(c).lower().strip(): c for c in df.columns if c != exclude}
    for candidate in RATING_COL_CANDIDATES:
        if candidate in lowered:
            col = lowered[candidate]
            numeric = pd.to_numeric(df[col], errors="coerce")
            # Accept 1-5 stars or 0-4 codes.
            if numeric.notna().sum() > 0 and numeric.dropna().between(0, 5).all():
                return col
    return None


def rating_to_sentiment(value: float) -> Optional[str]:
    """Map a star rating to a sentiment label using the training scheme.

    Accepts either 1-5 stars or the dataset's 0-4 encoding. Only 3 stars
    (or code 2) is NEUTRAL; 1-2 negative; 4-5 positive.
    """
    if pd.isna(value):
        return None
    v = float(value)
    if v > 4:  # 5-star scale
        v -= 1  # -> 0-4
    if v <= 1:
        return "NEGATIVE"
    if v == 2:
        return "NEUTRAL"
    return "POSITIVE"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch sentiment analysis on your own review file."
    )
    parser.add_argument("input", type=Path, help="CSV/TSV/Excel file of reviews")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output file (default: results/analyzed_<inputname>.csv)",
    )
    parser.add_argument(
        "--text-col", default=None,
        help="Name of the column containing review text (auto-detected if omitted)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Inference batch size"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only show detected columns and row count; do not run the model",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: file not found: {args.input}", file=sys.stderr)
        return 1

    print(f"Reading {args.input} ...")
    try:
        df = read_table(args.input)
    except Exception as exc:
        print(f"ERROR: could not read {args.input}\n  {exc}", file=sys.stderr)
        return 1
    print(f"  {len(df):,} rows · {len(df.columns)} columns")

    try:
        text_col = args.text_col or detect_text_column(df)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if text_col not in df.columns:
        print(
            f"ERROR: column '{text_col}' not found.\n"
            f"Available: {list(df.columns)}",
            file=sys.stderr,
        )
        return 1
    print(f"  text column : {text_col}")

    rating_col = detect_rating_column(df, exclude=text_col)
    print(f"  rating column: {rating_col or '(none — accuracy check skipped)'}")

    if args.dry_run:
        print("\nDry run — no model executed. Preview of first rows:")
        print(df[[text_col] + ([rating_col] if rating_col else [])].head(5).to_string())
        return 0

    # Drop rows with empty text but remember how many were skipped.
    total_rows = len(df)
    df = df[df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")].copy()
    skipped = total_rows - len(df)
    if skipped:
        print(f"  skipped {skipped} row(s) with empty text")

    texts = df[text_col].astype(str).tolist()
    print(f"\nRunning inference on {len(texts):,} reviews (device: {_device}) ...")
    preds = predict_batch(texts, batch_size=args.batch_size)

    out = pd.DataFrame(preds)
    result = pd.concat([df.reset_index(drop=True), out], axis=1)
    result["needs_review"] = result["confidence"] < LOW_CONFIDENCE_THRESHOLD

    # Optional accuracy check against the rating column.
    summary: Dict[str, object] = {
        "input_file": str(args.input),
        "rows_analyzed": int(len(result)),
        "rows_skipped": int(skipped),
        "text_column": text_col,
        "rating_column": rating_col,
        "device": _device,
        "model": "distilbert-base-uncased (fine-tuned, 3-class)",
        "label_distribution": result["sentiment"].value_counts().to_dict(),
        "mean_confidence": round(float(result["confidence"].mean()), 4),
        "low_confidence_rows": int(result["needs_review"].sum()),
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
    }

    if rating_col:
        actual = result[rating_col].map(rating_to_sentiment)
        mask = actual.notna()
        if mask.any():
            acc = float((result.loc[mask, "sentiment"] == actual[mask]).mean())
            summary["accuracy_vs_rating"] = round(acc, 4)
            summary["rows_with_rating"] = int(mask.sum())
            summary["confusion"] = (
                pd.crosstab(actual[mask], result.loc[mask, "sentiment"]).to_dict()
            )

    # Write outputs.
    if args.output is None:
        RESULTS_DIR.mkdir(exist_ok=True)
        out_path = RESULTS_DIR / f"analyzed_{args.input.stem}.csv"
    else:
        out_path = args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() in {".xlsx", ".xls"}:
        result.to_excel(out_path, index=False)
    else:
        result.to_csv(out_path, index=False)

    summary_path = RESULTS_DIR / "analyze_summary.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # Human-readable report.
    print("\n" + "=" * 62)
    print("HASIL ANALISIS")
    print("=" * 62)
    print(f"  Review dianalisis      : {len(result):,}")
    print(f"  Rata-rata confidence   : {result['confidence'].mean():.3f}")
    print(f"  Perlu review manual    : {int(result['needs_review'].sum()):,} "
          f"(confidence < {LOW_CONFIDENCE_THRESHOLD})")
    print("\n  Distribusi sentimen:")
    for label in ("POSITIVE", "NEUTRAL", "NEGATIVE"):
        n = int((result["sentiment"] == label).sum())
        pct = n / len(result) * 100 if len(result) else 0
        bar = "█" * int(pct / 2)
        print(f"    {label:<9} {n:>6,}  ({pct:>5.1f}%)  {bar}")
    if "accuracy_vs_rating" in summary:
        print(f"\n  Akurasi vs rating asli : {summary['accuracy_vs_rating']:.2%} "
              f"({summary['rows_with_rating']:,} baris punya rating)")
    print("\n" + "-" * 62)
    print(f"  Output  : {out_path}")
    print(f"  Ringkasan: {summary_path}")
    print("-" * 62)
    print("\n  Kolom baru di output:")
    print("    sentiment      -> POSITIVE / NEUTRAL / NEGATIVE")
    print("    confidence     -> 0..1 (keyakinan model)")
    print("    prob_negative  -> probabilitas NEGATIVE")
    print("    prob_neutral   -> probabilitas NEUTRAL")
    print("    prob_positive  -> probabilitas POSITIVE")
    print("    needs_review   -> True bila confidence < 0.5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
