"""Evaluate the fine-tuned DistilBERT model and dump predictions.

Run as a module (requires a trained model in ``model/final``)::

    python -m src.inference

Writes ``results/predictions.csv`` (10 random test samples), ``results/metrics.json``
(full-test metrics) and ``results/confusion_matrix.png``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # Headless-safe backend.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.preprocess import (
    ID2LABEL,
    apply_mapping,
    get_texts_labels,
    load_dataset_local,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
MODEL_DIR = REPO_ROOT / "model" / "final"

SEED = 42
MAX_LEN = 128
TEXT_TRUNCATE = 80
N_SAMPLES = 10


# Model & inference
def load_model(model_dir: Path = MODEL_DIR):
    """Load the fine-tuned model and tokenizer from ``model_dir``."""
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_dir}. Run `python -m src.train` first."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    return model, tokenizer


def _predict_batch(
    texts: List[str], model, tokenizer, batch_size: int = 64
) -> tuple[List[int], List[float]]:
    """Return predicted sentiment ids and softmax confidences for ``texts``."""
    predictions: List[int] = []
    confidences: List[float] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            predictions.extend(pred.tolist())
            confidences.extend(conf.tolist())

    return predictions, confidences


# Metrics & plotting
def f1_per_class(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Return F1 per class keyed by sentiment name."""
    scores = f1_score(y_true, y_pred, labels=list(ID2LABEL), average=None)
    return {ID2LABEL[i]: float(scores[i]) for i in ID2LABEL}


def save_confusion_matrix(
    y_true: List[int], y_pred: List[int], output_path: Path
) -> None:
    """Render and save a 3x3 seaborn heatmap confusion matrix."""
    labels = [ID2LABEL[i] for i in ID2LABEL]
    matrix = confusion_matrix(y_true, y_pred, labels=list(ID2LABEL))

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("DistilBERT Confusion Matrix (test)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# Main
def run() -> Dict[str, object]:
    """Evaluate on the full test split and write inference artifacts."""
    dataset = load_dataset_local()
    test_ds = apply_mapping(dataset["test"])
    test_texts, test_labels = get_texts_labels(test_ds)

    model, tokenizer = load_model()
    test_pred, _ = _predict_batch(test_texts, model, tokenizer)

    metrics: Dict[str, object] = {
        "accuracy": float(accuracy_score(test_labels, test_pred)),
        "f1_macro": float(f1_score(test_labels, test_pred, average="macro")),
        "f1_weighted": float(f1_score(test_labels, test_pred, average="weighted")),
        "f1_per_class": f1_per_class(test_labels, test_pred),
        "n_test": len(test_labels),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    save_confusion_matrix(
        test_labels, test_pred, RESULTS_DIR / "confusion_matrix.png"
    )

    rng = random.Random(SEED)
    sample_idx = rng.sample(range(len(test_texts)), min(N_SAMPLES, len(test_texts)))

    sample_texts = [test_texts[idx] for idx in sample_idx]
    sample_pred, sample_conf = _predict_batch(sample_texts, model, tokenizer)

    rows = []
    for idx, pred, conf in zip(sample_idx, sample_pred, sample_conf):
        rows.append(
            {
                "text": test_texts[idx][:TEXT_TRUNCATE],
                "true_label": ID2LABEL[test_labels[idx]],
                "predicted_label": ID2LABEL[pred],
                "confidence": round(float(conf), 3),
            }
        )

    predictions_df = pd.DataFrame(
        rows, columns=["text", "true_label", "predicted_label", "confidence"]
    )
    predictions_df.to_csv(RESULTS_DIR / "predictions.csv", index=False)

    print(f"Metrik test: accuracy={metrics['accuracy']:.4f} "
          f"f1_macro={metrics['f1_macro']:.4f} "
          f"f1_weighted={metrics['f1_weighted']:.4f}")
    print("10 prediksi acak dari test set (seed=42):")
    for i, row in enumerate(rows, 1):
        print(
            f"  [{i:02d}] asli={row['true_label']:<8} prediksi={row['predicted_label']:<8} "
            f"keyakinan={row['confidence']:.3f} | {row['text']}"
        )
    print(f"  tersimpan {(RESULTS_DIR / 'predictions.csv').relative_to(REPO_ROOT)}")
    print(f"  tersimpan {(RESULTS_DIR / 'metrics.json').relative_to(REPO_ROOT)}")
    print(
        f"  tersimpan {(RESULTS_DIR / 'confusion_matrix.png').relative_to(REPO_ROOT)}"
    )
    return metrics


if __name__ == "__main__":
    run()