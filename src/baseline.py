"""TF-IDF + Logistic Regression baseline.

Run as a module::

    python -m src.baseline

Trains on a stratified train subsample and evaluates on the full validation
split, writing metrics and a confusion matrix under ``results/``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # Headless-safe backend.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from src.preprocess import (
    ID2LABEL,
    apply_mapping,
    get_texts_labels,
    load_dataset_local,
    stratified_subsample,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

SEED = 42
N_PER_CLASS = 20_000
MAX_FEATURES = 50_000
NGRAM_RANGE = (1, 2)

_HTML_TAG_RE = re.compile(r"<[^>]+>")


# Text preprocessing
def strip_html(text: str) -> str:
    """Remove HTML tags from a raw review string."""
    return _HTML_TAG_RE.sub(" ", text)


def preprocess_texts(texts: List[str]) -> List[str]:
    """Apply HTML stripping to every review in ``texts``."""
    return [strip_html(text) for text in texts]


# Metrics & plotting
def f1_per_class(
    y_true: List[int], y_pred: List[int]
) -> Dict[str, float]:
    """Return F1 per class keyed by sentiment name."""
    scores = f1_score(y_true, y_pred, labels=list(ID2LABEL), average=None)
    return {ID2LABEL[i]: float(scores[i]) for i in ID2LABEL}


def save_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    output_path: Path,
) -> None:
    """Render and save a 3x3 seaborn heatmap confusion matrix."""
    labels = [ID2LABEL[i] for i in ID2LABEL]
    matrix = confusion_matrix(y_true, y_pred, labels=list(ID2LABEL))

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Baseline (TF-IDF + LogisticRegression) Confusion Matrix")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# Pipeline
def build_pipeline() -> Tuple[TfidfVectorizer, LogisticRegression]:
    """Create the unfitted vectorizer and classifier."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=SEED,
    )
    return vectorizer, classifier


# Main
def run() -> Dict[str, float]:
    """Train the baseline, evaluate on validation, and persist artifacts."""
    dataset = load_dataset_local()
    train_full = apply_mapping(dataset["train"])
    val_full = apply_mapping(dataset["validation"])

    train_ds = stratified_subsample(train_full, N_PER_CLASS, seed=SEED)
    train_texts, train_labels = get_texts_labels(train_ds)
    val_texts, val_labels = get_texts_labels(val_full)

    train_texts = preprocess_texts(train_texts)
    val_texts = preprocess_texts(val_texts)

    vectorizer, classifier = build_pipeline()
    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)

    classifier.fit(x_train, train_labels)
    val_pred = classifier.predict(x_val).tolist()

    metrics: Dict[str, object] = {
        "accuracy": float(accuracy_score(val_labels, val_pred)),
        "f1_macro": float(f1_score(val_labels, val_pred, average="macro")),
        "f1_weighted": float(f1_score(val_labels, val_pred, average="weighted")),
        "f1_per_class": f1_per_class(val_labels, val_pred),
        "n_train": len(train_labels),
        "n_val": len(val_labels),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / "metrics_baseline.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    save_confusion_matrix(
        val_labels, val_pred, RESULTS_DIR / "confusion_matrix_baseline.png"
    )

    print("Baseline (TF-IDF + LogisticRegression) — validation")
    print(f"  n_train={metrics['n_train']}  n_val={metrics['n_val']}")
    print(f"  accuracy    = {metrics['accuracy']:.4f}")
    print(f"  f1_macro    = {metrics['f1_macro']:.4f}")
    print(f"  f1_weighted = {metrics['f1_weighted']:.4f}")
    for name, score in metrics["f1_per_class"].items():
        print(f"  f1[{name:<8}] = {score:.4f}")
    print(f"  tersimpan {metrics_path.relative_to(REPO_ROOT)}")
    print(
        "  tersimpan "
        f"{(RESULTS_DIR / 'confusion_matrix_baseline.png').relative_to(REPO_ROOT)}"
    )
    return metrics


if __name__ == "__main__":
    run()