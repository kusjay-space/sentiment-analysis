"""TF-IDF + Logistic Regression tuning experiments.

Run as a module::

    python -m src.baseline_tuned

Trains several vectorizer/classifier configurations on a stratified train
subsample and evaluates them on the full validation split (test is never
touched). Writes the best config's metrics + confusion matrix under
``results/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.baseline import f1_per_class, preprocess_texts, save_confusion_matrix
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

# Each config: vectorizer kwargs + classifier kwargs.
CONFIGS: Dict[str, Dict[str, Dict[str, object]]] = {
    "A": {
        "vec": dict(
            lowercase=True,
            max_features=100_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        ),
        "clf": dict(
            class_weight="balanced", C=1.0, max_iter=2000, random_state=SEED
        ),
    },
    "B": {
        "vec": dict(
            lowercase=True,
            max_features=100_000,
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=2,
        ),
        "clf": dict(
            class_weight="balanced", C=1.0, max_iter=2000, random_state=SEED
        ),
    },
    "C": {
        "vec": dict(
            lowercase=True,
            max_features=100_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        ),
        "clf": dict(
            class_weight="balanced", C=3.0, max_iter=2000, random_state=SEED
        ),
    },
    "D": {
        "vec": dict(
            lowercase=True,
            max_features=100_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        ),
        "clf": dict(
            class_weight="balanced",
            C=2.0,
            max_iter=2000,
            solver="saga",
            random_state=SEED,
        ),
    },
}


def run() -> Dict[str, object]:
    """Train every config, evaluate on validation, persist best artifacts."""
    dataset = load_dataset_local()
    train_ds = stratified_subsample(
        apply_mapping(dataset["train"]), N_PER_CLASS, seed=SEED
    )
    val_ds = apply_mapping(dataset["validation"])

    train_texts, train_labels = get_texts_labels(train_ds)
    val_texts, val_labels = get_texts_labels(val_ds)
    train_texts = preprocess_texts(train_texts)
    val_texts = preprocess_texts(val_texts)

    results: Dict[str, Dict[str, object]] = {}
    for name, cfg in CONFIGS.items():
        print(f"[{name}] fitting ...", flush=True)
        vectorizer = TfidfVectorizer(**cfg["vec"])
        x_train = vectorizer.fit_transform(train_texts)
        x_val = vectorizer.transform(val_texts)
        classifier = LogisticRegression(**cfg["clf"])
        classifier.fit(x_train, train_labels)
        val_pred = classifier.predict(x_val).tolist()

        results[name] = {
            "accuracy": float(accuracy_score(val_labels, val_pred)),
            "f1_macro": float(f1_score(val_labels, val_pred, average="macro")),
            "f1_weighted": float(f1_score(val_labels, val_pred, average="weighted")),
            "f1_per_class": f1_per_class(val_labels, val_pred),
            "val_pred": val_pred,
        }

    class_names = [ID2LABEL[i] for i in ID2LABEL]
    print("\n=== Config comparison (validation) ===")
    header = (
        f"{'config':<7}{'accuracy':>10}{'f1_macro':>10}"
        + "".join(f"{'f1_' + c.capitalize():>12}" for c in class_names)
    )
    print(header)
    for name in CONFIGS:
        r = results[name]
        row = f"{name:<7}{r['accuracy']:>10.4f}{r['f1_macro']:>10.4f}"
        row += "".join(
            f"{r['f1_per_class'][c]:>12.4f}" for c in class_names
        )
        print(row)

    best_name = max(results, key=lambda n: results[n]["f1_macro"])
    best = results[best_name]
    print(
        f"\nBest by f1_macro: config {best_name} "
        f"(accuracy={best['accuracy']:.4f}, f1_macro={best['f1_macro']:.4f})"
    )

    metrics: Dict[str, object] = {
        "config_name": best_name,
        "accuracy": best["accuracy"],
        "f1_macro": best["f1_macro"],
        "f1_weighted": best["f1_weighted"],
        "f1_per_class": best["f1_per_class"],
        "n_train": len(train_labels),
        "n_val": len(val_labels),
        "best_params": {
            "TfidfVectorizer": {
                k: list(v) if isinstance(v, tuple) else v
                for k, v in CONFIGS[best_name]["vec"].items()
            },
            "LogisticRegression": dict(CONFIGS[best_name]["clf"]),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / "metrics_baseline_tuned.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cm_path = RESULTS_DIR / "confusion_matrix_baseline_tuned.png"
    save_confusion_matrix(val_labels, best["val_pred"], cm_path)
    print(f"wrote {metrics_path.relative_to(REPO_ROOT)}")
    print(f"wrote {cm_path.relative_to(REPO_ROOT)}")
    return metrics


if __name__ == "__main__":
    run()
