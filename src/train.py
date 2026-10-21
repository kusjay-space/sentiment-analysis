"""Fine-tune DistilBERT for 3-class sentiment classification.

Run as a module::

    python -m src.train

The environment here is CPU-only; running this module performs a real training
run, so it is intended for a GPU host (e.g. Colab T4). Importing it for
validation does not start training — ``run()`` is only called under
``__main__``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.preprocess import (
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS,
    apply_mapping,
    get_texts_labels,
    load_dataset_local,
    stratified_subsample,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model"
CHECKPOINT_DIR = MODEL_DIR / "checkpoint"
FINAL_DIR = MODEL_DIR / "final"

MODEL_NAME = "distilbert-base-uncased"
SEED = 42
MAX_LEN = 128

# 12k NEGATIVE / 6k NEUTRAL / 12k POSITIVE = 30k total.
N_PER_CLASS: Dict[int, int] = {0: 12_000, 1: 6_000, 2: 12_000}


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute accuracy and macro-F1 from Trainer predictions."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
    }


def get_tokenizer():
    """Load the DistilBERT tokenizer."""
    return DistilBertTokenizerFast.from_pretrained(MODEL_NAME)


def tokenize_dataset(dataset: Dataset, tokenizer) -> Dataset:
    """Tokenize reviews, drop raw columns, expose `labels` as the target."""
    tokenized = dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        ),
        batched=True,
        desc="Tokenizing",
    )
    # Drop raw star-rating columns so Trainer uses the mapped sentiment only.
    drop = [c for c in ("text", "label", "label_text") if c in tokenized.column_names]
    tokenized = tokenized.remove_columns(drop)
    return tokenized.rename_column("sentiment", "labels")


def build_training_args() -> TrainingArguments:
    """Assemble TrainingArguments per the project spec."""
    return TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=2,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        seed=SEED,
        data_seed=SEED,
        report_to=[],
    )


def run() -> None:
    """Load data, fine-tune DistilBERT, and save the final model."""
    if not torch.cuda.is_available():
        print(
            "WARNING: CUDA not available — training on CPU will be very slow. "
            "Prefer a GPU host (e.g. Colab T4)."
        )

    set_seed(SEED)  # Seed torch/np/python RNG before model head init.
    dataset = load_dataset_local()
    train_full = apply_mapping(dataset["train"])
    val_full = apply_mapping(dataset["validation"])

    train_ds = stratified_subsample(train_full, N_PER_CLASS, seed=SEED)
    val_ds = val_full

    tokenizer = get_tokenizer()
    train_tokenized = tokenize_dataset(train_ds, tokenizer)
    val_tokenized = tokenize_dataset(val_ds, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID
    )

    training_args = build_training_args()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_DIR))
    tokenizer.save_pretrained(str(FINAL_DIR))

    print(f"Saved model + tokenizer to {FINAL_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    run()