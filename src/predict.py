"""Interactive sentiment prediction — test your own text.

Usage:
    python -m src.predict "This app is amazing!"
    python -m src.predict          # interactive mode (type, Enter, 'quit' to exit)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model" / "final"

ID2LABEL = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
_model = None
_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load():
    """Lazy-load model + tokenizer once."""
    global _model, _tokenizer
    if _model is None:
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_DIR}. "
                "Download from Releases: gh release download v1.0 -p model_final.zip "
                "then: unzip model_final.zip -d model/"
            )
        _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        _model = DistilBertForSequenceClassification.from_pretrained(
            str(MODEL_DIR)
        ).to(_device)
        _model.eval()


def predict(text: str) -> dict:
    """Predict sentiment for a single text. Returns label + per-class probabilities."""
    _load()
    encoded = _tokenizer(
        text, truncation=True, max_length=128, padding="max_length", return_tensors="pt"
    ).to(_device)
    with torch.no_grad():
        logits = _model(**encoded).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(probs.argmax())
    return {
        "text": text,
        "label": ID2LABEL[pred_id],
        "confidence": round(float(probs[pred_id]), 3),
        "probabilities": {
            ID2LABEL[i]: round(float(probs[i]), 3) for i in range(3)
        },
    }


def _pretty(result: dict) -> str:
    p = result["probabilities"]
    return (
        f"  → {result['label']} (conf {result['confidence']})\n"
        f"    NEG {p['NEGATIVE']:.3f} | NEU {p['NEUTRAL']:.3f} | POS {p['POSITIVE']:.3f}"
    )


def main():
    if len(sys.argv) > 1:
        # One-shot: text from command-line args.
        print(_pretty(predict(" ".join(sys.argv[1:]))))
        return
    # Interactive loop.
    print("Sentiment test — type a sentence, Enter to predict, 'quit' to exit.")
    while True:
        try:
            text = input("\ntext> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit", "q"}:
            break
        if not text:
            continue
        print(_pretty(predict(text)))


if __name__ == "__main__":
    main()
