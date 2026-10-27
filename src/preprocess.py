"""Data loading, label mapping, and stratified sampling.

Amazon Reviews Multi (EN) ships 5 labels (0-4); we collapse them into 3 classes.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple, Union

from datasets import Dataset, DatasetDict, load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "amazon_reviews_multi_en"

ID2LABEL: Dict[int, str] = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
LABEL2ID: Dict[str, int] = {name: idx for idx, name in ID2LABEL.items()}
NUM_LABELS: int = len(ID2LABEL)

# The dataset stores the star rating zero-indexed: 0 = 1 star ... 4 = 5 stars.
# Stars 1-2 -> NEGATIVE, star 3 -> NEUTRAL, stars 4-5 -> POSITIVE.
_RAW_TO_SENTIMENT: Dict[int, int] = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}


def load_dataset_local(path: Union[str, Path] = DATA_DIR) -> DatasetDict:
    """Load the cached dataset written earlier by ``save_to_disk``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cached dataset not found at {path}. "
            "Run the download step or point to the correct cache directory."
        )
    return load_from_disk(str(path))


def map_label(label: int) -> int:
    """Map a raw 0-4 rating label to a sentiment id (0-2)."""
    try:
        return _RAW_TO_SENTIMENT[int(label)]
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Unsupported raw label: {label!r}") from exc


def apply_mapping(dataset: Dataset) -> Dataset:
    """Add a ``sentiment`` column derived from the raw ``label`` column."""
    return dataset.map(
        lambda example: {"sentiment": _RAW_TO_SENTIMENT[int(example["label"])]},
        desc="Mapping labels to sentiment",
    )


def stratified_subsample(
    dataset: Dataset,
    n_per_class: Union[int, Dict[int, int]],
    seed: int = 42,
) -> Dataset:
    """Keep at most ``n_per_class`` rows per class, in original row order.

    ``n_per_class`` is either one count for every class, or ``{class: count}``.
    Counts above the available pool are capped at the pool size.
    """
    if isinstance(n_per_class, int):
        counts = {sentiment: n_per_class for sentiment in ID2LABEL}
    else:
        counts = dict(n_per_class)

    indices_by_class: Dict[int, List[int]] = {sentiment: [] for sentiment in ID2LABEL}
    for idx, sentiment in enumerate(dataset["sentiment"]):
        indices_by_class[int(sentiment)].append(idx)

    rng = random.Random(seed)
    selected: List[int] = []
    for sentiment, count in counts.items():
        pool = indices_by_class.get(int(sentiment), [])
        take = min(int(count), len(pool))
        if take:
            selected.extend(rng.sample(pool, take))

    selected.sort()
    return dataset.select(selected)


def get_texts_labels(dataset: Dataset) -> Tuple[List[str], List[int]]:
    """Return parallel ``(texts, sentiment_ids)`` lists."""
    return list(dataset["text"]), [int(x) for x in dataset["sentiment"]]
