"""Data loading, label mapping, and stratified sampling utilities.

Raw Amazon Reviews Multi (English) ships 5 labels (0-4). We collapse them into
a 3-class sentiment scheme and expose helpers shared by baseline, training, and
inference scripts.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple, Union

from datasets import Dataset, DatasetDict, load_from_disk

# Repository root: this file lives in <root>/src/preprocess.py
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "amazon_reviews_multi_en"

# 3-class sentiment scheme.
ID2LABEL: Dict[int, str] = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
LABEL2ID: Dict[str, int] = {name: idx for idx, name in ID2LABEL.items()}

# Raw (0-4) -> sentiment (0-2). 0-1 negative, 2 neutral, 3-4 positive.
_RAW_TO_SENTIMENT: Dict[int, int] = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}

# Number of classes produced by the mapping.
NUM_LABELS: int = len(ID2LABEL)


def load_dataset_local(path: Union[str, Path] = DATA_DIR) -> DatasetDict:
    """Load the cached Amazon Reviews Multi (English) dataset from disk.

    Args:
        path: Directory previously written by ``datasets.save_to_disk``.

    Returns:
        DatasetDict with ``train``, ``validation`` and ``test`` splits.

    Raises:
        FileNotFoundError: If the cached dataset directory does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cached dataset not found at {path}. "
            "Run the download step or point to the correct cache directory."
        )
    return load_from_disk(str(path))


def map_label(label: int) -> int:
    """Map a raw 0-4 rating label to the 3-class sentiment scheme.

    Args:
        label: Raw label in ``[0, 4]``.

    Returns:
        Sentiment id in ``[0, 2]`` (see :data:`ID2LABEL`).

    Raises:
        ValueError: If ``label`` is outside the supported range.
    """
    try:
        return _RAW_TO_SENTIMENT[int(label)]
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Unsupported raw label: {label!r}") from exc


def apply_mapping(dataset: Dataset) -> Dataset:
    """Add a ``sentiment`` column derived from the raw ``label`` column.

    Args:
        dataset: Dataset with an integer ``label`` column.

    Returns:
        A copy of ``dataset`` with an added ``sentiment`` column.
    """
    mapped = {raw: senti for raw, senti in _RAW_TO_SENTIMENT.items()}
    return dataset.map(
        lambda example: {"sentiment": mapped[int(example["label"])]},
        desc="Mapping labels to sentiment",
    )


def stratified_subsample(
    dataset: Dataset,
    n_per_class: Union[int, Dict[int, int]],
    seed: int = 42,
) -> Dataset:
    """Sample a fixed number of examples per sentiment class.

    Args:
        dataset: Dataset that already has a ``sentiment`` column.
        n_per_class: Either a single count applied to every class, or a mapping
            ``{sentiment_id: count}``. Counts larger than the available pool
            are capped at the pool size.
        seed: RNG seed for reproducible sampling.

    Returns:
        A subset of ``dataset`` with at most ``n_per_class`` rows per class,
        returned in ascending original row order.
    """
    if isinstance(n_per_class, int):
        counts = {sentiment: n_per_class for sentiment in ID2LABEL}
    else:
        counts = dict(n_per_class)

    sentiments: List[int] = dataset["sentiment"]

    indices_by_class: Dict[int, List[int]] = {sentiment: [] for sentiment in ID2LABEL}
    for idx, sentiment in enumerate(sentiments):
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
    """Extract parallel text and sentiment-label lists from a dataset.

    Args:
        dataset: Dataset with ``text`` and ``sentiment`` columns.

    Returns:
        Tuple ``(texts, labels)`` where ``labels`` uses the sentiment ids.
    """
    return list(dataset["text"]), [int(x) for x in dataset["sentiment"]]