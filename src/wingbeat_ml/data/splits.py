"""Reproducible source-file splitting for the wingbeat dataset.

Preserves the exact behaviour of SupervisedDataset._split_paths():
  - sklearn.model_selection.train_test_split with random_state=seed
  - stratify=labels (falls back to non-stratified if class count too small)
  - File-level splitting (not recording-level); source-recording leakage is
    a known issue (see docs/known-issues.md).

The split unit is individual files (str paths), matching the legacy
implementation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplits:
    """Container for the three dataset splits.

    Each attribute is a tuple of absolute file-path strings (or Path objects)
    with a corresponding label tuple.
    """

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    train_labels: tuple[int, ...]
    validation_labels: tuple[int, ...]
    test_labels: tuple[int, ...]


def _split_paths(
    paths: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    split_name: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split paths/labels into two parts.

    Attempts stratified split; falls back to seeded non-stratified split when
    any class has fewer than 2 samples (matching legacy behaviour exactly).
    """
    try:
        return train_test_split(
            paths,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=seed,
        )
    except ValueError:
        warnings.warn(
            f"Warning: cannot stratify {split_name} split because classes "
            "have too few examples. "
            "Falling back to a seeded non-stratified split."
        )
        return train_test_split(
            paths,
            labels,
            test_size=test_size,
            random_state=seed,
        )


def split_files(
    file_paths: Sequence[str],
    labels: Sequence[int],
    split: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> DatasetSplits:
    """Split file paths into train / validation / test sets.

    Mirrors SupervisedDataset.build() two-stage splitting:
      1. Reserve val_test_size = split[1] + split[2] from total
      2. Then split the reserved portion into validation / test

    Args:
        file_paths: Iterable of file path strings.
        labels: Corresponding integer class indices (same length).
        split: (train_ratio, val_ratio, test_ratio).  Must sum to 1.0.
        seed: Random seed for sklearn train_test_split (random_state=seed).

    Returns:
        DatasetSplits with train/validation/test paths and labels.

    KNOWN ISSUE: splitting is at the file level, not the source-recording level.
    A single source recording may contribute files to both train and test
    (source-leakage).  See docs/known-issues.md.
    """
    if len(file_paths) != len(labels):
        raise ValueError(
            "file_paths and labels must contain the same number of values."
        )
    if len(split) != 3:
        raise ValueError("split must contain train, validation and test ratios.")

    ratios = tuple(float(value) for value in split)
    if not all(np.isfinite(value) for value in ratios):
        raise ValueError("Split ratios must be finite numbers.")
    if any(value < 0.0 for value in ratios):
        raise ValueError("Split ratios cannot be negative.")

    split = ratios
    paths_arr = np.array(file_paths, dtype=object)
    labels_arr = np.array(labels, dtype=np.int32)

    if len(paths_arr) == 0:
        empty: tuple[str, ...] = ()
        empty_labels: tuple[int, ...] = ()
        return DatasetSplits(empty, empty, empty, empty_labels, empty_labels, empty_labels)

    total = float(split[0]) + float(split[1]) + float(split[2])
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    val_test_size = split[1] + split[2]

    # Stage 1: separate train from eval pool
    train_paths, eval_paths, train_labels, eval_labels = _split_paths(
        paths_arr,
        labels_arr,
        test_size=val_test_size,
        split_name="train/eval",
        seed=seed,
    )

    # Stage 2: split eval pool into validation and test
    val_ratio = split[1] / val_test_size if val_test_size > 0 else 0.5
    val_paths, test_paths, val_labels, test_labels = _split_paths(
        eval_paths,
        eval_labels,
        test_size=1.0 - val_ratio,
        split_name="validation/test",
        seed=seed,
    )

    return DatasetSplits(
        train=tuple(str(p) for p in train_paths),
        validation=tuple(str(p) for p in val_paths),
        test=tuple(str(p) for p in test_paths),
        train_labels=tuple(int(l) for l in train_labels),
        validation_labels=tuple(int(l) for l in val_labels),
        test_labels=tuple(int(l) for l in test_labels),
    )
