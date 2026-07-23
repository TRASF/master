"""Reproducible file and source-recording dataset splitting."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from wingbeat_ml.data.manifest import source_recording_id


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


def source_recording_ids(
    paths: Sequence[str], dataset_root: str | Path
) -> np.ndarray:
    """Derive the same source IDs used by dataset manifests."""
    return np.asarray(
        [source_recording_id(path, dataset_root) for path in paths],
        dtype=object,
    )


def _split_paths(
    paths: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    split_name: str,
    seed: int,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split paths/labels, keeping optional source-recording groups intact."""
    split_paths = paths
    split_labels = labels
    if groups is not None:
        unique_groups, first_indices = np.unique(groups, return_index=True)
        group_labels = labels[first_indices]
        for group, label in zip(unique_groups, group_labels):
            if np.any(labels[groups == group] != label):
                raise ValueError(f"Source recording {group!r} has multiple labels.")
        split_paths = unique_groups
        split_labels = group_labels

    try:
        left, right, left_labels, right_labels = train_test_split(
            split_paths,
            split_labels,
            test_size=test_size,
            stratify=split_labels,
            random_state=seed,
        )
    except ValueError:
        warnings.warn(
            f"Warning: cannot stratify {split_name} split because classes "
            "have too few examples. "
            "Falling back to a seeded non-stratified split."
        )
        left, right, left_labels, right_labels = train_test_split(
            split_paths,
            split_labels,
            test_size=test_size,
            random_state=seed,
        )

    if groups is None:
        return left, right, left_labels, right_labels
    left_mask = np.isin(groups, left)
    return paths[left_mask], paths[~left_mask], labels[left_mask], labels[~left_mask]


def split_files(
    file_paths: Sequence[str],
    labels: Sequence[int],
    split: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    source_recordings: Sequence[str] | None = None,
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
        source_recordings: Optional source ID per file. IDs never cross splits.

    Returns:
        DatasetSplits with train/validation/test paths and labels.
    """
    if len(file_paths) != len(labels) or (
        source_recordings is not None and len(file_paths) != len(source_recordings)
    ):
        raise ValueError(
            "file_paths, labels and source_recordings must have matching lengths."
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
    groups_arr = (
        np.array(source_recordings, dtype=object)
        if source_recordings is not None
        else None
    )

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
        groups=groups_arr,
    )

    # Stage 2: split eval pool into validation and test
    val_ratio = split[1] / val_test_size if val_test_size > 0 else 0.5
    eval_groups = (
        groups_arr[np.isin(paths_arr, eval_paths)]
        if groups_arr is not None
        else None
    )
    val_paths, test_paths, val_labels, test_labels = _split_paths(
        eval_paths,
        eval_labels,
        test_size=1.0 - val_ratio,
        split_name="validation/test",
        seed=seed,
        groups=eval_groups,
    )

    return DatasetSplits(
        train=tuple(str(p) for p in train_paths),
        validation=tuple(str(p) for p in val_paths),
        test=tuple(str(p) for p in test_paths),
        train_labels=tuple(int(l) for l in train_labels),
        validation_labels=tuple(int(l) for l in val_labels),
        test_labels=tuple(int(l) for l in test_labels),
    )
