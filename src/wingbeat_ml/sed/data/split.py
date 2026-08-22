"""Dataset manifest splitter.

Splits dataset manifests strictly by `split_group` (parent recording/session/domain)
to guarantee zero data leakage between train, validation, and test sets.
Includes checks for duplicate SHA-256 audio content hashes.
"""

from __future__ import annotations

import random
from pathlib import Path
import pandas as pd


def split_sed_manifest(
    metadata_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign split_group column and split manifests into train/val/test.

    Args:
        metadata_dir: Directory containing recordings.csv, events.csv, clips.csv.
        train_ratio: Fraction of split_groups for training.
        val_ratio: Fraction of split_groups for validation.
        test_ratio: Fraction of split_groups for testing.
        seed: Random seed for group partition.

    Returns:
        DataFrames for partitioned recordings, events, and clips.
    """
    meta_p = Path(metadata_dir)
    rec_path = meta_p / "recordings.csv"
    evt_path = meta_p / "events.csv"
    clip_path = meta_p / "clips.csv"

    if not rec_path.exists():
        raise FileNotFoundError(f"Missing recordings.csv at {rec_path}")

    rec_df = pd.read_csv(rec_path)
    evt_df = pd.read_csv(evt_path) if evt_path.exists() and evt_path.stat().st_size > 0 else pd.DataFrame()
    clip_df = pd.read_csv(clip_path) if clip_path.exists() and clip_path.stat().st_size > 0 else pd.DataFrame()

    rng = random.Random(seed)
    train_groups: set[str] = set()
    val_groups: set[str] = set()
    test_groups: set[str] = set()

    group_strata = rec_df.groupby("split_group", as_index=False).agg(
        dataset=("dataset", "first"),
        supervision_type=("supervision_type", "first"),
        size=("file_id", "size"),
    )
    for _, stratum in group_strata.groupby(["dataset", "supervision_type"]):
        groups = stratum[["split_group", "size"]].to_dict("records")
        rng.shuffle(groups)
        groups.sort(key=lambda group: group["size"], reverse=True)
        if len(groups) < 3:
            train_groups.update(str(group["split_group"]) for group in groups)
            continue

        partitions = ["train"]
        if val_ratio > 0:
            partitions.append("validation")
        if test_ratio > 0:
            partitions.append("test")
        targets = {
            "train": max(1.0, sum(group["size"] for group in groups) * train_ratio),
            "validation": max(1.0, sum(group["size"] for group in groups) * val_ratio),
            "test": max(1.0, sum(group["size"] for group in groups) * test_ratio),
        }
        assigned = {partition: 0 for partition in partitions}
        partition_groups = {partition: [] for partition in partitions}

        for index, group in enumerate(groups):
            partition = partitions[index] if index < len(partitions) else min(
                partitions,
                key=lambda name: assigned[name] / targets[name],
            )
            partition_groups[partition].append(str(group["split_group"]))
            assigned[partition] += int(group["size"])

        train_groups.update(partition_groups["train"])
        val_groups.update(partition_groups.get("validation", []))
        test_groups.update(partition_groups.get("test", []))

    def assign_split(grp: str) -> str:
        if grp in train_groups:
            return "train"
        elif grp in val_groups:
            return "validation"
        else:
            return "test"

    rec_df["split"] = rec_df["split_group"].astype(str).map(assign_split)

    # Verification 1: Split groups must be disjoint
    train_g = set(rec_df[rec_df["split"] == "train"]["split_group"])
    val_g = set(rec_df[rec_df["split"] == "validation"]["split_group"])
    test_g = set(rec_df[rec_df["split"] == "test"]["split_group"])

    assert train_g.isdisjoint(val_g), "Leakage detected: train & val groups overlap!"
    assert train_g.isdisjoint(test_g), "Leakage detected: train & test groups overlap!"
    assert val_g.isdisjoint(test_g), "Leakage detected: val & test groups overlap!"

    # Verification 2: SHA-256 hashes must not cross splits
    train_hashes = set(rec_df[rec_df["split"] == "train"]["sha256"].dropna())
    val_hashes = set(rec_df[rec_df["split"] == "validation"]["sha256"].dropna())
    test_hashes = set(rec_df[rec_df["split"] == "test"]["sha256"].dropna())

    hash_overlap_val = train_hashes.intersection(val_hashes)
    hash_overlap_test = train_hashes.intersection(test_hashes)
    if hash_overlap_val:
        print(f"Warning: {len(hash_overlap_val)} duplicate audio files between train and validation!")
    if hash_overlap_test:
        print(f"Warning: {len(hash_overlap_test)} duplicate audio files between train and test!")

    # Propagate split to events and clips
    if not evt_df.empty and "split_group" in evt_df.columns:
        evt_df["split"] = evt_df["split_group"].astype(str).map(assign_split)
    if not clip_df.empty and "split_group" in clip_df.columns:
        clip_df["split"] = clip_df["split_group"].astype(str).map(assign_split)

    rec_df.to_csv(rec_path, index=False)
    if not evt_df.empty:
        evt_df.to_csv(evt_path, index=False)
    if not clip_df.empty:
        clip_df.to_csv(clip_path, index=False)

    return rec_df, evt_df, clip_df


if __name__ == "__main__":
    split_sed_manifest("/home/miru4090s/clones/Master Thesises/MosSongPlus/metadata")
