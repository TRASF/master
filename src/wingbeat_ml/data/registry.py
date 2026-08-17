"""Data loader and dataset registry."""

from __future__ import annotations

from typing import Any
from wingbeat_ml.registry import Registry
from wingbeat_ml.data.dataset import SupervisedDataset

DATASET_BUILDERS = Registry[Any]("dataset_builder")
DATASET_BUILDERS.register("supervised_audio", SupervisedDataset)
DATASET_BUILDERS.register("supervised", SupervisedDataset)


def register_dataset_builder(name: str):
    return DATASET_BUILDERS.register(name)


__all__ = [
    "DATASET_BUILDERS",
    "register_dataset_builder",
]
