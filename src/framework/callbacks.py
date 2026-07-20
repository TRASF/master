"""Legacy compatibility imports; use wingbeat_ml.training.callbacks."""

from wingbeat_ml.training.callbacks import (
    CallbackFactory,
    CosineAnnealing,
    EarlyStopping,
    MetricMonitor,
    ModelCheckpoint,
    ReduceLROnPlateau,
    WandbLogger,
)

__all__ = [
    "CallbackFactory",
    "CosineAnnealing",
    "EarlyStopping",
    "MetricMonitor",
    "ModelCheckpoint",
    "ReduceLROnPlateau",
    "WandbLogger",
]
