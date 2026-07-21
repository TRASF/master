"""Legacy callback API; new code should use ``build_callbacks``."""

from wingbeat_ml.training.callbacks import (
    CosineAnnealing,
    EarlyStopping,
    MetricMonitor,
    ModelCheckpoint,
    ReduceLROnPlateau,
    WandbLogger,
    build_callbacks,
)


class CallbackFactory:
    """Compatibility shim for the former class-based API."""

    @staticmethod
    def get_callbacks(
        config,
        optimizer,
        model,
        model_save_path,
        val_x=None,
    ):
        return build_callbacks(
            config,
            optimizer,
            model,
            model_save_path,
            val_x=val_x,
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
