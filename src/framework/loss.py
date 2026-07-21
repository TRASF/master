"""Legacy loss API; new code should use ``build_loss``."""

from wingbeat_ml.training.losses import (
    SupervisedContrastiveLoss,
    build_loss,
)


class LossFactory:
    """Compatibility shim for the former class-based API."""

    @staticmethod
    def get_loss(config=None):
        config = config or {}
        return build_loss(config.get("loss", config))


__all__ = ["LossFactory", "SupervisedContrastiveLoss"]
