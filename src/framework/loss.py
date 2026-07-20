"""Legacy compatibility imports; use wingbeat_ml.training.losses."""

from wingbeat_ml.training.losses import (
    LossFactory,
    SupervisedContrastiveLoss,
)

__all__ = ["LossFactory", "SupervisedContrastiveLoss"]
