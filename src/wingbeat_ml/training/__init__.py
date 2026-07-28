"""Reusable Wingbeat ML training components."""

from wingbeat_ml.training.callbacks import build_callbacks
from wingbeat_ml.training.losses import SupervisedContrastiveLoss, build_loss
from wingbeat_ml.training.optimizers import build_optimizer
from wingbeat_ml.training.ssl_losses import (
    FlexMatchLoss,
    compute_fixmatch_loss,
    evaluate_domain_performance,
    train_fixmatch_step,
    train_flexmatch_step,
)
from wingbeat_ml.training.strategies.registry import build_strategy
from wingbeat_ml.training.strategies.ssl import FixMatchStrategy, FlexMatchStrategy
from wingbeat_ml.training.strategies.supervised import SupervisedStrategy
from wingbeat_ml.training.trainer import Train, Trainer

__all__ = [
    "FlexMatchLoss",
    "FixMatchStrategy",
    "FlexMatchStrategy",
    "SupervisedContrastiveLoss",
    "SupervisedStrategy",
    "Train",
    "Trainer",
    "build_callbacks",
    "build_loss",
    "build_optimizer",
    "build_strategy",
    "compute_fixmatch_loss",
    "evaluate_domain_performance",
    "train_fixmatch_step",
    "train_flexmatch_step",
]
