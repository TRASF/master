"""Reusable Wingbeat ML training components."""

from wingbeat_ml.training.callbacks import build_callbacks
from wingbeat_ml.training.losses import SupervisedContrastiveLoss, build_loss
from wingbeat_ml.training.optimizers import build_optimizer
from wingbeat_ml.training.tf_ssl_losses import (
    TFFlexMatchLoss,
    compute_classification_metrics,
    evaluate_tf_domain_performance,
    tf_compute_fixmatch_loss,
    train_tf_fixmatch_step,
    train_tf_flexmatch_step,
)
from wingbeat_ml.training.strategies.registry import build_strategy
from wingbeat_ml.training.strategies.ssl_tf import FixMatchStrategy, FlexMatchStrategy
from wingbeat_ml.training.strategies.supervised import SupervisedStrategy
from wingbeat_ml.training.trainer import Train, Trainer

__all__ = [
    "TFFlexMatchLoss",
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
    "compute_classification_metrics",
    "evaluate_tf_domain_performance",
    "tf_compute_fixmatch_loss",
    "train_tf_fixmatch_step",
    "train_tf_flexmatch_step",
]
