"""TensorFlow SSL strategies for FixMatch and FlexMatch."""

from __future__ import annotations

from typing import Any, Dict, Optional
import tensorflow as tf

from wingbeat_ml.classification.training.strategies.base import TrainingStrategy
from wingbeat_ml.classification.training.tf_ssl_losses import (
    TFFlexMatchLoss,
    train_tf_fixmatch_step,
    train_tf_flexmatch_step,
)


class FixMatchStrategy(TrainingStrategy):
    required_datasets = {"train_labeled", "train_unlabeled"}

    def __init__(
        self,
        model: Any = None,
        optimizer: Any = None,
        loss_fn: Any = None,
        config: Any = None,
        **kwargs,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        ssl_cfg = getattr(config, "ssl", None) if config else None
        self.tau = float(getattr(ssl_cfg, "tau", 0.95)) if ssl_cfg else 0.95
        self.lambda_u = float(getattr(ssl_cfg, "lambda_u", 1.0)) if ssl_cfg else 1.0
        self.global_step = 0

    def train_step(self, x_l, y_l, x_u_w, x_u_s):
        return train_tf_fixmatch_step(
            model=self.model,
            optimizer=self.optimizer,
            x_l=x_l,
            y_l=y_l,
            x_u_w=x_u_w,
            x_u_s=x_u_s,
            tau=self.tau,
            lambda_u=self.lambda_u,
        )

    def train_epoch(self, datasets: Any = None, *, epoch: int = 0) -> Dict[str, Any]:
        ds = datasets
        steps = 0
        total_loss, total_s, total_u, total_mask = 0.0, 0.0, 0.0, 0.0

        for (x_l, y_l), (x_u_w, x_u_s) in ds:
            res = self.train_step(x_l, y_l, x_u_w, x_u_s)
            total_loss += res["total_loss"]
            total_s += res["loss_s"]
            total_u += res["loss_u"]
            total_mask += res["mask_ratio"]
            steps += 1
            self.global_step += 1

        avg_steps = max(steps, 1)
        return {
            "loss": total_loss / avg_steps,
            "loss_s": total_s / avg_steps,
            "loss_u": total_u / avg_steps,
            "mask_ratio": total_mask / avg_steps,
            "batches": steps,
            "global_step": self.global_step,
        }


class FlexMatchStrategy(TrainingStrategy):
    required_datasets = {"train_labeled", "train_unlabeled"}

    def __init__(
        self,
        model: Any = None,
        optimizer: Any = None,
        loss_fn: Any = None,
        config: Any = None,
        **kwargs,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        ssl_cfg = getattr(config, "ssl", None) if config else None
        num_classes = getattr(config, "num_classes", 11) if config else 11
        tau = float(getattr(ssl_cfg, "tau", 0.95)) if ssl_cfg else 0.95
        lambda_u = float(getattr(ssl_cfg, "lambda_u", 1.0)) if ssl_cfg else 1.0
        mapping = str(getattr(ssl_cfg, "mapping", "convex")) if ssl_cfg else "convex"

        self.flex_layer = TFFlexMatchLoss(
            num_classes=num_classes,
            tau=tau,
            lambda_u=lambda_u,
            mapping=mapping,
        )
        self.global_step = 0

    def train_step(self, x_l, y_l, x_u_w, x_u_s):
        return train_tf_flexmatch_step(
            model=self.model,
            flexmatch_layer=self.flex_layer,
            optimizer=self.optimizer,
            x_l=x_l,
            y_l=y_l,
            x_u_w=x_u_w,
            x_u_s=x_u_s,
        )

    def train_epoch(self, datasets: Any = None, *, epoch: int = 0) -> Dict[str, Any]:
        ds = datasets
        steps = 0
        total_loss, total_s, total_u, total_mask = 0.0, 0.0, 0.0, 0.0

        for (x_l, y_l), (x_u_w, x_u_s) in ds:
            res = self.train_step(x_l, y_l, x_u_w, x_u_s)
            total_loss += res["total_loss"]
            total_s += res["loss_s"]
            total_u += res["loss_u"]
            total_mask += res["mask_ratio"]
            steps += 1
            self.global_step += 1

        avg_steps = max(steps, 1)
        return {
            "loss": total_loss / avg_steps,
            "loss_s": total_s / avg_steps,
            "loss_u": total_u / avg_steps,
            "mask_ratio": total_mask / avg_steps,
            "batches": steps,
            "global_step": self.global_step,
        }


__all__ = ["FixMatchStrategy", "FlexMatchStrategy"]