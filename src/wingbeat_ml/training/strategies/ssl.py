"""Dedicated SSL Training Strategies for FixMatch and FlexMatch."""

from typing import Any, Dict, Optional
import torch

from wingbeat_ml.training.ssl_losses import (
    FlexMatchLoss,
    compute_fixmatch_loss,
    evaluate_domain_performance,
    train_fixmatch_step,
    train_flexmatch_step,
)
from wingbeat_ml.training.strategies.base import TrainingStrategy


class FixMatchStrategy(TrainingStrategy):
    """
    Dedicated strategy for training with FixMatch algorithm on source (supervised)
    and target (unlabeled) dataset loaders.
    """

    required_datasets = {"train_labeled", "train_unlabeled", "val_source", "val_target"}

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        tau: float = 0.95,
        lambda_u: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.tau = tau
        self.lambda_u = lambda_u
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)

    def train_epoch(self, datasets: Dict[str, Any], *, epoch: int) -> Dict[str, float]:
        labeled_loader = datasets["train_labeled"]
        unlabeled_loader = datasets["train_unlabeled"]

        self.model.train()
        unlabeled_iter = iter(unlabeled_loader)
        total_loss, total_s, total_u, total_mask = 0.0, 0.0, 0.0, 0.0
        steps = 0

        for labeled_batch in labeled_loader:
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            # labeled_batch: (x_l, y_l)
            # unlabeled_batch: (x_u_w, x_u_s)
            x_l, y_l = labeled_batch[0].to(self.device), labeled_batch[1].to(self.device)
            x_u_w, x_u_s = unlabeled_batch[0].to(self.device), unlabeled_batch[1].to(self.device)

            step_res = train_fixmatch_step(
                self.model,
                self.optimizer,
                x_l,
                y_l,
                x_u_w,
                x_u_s,
                tau=self.tau,
                lambda_u=self.lambda_u,
            )

            total_loss += step_res["total_loss"]
            total_s += step_res["loss_s"]
            total_u += step_res["loss_u"]
            total_mask += step_res["mask_ratio"]
            steps += 1

        return {
            "loss": total_loss / max(steps, 1),
            "loss_s": total_s / max(steps, 1),
            "loss_u": total_u / max(steps, 1),
            "mask_ratio": total_mask / max(steps, 1),
        }

    def validate_epoch(self, datasets: Dict[str, Any], *, epoch: int) -> Dict[str, float]:
        source_val = datasets.get("val_source")
        target_val = datasets.get("val_target")
        if source_val is not None and target_val is not None:
            return evaluate_domain_performance(self.model, source_val, target_val, self.device)
        return {}


class FlexMatchStrategy(TrainingStrategy):
    """
    Dedicated strategy for training with FlexMatch (Curriculum Pseudo Labeling) algorithm.
    """

    required_datasets = {"train_labeled", "train_unlabeled", "val_source", "val_target"}

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        num_classes: int,
        config: Dict[str, Any],
        tau: float = 0.95,
        lambda_u: float = 1.0,
        mapping: str = "convex",
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)
        self.flex_loss = FlexMatchLoss(
            num_classes=num_classes,
            tau=tau,
            lambda_u=lambda_u,
            mapping=mapping,
            device=self.device,
        )

    def train_epoch(self, datasets: Dict[str, Any], *, epoch: int) -> Dict[str, float]:
        labeled_loader = datasets["train_labeled"]
        unlabeled_loader = datasets["train_unlabeled"]

        self.model.train()
        unlabeled_iter = iter(unlabeled_loader)
        total_loss, total_s, total_u, total_mask = 0.0, 0.0, 0.0, 0.0
        steps = 0

        for labeled_batch in labeled_loader:
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            x_l, y_l = labeled_batch[0].to(self.device), labeled_batch[1].to(self.device)
            x_u_w, x_u_s = unlabeled_batch[0].to(self.device), unlabeled_batch[1].to(self.device)

            step_res = train_flexmatch_step(
                self.model,
                self.flex_loss,
                self.optimizer,
                x_l,
                y_l,
                x_u_w,
                x_u_s,
            )

            total_loss += step_res["total_loss"]
            total_s += step_res["loss_s"]
            total_u += step_res["loss_u"]
            total_mask += step_res["mask_ratio"]
            steps += 1

        return {
            "loss": total_loss / max(steps, 1),
            "loss_s": total_s / max(steps, 1),
            "loss_u": total_u / max(steps, 1),
            "mask_ratio": total_mask / max(steps, 1),
        }

    def validate_epoch(self, datasets: Dict[str, Any], *, epoch: int) -> Dict[str, float]:
        source_val = datasets.get("val_source")
        target_val = datasets.get("val_target")
        if source_val is not None and target_val is not None:
            return evaluate_domain_performance(self.model, source_val, target_val, self.device)
        return {}


__all__ = ["FixMatchStrategy", "FlexMatchStrategy"]
