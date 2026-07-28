"""Semi-Supervised Learning (SSL) pipeline for FixMatch and FlexMatch."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

from wingbeat_ml.training.ssl_losses import evaluate_domain_performance
from wingbeat_ml.training.strategies.ssl import FixMatchStrategy, FlexMatchStrategy


class SimpleCNN(nn.Module):
    """Default PyTorch model for mosquito wingbeat SSL classification."""

    def __init__(self, in_channels: int = 1, num_classes: int = 11):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 3 and x.shape[1] != 1 and x.shape[2] == 1:
            x = x.transpose(1, 2)
        return self.classifier(self.features(x))


def run_ssl_pipeline(
    method: str = "fixmatch",
    model: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    datasets: Optional[Dict[str, Any]] = None,
    num_classes: int = 11,
    epochs: int = 2,
    tau: float = 0.95,
    lambda_u: float = 1.0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run semi-supervised training pipeline (FixMatch or FlexMatch) across
    supervised domain (labeled) and pseudo-unsupervised domain (unlabeled) datasets.

    Evaluates performance on both source domain and target domain.
    """
    method_norm = method.lower().strip()
    if method_norm not in ("fixmatch", "flexmatch"):
        raise ValueError(f"Unsupported SSL method {method!r}. Expected 'fixmatch' or 'flexmatch'.")

    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    if model is None:
        model = SimpleCNN(in_channels=1, num_classes=num_classes)
    model.to(device)

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Dummy datasets if none supplied (for quick test runs / standalone evaluation)
    if datasets is None:
        batch_l = 8
        batch_u = 16
        seq_len = 2400
        
        # Synthetic source labeled dataset
        x_l = torch.randn(batch_l * 4, seq_len)
        y_l = torch.randint(0, num_classes, (batch_l * 4,))
        dataset_l = torch.utils.data.TensorDataset(x_l, y_l)
        loader_l = torch.utils.data.DataLoader(dataset_l, batch_size=batch_l, shuffle=True)

        # Synthetic target unlabeled dataset (weak + strong views)
        x_u_w = torch.randn(batch_u * 4, seq_len)
        x_u_s = torch.randn(batch_u * 4, seq_len)
        dataset_u = torch.utils.data.TensorDataset(x_u_w, x_u_s)
        loader_u = torch.utils.data.DataLoader(dataset_u, batch_size=batch_u, shuffle=True)

        # Validation loaders
        val_source_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.randn(16, seq_len), torch.randint(0, num_classes, (16,))),
            batch_size=8,
        )
        val_target_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.randn(16, seq_len), torch.randint(0, num_classes, (16,))),
            batch_size=8,
        )

        datasets = {
            "train_labeled": loader_l,
            "train_unlabeled": loader_u,
            "val_source": val_source_loader,
            "val_target": val_target_loader,
        }

    # Instantiate appropriate SSL Strategy
    config = {"epochs": epochs, "tau": tau, "lambda_u": lambda_u}
    if method_norm == "fixmatch":
        strategy = FixMatchStrategy(
            model=model,
            optimizer=optimizer,
            config=config,
            tau=tau,
            lambda_u=lambda_u,
            device=device,
        )
    else:
        strategy = FlexMatchStrategy(
            model=model,
            optimizer=optimizer,
            num_classes=num_classes,
            config=config,
            tau=tau,
            lambda_u=lambda_u,
            device=device,
        )

    history = []
    if verbose:
        print(f"=== Starting {method_norm.upper()} SSL Pipeline Training ({epochs} Epochs) ===")

    for epoch in range(epochs):
        t0 = time.time()
        train_metrics = strategy.train_epoch(datasets, epoch=epoch)
        val_metrics = strategy.validate_epoch(datasets, epoch=epoch)
        duration = time.time() - t0

        log_entry = {
            "epoch": epoch + 1,
            "duration_sec": round(duration, 3),
            **{f"train_{k}": round(v, 4) for k, v in train_metrics.items()},
            **{f"val_{k}": round(v, 4) for k, v in val_metrics.items()},
        }
        history.append(log_entry)

        if verbose:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {log_entry['train_loss']:.4f} (s: {log_entry['train_loss_s']:.4f}, u: {log_entry['train_loss_u']:.4f}) | "
                f"Mask Ratio: {log_entry['train_mask_ratio']:.4f} | "
                f"Src Acc: {log_entry.get('val_source_accuracy', 0.0):.4f} | "
                f"Tgt Acc: {log_entry.get('val_target_accuracy', 0.0):.4f}"
            )

    final_eval = evaluate_domain_performance(
        model,
        datasets["val_source"],
        datasets["val_target"],
        device,
    )

    return {
        "status": "success",
        "method": method_norm,
        "epochs": epochs,
        "history": history,
        "final_evaluation": final_eval,
    }


def train_fixmatch(**kwargs) -> Dict[str, Any]:
    """Helper entry point to run FixMatch SSL pipeline."""
    return run_ssl_pipeline(method="fixmatch", **kwargs)


def train_flexmatch(**kwargs) -> Dict[str, Any]:
    """Helper entry point to run FlexMatch SSL pipeline."""
    return run_ssl_pipeline(method="flexmatch", **kwargs)


def main(args=None):
    parser = argparse.ArgumentParser(description="Run SSL FixMatch/FlexMatch training pipeline")
    parser.add_argument("--method", type=str, default="fixmatch", choices=["fixmatch", "flexmatch"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--lambda_u", type=float, default=1.0)

    parsed, _ = parser.parse_known_args(args)
    res = run_ssl_pipeline(
        method=parsed.method,
        epochs=parsed.epochs,
        tau=parsed.tau,
        lambda_u=parsed.lambda_u,
    )
    print(f"SSL Run complete. Final target domain accuracy: {res['final_evaluation']['target_accuracy']:.4f}")


if __name__ == "__main__":
    main()
