"""Weights & Biases integration adapter."""

from __future__ import annotations

from typing import Any, Protocol, Sequence
from wingbeat_ml.tracking.wandb import (
    initialize_training_run,
    promote_artifact,
    registry_target,
)


class TrackerProtocol(Protocol):
    """Protocol interface for experiment tracking adapters."""

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        ...

    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None:
        ...

    def log_artifact(self, artifact_path: str, artifact_type: str) -> None:
        ...

    def finish(self) -> None:
        ...


class NoOpTracker:
    """No-op tracker implementation when tracking is disabled."""

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None:
        pass

    def log_artifact(self, artifact_path: str, artifact_type: str) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbTracker:
    """W&B tracker implementation."""

    def __init__(self, run: Any = None):
        self.run = run

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.log(logs)

    def log_artifact(self, artifact_path: str, artifact_type: str) -> None:
        if self.run is not None:
            import wandb
            art = wandb.Artifact(name=artifact_path, type=artifact_type)
            art.add_file(artifact_path)
            self.run.log_artifact(art)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


__all__ = [
    "TrackerProtocol",
    "NoOpTracker",
    "WandbTracker",
    "initialize_training_run",
    "promote_artifact",
    "registry_target",
]
