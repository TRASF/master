"""Runtime and artifact preparation shared by training pipelines."""

from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
from typing import Any, Optional

from wingbeat_ml.config.runtime import (
    configure_training_runtime,
    generate_experiment_name,
    resolve_experiment_paths,
)
from wingbeat_ml.config.schema import AppConfig, validate_config
from wingbeat_ml.tracking import initialize_training_run


@dataclass(frozen=True)
class TrainingRunContext:
    """Resolved identity and paths for one training execution."""

    experiment_name: str
    save_path: str
    results_dir: str
    tracking_run: Optional[Any]


def _git_revision() -> str:
    revision = os.environ.get("GIT_SHA") or os.environ.get("WANDB_GIT_COMMIT")
    if revision:
        return revision
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _pretrain_tracking_name(config: Any, base_name: str) -> str:
    from wingbeat_ml.config.schema import validate_config

    app_cfg = validate_config(config)
    high_pass = app_cfg.augment.high_pass.p
    seed = app_cfg.reproducibility.seed
    group = app_cfg.wandb.group
    num_classes = app_cfg.num_classes

    task = group or f"{num_classes}class"
    return f"{task}_{base_name}_hpf{high_pass}_seed{seed}"


def prepare_training_run(
    config: Any,
    *,
    mode: str,
    save_path: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> TrainingRunContext:
    """Initialize tracking, artifact paths, and deterministic runtime."""
    app_cfg = validate_config(config)
    tracking_run = initialize_training_run(app_cfg)

    base_name = generate_experiment_name(app_cfg, mode=mode)
    experiment_name = (
        _pretrain_tracking_name(app_cfg, base_name)
        if tracking_run is not None and mode.casefold() == "pretrain"
        else base_name
    )
    seed = app_cfg.reproducibility.seed
    if f"seed{seed}" not in experiment_name:
        experiment_name = f"{experiment_name}_seed{seed}"
    if tracking_run is not None:
        tracking_run.name = experiment_name

    paths = resolve_experiment_paths(app_cfg, experiment_name)
    resolved_save_path = str(save_path or paths["save_path"])
    resolved_results_dir = str(results_dir or paths["results_dir"])

    console = app_cfg.logging.console
    if console != "quiet":
        print(f"Experiment Name: {experiment_name}")
        print(f"Saving weights to: {resolved_save_path}")
        print(f"Saving results to: {resolved_results_dir}")

    runtime_info = configure_training_runtime(
        app_cfg.reproducibility,
        performance=app_cfg.performance,
        logging=app_cfg.logging,
    )

    launch_seed = app_cfg.resolved_launch_seed if app_cfg.resolved_launch_seed is not None else seed
    runtime_seed = int(runtime_info["seed"])
    if launch_seed != seed or runtime_seed != seed or f"seed{seed}" not in experiment_name:
        raise RuntimeError(
            "Seed mismatch: "
            f"W&B={launch_seed}, resolved={seed}, runtime={runtime_seed}, "
            f"run_name={experiment_name!r}"
        )

    return TrainingRunContext(
        experiment_name=experiment_name,
        save_path=resolved_save_path,
        results_dir=resolved_results_dir,
        tracking_run=tracking_run,
    )


def prepare_export_runtime(config: AppConfig, *, save_path: str) -> None:
    """Configure deterministic runtime state for export operations."""
    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"Checkpoint file not found for export verification: {save_path}"
        )
    configure_training_runtime(
        config.reproducibility,
        performance=config.performance,
        logging=config.logging,
    )


__all__ = [
    "TrainingRunContext",
    "prepare_export_runtime",
    "prepare_training_run",
]
