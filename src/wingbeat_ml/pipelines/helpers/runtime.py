"""Runtime and artifact preparation shared by training pipelines."""

from dataclasses import dataclass
import os
import subprocess

from wingbeat_ml.config.runtime import (
    configure_training_runtime,
    generate_experiment_name,
    resolve_experiment_paths,
)
from wingbeat_ml.tracking import initialize_training_run
from wingbeat_ml.config.schema import validate_config


@dataclass(frozen=True)
class TrainingRunContext:
    """Resolved identity and paths for one training execution."""

    experiment_name: str
    save_path: str
    results_dir: str
    tracking_run: object | None


def _git_revision():
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


def _pretrain_tracking_name(config, base_name):
    high_pass = config["augment"]["high_pass"]["p"]
    seed = config["reproducibility"]["seed"]
    task = (
        config["wandb"].get("group")
        or f"{config['num_classes']}class"
    )
    return f"{task}_{base_name}_hpf{high_pass}_seed{seed}"


def prepare_training_run(
    config,
    *,
    mode,
    save_path=None,
    results_dir=None,
):
    """Initialize tracking, artifact paths, and deterministic runtime."""
    tracking_run = initialize_training_run(config)
    try:
        validate_config(config)
    except Exception:
        if tracking_run is not None:
            finish = getattr(tracking_run, "finish", None)
            if callable(finish):
                finish(exit_code=1)
        raise
    base_name = generate_experiment_name(config, mode=mode)
    experiment_name = (
        _pretrain_tracking_name(config, base_name)
        if tracking_run is not None and mode.casefold() == "pretrain"
        else base_name
    )
    seed = int(config["reproducibility"]["seed"])
    if f"seed{seed}" not in experiment_name:
        experiment_name = f"{experiment_name}_seed{seed}"
    if tracking_run is not None:
        tracking_run.name = experiment_name

    paths = resolve_experiment_paths(config, experiment_name)
    save_path = save_path or paths["save_path"]
    results_dir = results_dir or paths["results_dir"]
    config["resolved_run"] = {
        "experiment_name": experiment_name,
        "save_path": str(save_path),
        "results_dir": str(results_dir),
    }

    console = str(config.get("logging", {}).get("console", "normal"))
    if console != "quiet":
        print(f"Experiment Name: {experiment_name}")
        print(f"Saving weights to: {save_path}")
        print(f"Saving results to: {results_dir}")
    runtime_info = configure_training_runtime(
        config["reproducibility"],
        performance=config.get("performance", {}),
        logging=config.get("logging", {}),
    )
    config["resolved_runtime"] = runtime_info

    launch_seed = int(config.get("resolved_launch_seed", seed))
    runtime_seed = int(runtime_info["seed"])
    if launch_seed != seed or runtime_seed != seed or f"seed{seed}" not in experiment_name:
        raise RuntimeError(
            "Seed mismatch: "
            f"W&B={launch_seed}, resolved={seed}, runtime={runtime_seed}, "
            f"run_name={experiment_name!r}"
        )

    resolved = {
        "seed": seed,
        "profile": config.get(
            "resolved_profile",
            config.get("profile", os.environ.get("WINGBEAT_PROFILE", "unknown")),
        ),
        "git_revision": _git_revision(),
        "image_revision": os.environ.get("WINGBEAT_IMAGE_REVISION", "unknown"),
        "cache_schema": config.get("cache", {}).get("schema_version"),
    }
    config["resolved_provenance"] = resolved
    if console != "quiet":
        print(f"Resolved runtime: {resolved}")
    if tracking_run is not None:
        import wandb
        wandb.config.update(
            {f"resolved.{key}": value for key, value in resolved.items()},
            allow_val_change=True,
        )

    return TrainingRunContext(
        experiment_name=experiment_name,
        save_path=save_path,
        results_dir=results_dir,
        tracking_run=tracking_run,
    )


def prepare_export_runtime(config):
    """Initialize deterministic export runtime and return its seed."""
    configure_training_runtime(config["reproducibility"])
    return config["reproducibility"]["seed"]


__all__ = [
    "TrainingRunContext",
    "prepare_export_runtime",
    "prepare_training_run",
]
