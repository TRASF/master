"""Runtime and artifact preparation shared by training pipelines."""

from dataclasses import dataclass

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
