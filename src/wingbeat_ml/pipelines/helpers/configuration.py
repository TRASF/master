"""Configuration selection for canonical pipeline entrypoints."""

from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

from wingbeat_ml.config import (
    AppConfig,
    load_config,
    write_resolved_config,
)
from wingbeat_ml.config.loader import load_yaml


def find_project_root(start: Optional[Union[str, Path]] = None) -> Path:
    """Find a checkout containing the canonical configuration layers."""
    starting_path = Path(start or Path.cwd()).resolve()
    source_root = Path(__file__).resolve().parents[4]

    for candidate in (starting_path, *starting_path.parents, source_root):
        if (
            (candidate / "configs" / "base.yaml").is_file()
            and (
                candidate
                / "configs"
                / "models"
                / "mossong_plus.yaml"
            ).is_file()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not find configs/base.yaml. Run from a MosSongPlus "
        "checkout or provide explicit configuration paths."
    )


def _absolute_from_project(value: Union[str, Path], project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _override(path: str, value: Any) -> str:
    encoded = yaml.safe_dump(value, default_flow_style=True).strip()
    if encoded.endswith("\n..."):
        encoded = encoded[:-4]
    return f"{path}={encoded}"


def load_pipeline_configuration(
    defaults_path: Union[str, Path] = "configs/defaults.yaml",
    model_config_path: Union[str, Path] = "configs/model.yaml",
) -> Tuple[AppConfig, Dict[str, Any]]:
    """Load the canonical configuration pair used by entrypoints."""
    config = load_config(
        defaults_path=defaults_path,
        model_path=model_config_path if os.path.exists(str(model_config_path)) else None,
    )
    validate_pipeline_configuration(config)
    model_dict = load_yaml(model_config_path) if os.path.exists(str(model_config_path)) else config.model.model_dump()
    return config, model_dict


def validate_pipeline_configuration(config: Any) -> AppConfig:
    """Require operational settings before expensive runtime setup."""
    required = (
        ("dataset", "train_dir"),
        ("dataset", "val_dir"),
        ("dataset", "test_dir"),
        ("train", "epochs"),
        ("train", "batch_size"),
        ("reproducibility", "seed"),
        ("runtime", "experiments_dir"),
        ("runtime", "root"),
        ("wandb", "enabled"),
    )
    if isinstance(config, dict):
        missing = [
            ".".join(path)
            for path in required
            if path[0] not in config or not isinstance(config[path[0]], dict) or path[1] not in config[path[0]]
        ]
        if missing:
            raise ValueError(
                "Missing required pipeline configuration: "
                + ", ".join(missing)
            )
    from wingbeat_ml.config.schema import validate_config

    return validate_config(config)


def prepare_default_pilot(
    project_root: Optional[Union[str, Path]] = None,
    runtime_root: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path, Path]:
    """Resolve the configured pilot profile for a bare invocation."""
    root = find_project_root(project_root)
    base_path = root / "configs" / "base.yaml"
    model_path = root / "configs" / "models" / "mossong_plus.yaml"
    experiment_path = root / "configs" / "experiments" / "pretrain.yaml"
    profile_path = root / "configs" / "profiles" / "pilot.yaml"

    policy = load_config(
        base_path=str(base_path),
        model_path=str(model_path),
        experiment_path=str(experiment_path),
        profile_path=str(profile_path),
    )

    dataset_value = (
        os.environ.get("WINGBEAT_DATASET_DIR")
        or policy.dataset.train_dir
    )
    dataset = _absolute_from_project(dataset_value, root)
    if not dataset.is_dir():
        raise FileNotFoundError(
            f"Pilot dataset not found: {dataset}. Set "
            "WINGBEAT_DATASET_DIR or update the pilot profile."
        )

    runtime_value = (
        runtime_root
        or os.environ.get("WINGBEAT_RUNTIME_ROOT")
        or policy.runtime.root
    )
    runtime = _absolute_from_project(runtime_value, root)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    execution_root = runtime / "pilots" / timestamp
    config_dir = execution_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    resolved = load_config(
        base_path=str(base_path),
        model_path=str(model_path),
        experiment_path=str(experiment_path),
        profile_path=str(profile_path),
        overrides=[
            _override("dataset.train_dir", str(dataset)),
            _override("runtime.root", str(execution_root)),
            _override(
                "cache.root",
                str(runtime / "dataset" / ".tf_cache"),
            ),
        ],
    )
    resolved_path = config_dir / "resolved.yaml"
    write_resolved_config(resolved, str(resolved_path))

    print(f"Zero-argument pilot config: {resolved_path}")
    print(f"Pilot run directory: {execution_root}")
    return resolved_path, model_path, execution_root


__all__ = [
    "find_project_root",
    "load_pipeline_configuration",
    "prepare_default_pilot",
    "validate_pipeline_configuration",
]
