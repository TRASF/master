"""Canonical configuration loader with strict Pydantic v2 merging and single-step validation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import yaml

from wingbeat_ml.config.schema import AppConfig, validate_config


@dataclass(frozen=True)
class ResolvedConfig:
    data: AppConfig
    sources: Tuple[Path, ...] = ()
    sha256: str = ""

LEGACY_MAPPINGS: Dict[str, str] = {
    "seed": "reproducibility.seed",
    "overlap": "augment.segment_overlap",
    "rms_normalization": "augment.rms_norm",
    "learning_rate": "optimizer.learning_rate",
    "epochs": "train.epochs",
    "batch_size": "train.batch_size",
    "data": "dataset",
    "augmentation": "augment",
    "train.seed": "reproducibility.seed",
    "train.epochs": "train.epochs",
    "train.batch_size": "train.batch_size",
    "reproducibility.seed": "reproducibility.seed",
    "optimizer.learning_rate": "optimizer.learning_rate",
}


KNOWN_SECTION_PREFIXES = (
    "train.",
    "reproducibility.",
    "augment.",
    "dataset.",
    "model.",
    "loss.",
    "optimizer.",
    "performance.",
    "logging.",
    "evaluation.",
    "cache.",
    "wandb.",
    "preprocess.",
    "class_weights.",
)


def expand_dotted_keys(d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
    """Expand dotted keys into nested dictionary structures."""
    if not isinstance(d, dict) or parent_key in {"labels", "values"}:
        return d
    result: Dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, dict):
            value = expand_dotted_keys(value, parent_key=key)
        if "." in key and (not parent_key and key.startswith(KNOWN_SECTION_PREFIXES) or parent_key):
            set_nested_value(result, key, value)
        else:
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
    return result


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load raw YAML file into dictionary."""
    path_str = str(path)
    if not path_str or not os.path.exists(path_str):
        raise FileNotFoundError(f"Configuration file not found: {path_str}")
    with open(path_str, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"Configuration root in file must be a mapping: {path_str}")
        data = handle_legacy_keys(data)
        return expand_dotted_keys(data)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionary override into base."""
    if not isinstance(override, dict):
        return copy.deepcopy(override)
    merged = copy.deepcopy(base) if base else {}
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def set_nested_value(d: Dict[str, Any], path: str, value: Any) -> None:
    """Set a value in a nested dictionary using a dotted path."""
    parts = path.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def has_nested_value(d: Dict[str, Any], path: str) -> bool:
    """Check if a dotted path exists in a nested dictionary."""
    parts = path.split(".")
    curr = d
    for part in parts:
        if not isinstance(curr, dict) or part not in curr:
            return False
        curr = curr[part]
    return True


def handle_legacy_keys(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize legacy top-level keys into canonical nested paths."""
    normalized = copy.deepcopy(config_dict)
    keys_to_process = list(normalized.keys())
    for key in keys_to_process:
        if key in LEGACY_MAPPINGS:
            canonical_path = LEGACY_MAPPINGS[key]
            val = normalized.pop(key)
            if has_nested_value(normalized, canonical_path):
                warnings.warn(
                    f"Legacy key '{key}' ignored in favor of canonical key '{canonical_path}'. Compatibility behavior preserved.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"Legacy key '{key}' is deprecated. Use '{canonical_path}' instead. Compatibility behavior preserved.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                set_nested_value(normalized, canonical_path, val)
    return normalized


def parse_override(override: str) -> Tuple[str, Any]:
    """Parse dotted key=value string into (key, value) tuple."""
    if "=" not in override:
        raise ValueError(f"Invalid CLI override '{override}'; must be key=value pair")
    key, raw_val = override.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid CLI override '{override}'; empty key component")
    return key, yaml.safe_load(raw_val)


def check_and_set_nested_value(d: Dict[str, Any], path: str, value: Any) -> None:
    """Validate that dotted path exists in dictionary schema and set value."""
    parts = path.split(".")
    if any(not part for part in parts):
        raise ValueError(f"Malformed override expression: empty key component in path {repr(path)}")
    curr = d
    for part in parts[:-1]:
        if not isinstance(curr, dict) or part not in curr:
            raise KeyError(f"Override path '{path}' does not exist in the configuration (missing '{part}')")
        curr = curr[part]
    if not isinstance(curr, dict) or parts[-1] not in curr:
        raise KeyError(f"Override path '{path}' does not exist in the configuration (missing '{parts[-1]}')")
    curr[parts[-1]] = value


def apply_cli_overrides(config_dict: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """Apply dotted key=value CLI overrides to configuration dictionary."""
    result = copy.deepcopy(config_dict)
    for override in overrides:
        key, val = parse_override(override)
        check_and_set_nested_value(result, key, val)
    return result


apply_overrides = apply_cli_overrides


def compute_config_sha256(config_dict: Dict[str, Any]) -> str:
    """Compute SHA256 signature of serialized config dictionary."""
    serialized = json.dumps(config_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def load_config(
    defaults_path: Optional[Union[str, Path]] = None,
    base_path: Optional[Union[str, Path]] = None,
    model_path: Optional[Union[str, Path]] = None,
    experiment_path: Optional[Union[str, Path]] = None,
    profile_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Sequence[str]] = None,
) -> AppConfig:
    """
    Load, merge, and validate configuration using Pydantic v2 AppConfig.

    Precedence order:
    1. Base defaults (defaults_path or base_path or configs/defaults.yaml)
    2. Model configuration (model_path)
    3. Experiment configuration (experiment_path)
    4. Profile configuration (profile_path)
    5. CLI overrides
    """
    raw_dict: Dict[str, Any] = {}

    # Step 1: Base defaults
    primary_defaults = defaults_path or base_path
    if primary_defaults:
        if os.path.exists(str(primary_defaults)):
            raw_dict = deep_merge(raw_dict, handle_legacy_keys(load_yaml(primary_defaults)))
    else:
        fallback_defaults = Path("configs/defaults.yaml")
        if fallback_defaults.exists():
            raw_dict = deep_merge(raw_dict, handle_legacy_keys(load_yaml(fallback_defaults)))

    # Step 2: Model config
    if model_path and os.path.exists(str(model_path)):
        raw_dict = deep_merge(raw_dict, handle_legacy_keys(load_yaml(model_path)))

    # Step 3: Experiment config
    if experiment_path and os.path.exists(str(experiment_path)):
        raw_dict = deep_merge(raw_dict, handle_legacy_keys(load_yaml(experiment_path)))

    # Step 4: Profile config
    if profile_path and os.path.exists(str(profile_path)):
        profile_dict = handle_legacy_keys(load_yaml(profile_path))
        raw_dict = deep_merge(raw_dict, profile_dict)
        if "profile" not in raw_dict:
            raw_dict["profile"] = Path(profile_path).stem

    # Step 5: CLI overrides
    if overrides:
        schema_dict = deep_merge(AppConfig().model_dump(mode="python"), handle_legacy_keys(raw_dict))
        apply_cli_overrides(schema_dict, overrides)
        raw_dict = apply_cli_overrides(raw_dict, overrides)

    # Normalize legacy keys
    normalized_dict = handle_legacy_keys(raw_dict)

    # Validate exactly once via AppConfig
    return validate_config(normalized_dict)


def resolve_config(*args, **kwargs) -> AppConfig:
    """Convenience alias for load_config returning typed AppConfig."""
    return load_config(*args, **kwargs)


def write_resolved_config(config: AppConfig, output_path: Union[str, Path]) -> Path:
    """Write validated AppConfig instance to disk as JSON or YAML."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = config.model_dump(mode="json")
    if path.suffix in (".yaml", ".yml"):
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(content, f, default_flow_style=False, sort_keys=False)
        sha_path = Path(str(path).rsplit(".", 1)[0] + ".sha256")
        sha_path.write_text(config.sha256 + "\n", encoding="utf-8")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)
        sha_path = Path(str(path) + ".sha256")
        sha_path.write_text(config.sha256 + "\n", encoding="utf-8")
    return path
