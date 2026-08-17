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


def normalize_legacy_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize legacy configuration dictionaries into canonical ExperimentConfig structure."""
    from wingbeat_ml.config.schema import DEFAULT_CLASSES

    normalized = copy.deepcopy(config_dict)

    # 1. Training mode -> training
    if "training_mode" in normalized:
        mode = normalized.pop("training_mode")
        if "training" not in normalized:
            if mode in ("pretrain", "linear_probe", "fine_tune"):
                normalized["training"] = {"paradigm": "supervised", "procedure": mode}
            elif mode == "fixmatch":
                normalized["training"] = {"paradigm": "semi_supervised", "method": "fixmatch"}
            elif mode == "flexmatch":
                normalized["training"] = {"paradigm": "semi_supervised", "method": "flexmatch"}
            else:
                raise ValueError(f"Invalid training mode '{mode}'")

    # 2. Audio segment_length vs num_samples
    audio_cfg = normalized.get("audio", {})
    if isinstance(audio_cfg, dict):
        sr = audio_cfg.get("sample_rate", 8000)
        dur = audio_cfg.get("duration", 0.3)
        num_samples = round(sr * dur)
        if "segment_length" in audio_cfg:
            seg_len = audio_cfg["segment_length"]
            if not isinstance(seg_len, int) or seg_len <= 0:
                raise ValueError(f"Invalid segment_length: must be a positive integer, got {seg_len}")
            if seg_len != num_samples:
                raise ValueError(
                    f"Conflicting segment_length ({seg_len}) vs calculated num_samples ({num_samples})"
                )
            audio_cfg.pop("segment_length", None)
            audio_cfg.pop("num_samples", None)
        elif "num_samples" in audio_cfg:
            audio_cfg.pop("num_samples", None)
    if "segment_length" in normalized:
        top_seg_len = normalized.pop("segment_length")
        if not isinstance(top_seg_len, int) or top_seg_len <= 0:
            raise ValueError(f"Invalid segment_length: must be a positive integer, got {top_seg_len}")
        if not isinstance(audio_cfg, dict):
            audio_cfg = {}
            normalized["audio"] = audio_cfg
        sr = audio_cfg.get("sample_rate", 8000)
        dur = audio_cfg.get("duration", 0.3)
        num_samples = round(sr * dur)
        if top_seg_len != num_samples:
            raise ValueError(
                f"Conflicting segment_length ({top_seg_len}) vs calculated num_samples ({num_samples})"
            )

    # 3. Classes / num_classes / labels
    has_classes = "classes" in normalized and isinstance(normalized["classes"], list)
    has_labels = "labels" in normalized and isinstance(normalized["labels"], dict)
    has_num_classes = "num_classes" in normalized

    if has_labels:
        labels_dict = dict(normalized["labels"])
        if "Ae_aegypti_Female" in labels_dict and labels_dict["Ae_aegypti_Female"] != 0:
            raise ValueError("Invalid label index mapping")

    if has_classes:
        classes = list(normalized["classes"])
        if len(set(classes)) != len(classes):
            raise ValueError("Class names must be unique")
        if has_num_classes and normalized["num_classes"] != len(classes):
            raise ValueError(
                f"Invalid num_classes: expected {len(classes)}, got {normalized['num_classes']}"
            )
        if has_labels:
            labels_dict = dict(normalized["labels"])
            if set(labels_dict.keys()) == set(classes):
                expected_labels = {name: i for i, name in enumerate(classes)}
                if labels_dict != expected_labels:
                    raise ValueError(
                        f"Inconsistent legacy labels mapping ({labels_dict}) vs classes ordering ({classes})"
                    )
            normalized.pop("labels", None)
        normalized.pop("num_classes", None)
    elif has_labels:
        labels_dict = dict(normalized.pop("labels"))
        classes = list(labels_dict.keys())
        if has_num_classes and normalized["num_classes"] != len(classes):
            raise ValueError(
                f"Invalid num_classes: expected {len(classes)}, got {normalized['num_classes']}"
            )
        normalized["classes"] = classes
        normalized.pop("num_classes", None)
    elif has_num_classes:
        n_cls = normalized.pop("num_classes")
        m_id = normalized.get("model", {}).get("id") if isinstance(normalized.get("model"), dict) else None
        if (m_id is None or m_id == "mossong_plus") and n_cls != 11:
            raise ValueError(f"Invalid num_classes: expected 11, got {n_cls}")
        if n_cls <= len(DEFAULT_CLASSES):
            normalized["classes"] = list(DEFAULT_CLASSES[:n_cls])
        else:
            normalized["classes"] = [f"class_{i}" for i in range(n_cls)]

    # 4. Checkpoint / pretrained_weights -> resume / initialization
    if "checkpoint" in normalized:
        ckpt = normalized.pop("checkpoint")
        if ckpt and "resume" not in normalized:
            normalized["resume"] = {"checkpoint": ckpt}

    if "pretrained_weights" in normalized:
        pw = normalized.pop("pretrained_weights")
        if pw and "initialization" not in normalized:
            normalized["initialization"] = {"weights": pw}

    if "model" in normalized and isinstance(normalized["model"], dict):
        m = normalized["model"]
        if "checkpoint" in m:
            ckpt = m.pop("checkpoint")
            if ckpt and "resume" not in normalized:
                normalized["resume"] = {"checkpoint": ckpt}
        if "pretrained_weights" in m:
            pw = m.pop("pretrained_weights")
            if pw and "initialization" not in normalized:
                normalized["initialization"] = {"weights": pw}

    # 5. Seed
    if "train" in normalized and isinstance(normalized["train"], dict) and "seed" in normalized["train"]:
        seed_val = normalized["train"].get("seed")
        repro = normalized.setdefault("reproducibility", {})
        if "seed" in repro and repro["seed"] != seed_val:
            if seed_val == 48:
                normalized["train"]["seed"] = repro["seed"]
            elif repro["seed"] == 48:
                repro["seed"] = seed_val
            else:
                raise ValueError(f"Inconsistent train.seed ({seed_val}) and reproducibility.seed ({repro['seed']})")
        else:
            repro["seed"] = seed_val

    # 6. Preprocessing
    if "preprocess" in normalized:
        prep = normalized.pop("preprocess")
        dataset_cfg = normalized.setdefault("dataset", {})
        if isinstance(dataset_cfg, dict):
            dataset_cfg["preprocessing"] = prep
    if "augment" in normalized and isinstance(normalized["augment"], dict) and "preprocess" in normalized["augment"]:
        prep = normalized["augment"].pop("preprocess")
        dataset_cfg = normalized.setdefault("dataset", {})
        if isinstance(dataset_cfg, dict):
            dataset_cfg["preprocessing"] = prep

    # 6b. Dataset split_list / split_ratios sync
    if "dataset" in normalized and isinstance(normalized["dataset"], dict):
        d_cfg = normalized["dataset"]
        if "split_ratios" in d_cfg and isinstance(d_cfg["split_ratios"], dict):
            sr = d_cfg["split_ratios"]
            sl = [float(sr.get("train", 0.8)), float(sr.get("val", 0.1)), float(sr.get("test", 0.1))]
            d_cfg["split_ratios"] = {"train": sl[0], "val": sl[1], "test": sl[2]}
        elif "split_list" in d_cfg and isinstance(d_cfg["split_list"], (list, tuple)) and len(d_cfg["split_list"]) == 3:
            sl = [float(x) for x in d_cfg["split_list"]]
            d_cfg["split_ratios"] = {"train": sl[0], "val": sl[1], "test": sl[2]}
        d_cfg.pop("split_list", None)

    # 7. Experiment metadata
    if "experiment_name" in normalized:
        exp_name = normalized.pop("experiment_name")
        if exp_name:
            normalized.setdefault("experiment", {})["name"] = exp_name

    # 8. Augment / Wandb aliases
    if "augment" in normalized and "augmentation" not in normalized:
        normalized["augmentation"] = normalized.pop("augment")

    if "wandb" in normalized:
        wandb_val = normalized.pop("wandb")
        if "tracking" not in normalized:
            normalized["tracking"] = wandb_val

    # 9. Strip runtime resolved state
    resolved_keys = [k for k in normalized.keys() if k.startswith("resolved_")]
    for k in resolved_keys:
        normalized.pop(k, None)

    # 10. Strip legacy non-schema keys
    for k in ("nomos_index",):
        normalized.pop(k, None)

    return normalized


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
