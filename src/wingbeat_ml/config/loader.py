import os
import yaml
import warnings
import json
import hashlib
import copy
from pathlib import Path
from dataclasses import dataclass
from .schema import validate_config

# Canonical mappings and precedence
LEGACY_MAPPINGS = {
    "seed": "reproducibility.seed",
    "overlap": "augment.segment_overlap",
    "rms_normalization": "augment.rms_norm",
    "learning_rate": "optimizer.learning_rate",
    "epochs": "train.epochs",
    "batch_size": "train.batch_size",
    "train.seed": "reproducibility.seed",
    "train.epochs": "train.epochs",
    "train.batch_size": "train.batch_size",
    "reproducibility.seed": "reproducibility.seed",
    "optimizer.learning_rate": "optimizer.learning_rate",
}

# Fill other dotted keys mapping to themselves
for key in [
    "wandb.tags", "wandb.group", "wandb.notes", "wandb.enabled", "wandb.project", "wandb.job_type",
    "wandb.log_weights_freq", "wandb.aggregate_plot_freq", "wandb.log_prediction_audio", "wandb.prediction_table_max_rows",
    "dataset.indoor", "dataset.mosLab", "dataset.outdoor", "dataset.val_dir", "dataset.test_dir", "dataset.train_dir",
    "augment.mixup.p", "augment.mixup.alpha", "augment.high_pass.p", "augment.time_shift.p", "augment.pitch_shift.p",
    "augment.random_gain.p", "augment.random_gain.gain_db", "augment.noise_overlay.p", "augment.noise_overlay.snr_db",
    "augment.noise_overlay.post_gain_db", "augment.noise_overlay.envelope_gain", "augment.gaussian_noise.p", "augment.gaussian_noise.snr_db"
]:
    LEGACY_MAPPINGS[key] = key


@dataclass(frozen=True)
class ResolvedConfig:
    data: dict[str, object]
    sources: tuple[Path, ...]
    sha256: str


def load_yaml(path: str) -> dict:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"Configuration root in file must be a mapping: {path}")
        return data


def deep_merge(base: dict, override: dict) -> dict:
    if not isinstance(override, dict):
        return copy.deepcopy(override)
    merged = copy.deepcopy(base) if base else {}
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def check_and_set_nested_value(d: dict, path: str, value: any) -> None:
    parts = path.split('.')
    if any(not part for part in parts):
        raise ValueError(f"Malformed override expression: empty key component in path {repr(path)}")
        
    curr = d
    for i, part in enumerate(parts[:-1]):
        if not isinstance(curr, dict) or part not in curr:
            raise KeyError(f"Override path '{path}' does not exist in the configuration (missing '{part}')")
        curr = curr[part]
        
    if not isinstance(curr, dict) or parts[-1] not in curr:
        raise KeyError(f"Override path '{path}' does not exist in the configuration (missing '{parts[-1]}')")
    
    curr[parts[-1]] = value


def set_nested_value(d: dict, path: str, value: any) -> None:
    parts = path.split('.')
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def has_nested_value(d: dict, path: str) -> bool:
    parts = path.split('.')
    for part in parts:
        if not isinstance(d, dict) or part not in d:
            return False
        d = d[part]
    return True


def handle_legacy_keys(config: dict) -> dict:
    normalized = config.copy()
    keys_to_process = list(normalized.keys())
    for key in keys_to_process:
        if key in LEGACY_MAPPINGS:
            canonical_path = LEGACY_MAPPINGS[key]
            value = normalized[key]
            
            if has_nested_value(normalized, canonical_path):
                warnings.warn(
                    f"Legacy key '{key}' ignored in favor of canonical key '{canonical_path}'. Compatibility behavior preserved.",
                    DeprecationWarning,
                    stacklevel=2
                )
            else:
                warnings.warn(
                    f"Legacy key '{key}' is deprecated. Use '{canonical_path}' instead. Compatibility behavior preserved.",
                    DeprecationWarning,
                    stacklevel=2
                )
                set_nested_value(normalized, canonical_path, value)
            
            normalized.pop(key, None)
            
    return normalized


def parse_override(expression: str) -> tuple[str, any]:
    if '=' not in expression:
        raise ValueError(f"Malformed override expression: must contain '=' (got {repr(expression)})")
    key_path, value_str = expression.split('=', 1)
    key_path = key_path.strip()
    if not key_path:
        raise ValueError(f"Malformed override expression: empty key path in {repr(expression)}")
    try:
        value = yaml.safe_load(value_str.strip())
    except Exception as e:
        raise ValueError(f"Malformed override value in expression {repr(expression)}: {e}")
    return key_path, value


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    config_copy = copy.deepcopy(config)
    for expr in overrides or []:
        key_path, value = parse_override(expr)
        check_and_set_nested_value(config_copy, key_path, value)
    return config_copy


def load_resolved_config(path: str) -> dict:
    return load_yaml(path)


def save_resolved_config(config: dict, output_path: str) -> None:
    if output_path:
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=True)


def load_config(
    base_path: str,
    model_path: str = None,
    experiment_path: str = None,
    profile_path: str = None,
    overrides: list[str] = None
) -> ResolvedConfig:
    sources_list = []
    
    cfg = load_yaml(base_path)
    sources_list.append(Path(base_path).resolve())
    cfg = handle_legacy_keys(cfg)
    
    if model_path:
        model_cfg = load_yaml(model_path)
        sources_list.append(Path(model_path).resolve())
        model_cfg = handle_legacy_keys(model_cfg)
        cfg = deep_merge(cfg, model_cfg)
        
    if experiment_path:
        exp_cfg = load_yaml(experiment_path)
        sources_list.append(Path(experiment_path).resolve())
        exp_cfg = handle_legacy_keys(exp_cfg)
        cfg = deep_merge(cfg, exp_cfg)
        
    if profile_path:
        prof_cfg = load_yaml(profile_path)
        sources_list.append(Path(profile_path).resolve())
        prof_cfg = handle_legacy_keys(prof_cfg)
        cfg = deep_merge(cfg, prof_cfg)
        
    if overrides:
        # Check and convert override expressions safely
        processed_overrides = []
        for expr in overrides:
            key_path, value = parse_override(expr)
            if key_path in LEGACY_MAPPINGS:
                canonical = LEGACY_MAPPINGS[key_path]
                warnings.warn(
                    f"Legacy override '{key_path}' is deprecated. Use '{canonical}' instead. Compatibility behavior preserved.",
                    DeprecationWarning,
                    stacklevel=2
                )
                key_path = canonical
            # Keep override values parsed by yaml.safe_load
            # Convert back to a yaml string component for compatibility with apply_overrides if needed
            val_str = yaml.safe_dump(value).strip()
            if val_str.endswith('\n...'):
                val_str = val_str[:-4]
            processed_overrides.append(f"{key_path}={val_str}")
        cfg = apply_overrides(cfg, processed_overrides)
        
    cfg = handle_legacy_keys(cfg)
    
    # Derived properties logic
    if "audio" in cfg and "sample_rate" in cfg["audio"] and "duration" in cfg["audio"]:
        cfg["audio"]["segment_length"] = int(cfg["audio"]["duration"] * cfg["audio"]["sample_rate"])
        cfg["segment_length"] = cfg["audio"]["segment_length"]
        
    if "labels" in cfg and cfg["labels"]:
        num_classes = max(cfg["labels"].values()) + 1
        classes_list = [""] * num_classes
        for folder_name, class_idx in cfg["labels"].items():
            if classes_list[class_idx] == "":
                classes_list[class_idx] = folder_name
            else:
                if "Female" in folder_name:
                    classes_list[class_idx] = "Female"
                elif "Male" in folder_name:
                    classes_list[class_idx] = "Male"
        cfg["classes"] = classes_list
        cfg["num_classes"] = num_classes
        
    validate_config(cfg)
    
    # Deterministic SHA-256 hash using sorted компакт JSON
    canonical_json = json.dumps(cfg, sort_keys=True, separators=(',', ':'))
    sha256_hash = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
    
    return ResolvedConfig(
        data=cfg,
        sources=tuple(sources_list),
        sha256=sha256_hash
    )


def write_resolved_config(resolved: ResolvedConfig, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write yaml file
    with open(path, 'w') as f:
        yaml.safe_dump(resolved.data, f, default_flow_style=False, sort_keys=True)
        
    # Write hash file next to it
    hash_path = path.with_suffix('.sha256')
    with open(hash_path, 'w') as f:
        f.write(resolved.sha256)


# Legacy backward compatibility alias returning a dictionary
def resolve_config(*args, **kwargs) -> dict:
    return load_config(*args, **kwargs).data
