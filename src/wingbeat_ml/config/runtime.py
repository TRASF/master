"""Runtime configuration helpers used by training pipelines."""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from wingbeat_ml.config.schema import AppConfig, ClassWeightsConfig, ReproducibilityConfig


def load_config(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Load a YAML configuration file safely, returning empty dict if missing."""
    if not path or not os.path.exists(path):
        return {}
    from wingbeat_ml.config.loader import load_yaml
    return load_yaml(path)


def apply_reproducibility_environment(settings: Union[AppConfig, ReproducibilityConfig, Dict[str, Any]]) -> None:
    """Apply environment variables for reproducibility based on settings."""
    if isinstance(settings, AppConfig):
        repro = settings.reproducibility
    elif isinstance(settings, ReproducibilityConfig):
        repro = settings
    elif isinstance(settings, dict):
        repro = settings
    else:
        return

    enabled = getattr(repro, "enabled", False) if not isinstance(repro, dict) else repro.get("enabled", False)
    if enabled:
        seed = getattr(repro, "seed", 48) if not isinstance(repro, dict) else repro.get("seed", 48)
        os.environ["PYTHONHASHSEED"] = str(seed)
        deterministic_ops = (
            getattr(repro, "deterministic_ops", True)
            if not isinstance(repro, dict)
            else repro.get("deterministic_ops", True)
        )
        if deterministic_ops:
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
            os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _supports_mixed_float16(tf: Any, gpu: Any) -> bool:
    """Return whether a GPU has Tensor Core-era compute capability."""
    try:
        details = tf.config.experimental.get_device_details(gpu)
        capability = details.get("compute_capability")
        return bool(
            isinstance(capability, (tuple, list))
            and len(capability) >= 2
            and tuple(capability[:2]) >= (7, 0)
        )
    except Exception:
        return False


def configure_compute_policy(settings: Any, *, tf_module: Any = None, gpus: Any = None) -> str:
    """Select a safe global precision policy and return its name."""
    tf = tf_module
    if tf is None:
        import tensorflow as tf

    if hasattr(settings, "precision"):
        requested = str(settings.precision).casefold()
    elif isinstance(settings, dict):
        requested = str(settings.get("precision", "auto")).casefold()
    else:
        requested = "auto"

    if gpus is None:
        gpus = tf.config.list_physical_devices("GPU")

    if requested == "auto":
        policy = (
            "mixed_float16"
            if gpus and all(_supports_mixed_float16(tf, gpu) for gpu in gpus)
            else "float32"
        )
    elif requested in {"float32", "mixed_float16"}:
        policy = requested
    else:
        raise ValueError("performance.precision must be auto, float32, or mixed_float16")

    if policy == "mixed_float16" and not gpus:
        raise RuntimeError("mixed_float16 requires a visible supported GPU")

    tf.keras.mixed_precision.set_global_policy(policy)
    return policy


def configure_training_runtime(
    settings: Union[AppConfig, ReproducibilityConfig, Dict[str, Any]],
    performance: Any = None,
    logging: Any = None,
) -> Dict[str, Any]:
    """Configure reproducibility, devices, console noise, and precision."""
    if isinstance(settings, AppConfig):
        app_cfg = settings
        repro = app_cfg.reproducibility
        performance = app_cfg.performance
        logging = app_cfg.logging
    else:
        repro = settings

    apply_reproducibility_environment(repro)

    console_str = "normal"
    if hasattr(logging, "console"):
        console_str = str(logging.console).casefold()
    elif isinstance(logging, dict):
        console_str = str(logging.get("console", "normal")).casefold()

    if console_str != "verbose":
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    import tensorflow as tf

    enabled = getattr(repro, "enabled", False) if not isinstance(repro, dict) else repro.get("enabled", False)
    seed = getattr(repro, "seed", 48) if not isinstance(repro, dict) else repro.get("seed", 48)

    if enabled:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        if console_str == "verbose":
            print(f"Reproducibility enabled. Seed: {seed}")

    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus and console_str == "verbose":
            print(f"Dynamic GPU memory allocation enabled for {len(gpus)} GPU(s).")
    except Exception as error:
        if console_str != "quiet":
            print(f"Failed to configure dynamic GPU memory allocation: {error}")

    policy = configure_compute_policy(
        performance or {},
        tf_module=tf,
        gpus=gpus,
    )
    return {
        "seed": int(seed),
        "gpu_count": len(gpus),
        "precision_policy": policy,
    }


def resolve_class_weights(
    config_weights: Union[ClassWeightsConfig, Dict[str, Any], Sequence[float], None],
    fallback_weights: Optional[Sequence[float]],
    num_classes: int,
    labels_dict: Optional[Dict[str, int]] = None,
) -> Tuple[bool, Optional[np.ndarray]]:
    """Resolve explicit auto/manual/off class-weight policy."""
    if isinstance(config_weights, ClassWeightsConfig):
        mode = config_weights.mode
        values = config_weights.values
    elif isinstance(config_weights, dict):
        mode = config_weights.get("mode")
        values = config_weights.get("values")
    elif isinstance(config_weights, (list, tuple, np.ndarray)):
        mode = "manual"
        values = list(config_weights)
    else:
        mode = "auto"
        values = None

    if mode is None or (mode == "manual" and values is None and fallback_weights is not None):
        mode = "auto" if values is None else "manual"
    mode = str(mode).casefold()

    if mode in {"off", "none", "disabled"}:
        return False, None

    if mode == "auto":
        if fallback_weights is None:
            raise ValueError("Automatic class weights require training class counts")
        resolved_weights = np.asarray(fallback_weights, dtype=np.float32)
    elif mode == "manual":
        if isinstance(values, dict):
            resolved_weights = np.ones(num_classes, dtype=np.float32)
            assigned = set()
            canonical_names = (
                {str(name).casefold(): int(index) for name, index in labels_dict.items()}
                if labels_dict
                else {}
            )
            for supplied_name, weight in values.items():
                key_str = str(supplied_name).casefold()
                if key_str in canonical_names:
                    class_index = canonical_names[key_str]
                elif str(supplied_name).isdigit() and 0 <= int(supplied_name) < num_classes:
                    class_index = int(supplied_name)
                else:
                    raise ValueError(f"Unknown class weight name: {supplied_name!r}")
                if class_index in assigned:
                    raise ValueError(f"Duplicate class weight for index {class_index}")
                assigned.add(class_index)
                resolved_weights[class_index] = float(weight)
        elif values is None or len(values) != num_classes:
            size = 0 if values is None else len(values)
            raise ValueError(f"class_weights must contain {num_classes} values, got {size}")
        else:
            resolved_weights = np.asarray(values, dtype=np.float32)
    else:
        raise ValueError("class_weights.mode must be auto, manual, or off")

    if resolved_weights.shape != (num_classes,):
        raise ValueError(f"class_weights must contain {num_classes} values, got {resolved_weights.size}")
    if not np.all(np.isfinite(resolved_weights)) or np.any(resolved_weights <= 0):
        raise ValueError("class_weights values must be finite and greater than zero")

    return True, resolved_weights


def generate_experiment_name(config: AppConfig, mode: str = "Pretrain") -> str:
    """Generate structured experiment name based on AppConfig attributes."""
    train_path = config.dataset.train_dir or config.dataset.indoor or ""
    if "indoor" in train_path.lower():
        ds_str = "ds-indoor"
    elif "outdoor" in train_path.lower():
        ds_str = "ds-outdoor"
    elif train_path:
        ds_str = f"ds-{os.path.basename(os.path.normpath(train_path))}"
    else:
        ds_str = "ds-unknown"

    loss_name = config.loss.name
    if "focal" in loss_name.lower():
        loss_str = "loss-Focal"
    elif "crossentropy" in loss_name.lower():
        loss_str = "loss-CE"
    else:
        loss_str = f"loss-{loss_name}"

    cw_mode = config.class_weights.mode
    cw_enabled = cw_mode in {"auto", "manual"}
    cw_str = "cw" if cw_enabled else "nocw"

    # Identify active augmentations
    active_augs = []
    aug = config.augment
    for field_name in aug.model_fields:
        sub = getattr(aug, field_name)
        if hasattr(sub, "p") and getattr(sub, "p", 0.0) > 0.0:
            short_name = field_name.replace("noise_", "").replace("random_", "")
            active_augs.append(short_name)

    aug_str = "aug-" + "-".join(sorted(active_augs)) if active_augs else "noaug"
    opt_name = config.optimizer.name
    lr = config.optimizer.learning_rate
    opt_str = f"{opt_name}-lr{lr}"
    bz = config.train.batch_size
    bz_str = f"bz{bz}"

    return f"{mode}_{ds_str}_{loss_str}_{cw_str}_{aug_str}_{opt_str}_{bz_str}"


def resolve_experiment_paths(config: Union[AppConfig, Dict[str, Any]], experiment_name: str) -> Dict[str, str]:
    """Resolve and return save directories and weight paths for experiment."""
    if isinstance(config, AppConfig):
        experiments_dir = config.runtime.experiments_dir
    elif isinstance(config, dict):
        experiments_dir = config.get("runtime", {}).get("experiments_dir", "models/experiments")
    else:
        experiments_dir = "models/experiments"

    base_dir = os.path.join(
        experiments_dir,
        experiment_name,
    )
    results_dir = os.path.join(base_dir, "results")
    save_path = os.path.join(base_dir, "best_model.weights.h5")

    os.makedirs(results_dir, exist_ok=True)
    return {
        "save_dir": base_dir,
        "results_dir": results_dir,
        "save_path": save_path,
    }
