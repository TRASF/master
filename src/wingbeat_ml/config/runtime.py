"""Runtime configuration helpers used by training pipelines."""

import os
import random

import numpy as np
import yaml


def load_config(path):
    """Load a YAML configuration file."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def recursive_merge(default, override):
    """Recursively merge override dictionary into default dictionary."""
    if not isinstance(override, dict):
        return override
    merged = default.copy() if default else {}
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = recursive_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def apply_reproducibility_environment(settings):
    """Apply environment variables for reproducibility based on settings."""
    if settings.get("enabled"):
        seed = settings["seed"]
        os.environ["PYTHONHASHSEED"] = str(seed)
        if settings.get("deterministic_ops"):
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
            os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _supports_mixed_float16(tf, gpu):
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


def configure_compute_policy(settings, *, tf_module=None, gpus=None):
    """Select a safe global precision policy and return its name."""
    tf = tf_module
    if tf is None:
        import tensorflow as tf

    requested = str(settings.get("precision", "auto")).casefold()
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
        raise ValueError(
            "performance.precision must be auto, float32, or mixed_float16"
        )

    if policy == "mixed_float16" and not gpus:
        raise RuntimeError("mixed_float16 requires a visible supported GPU")

    tf.keras.mixed_precision.set_global_policy(policy)
    return policy


def configure_training_runtime(settings, performance=None, logging=None):
    """Configure reproducibility, devices, console noise, and precision."""
    apply_reproducibility_environment(settings)

    console = str((logging or {}).get("console", "normal")).casefold()
    if console != "verbose":
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    import tensorflow as tf

    if settings.get("enabled"):
        seed = settings["seed"]
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        if console == "verbose":
            print(f"Reproducibility enabled. Seed: {seed}")

    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus and console == "verbose":
            print(
                "Dynamic GPU memory allocation enabled for "
                f"{len(gpus)} GPU(s)."
            )
    except Exception as error:
        if console != "quiet":
            print(f"Failed to configure dynamic GPU memory allocation: {error}")

    policy = configure_compute_policy(
        performance or {},
        tf_module=tf,
        gpus=gpus,
    )
    return {
        "gpu_count": len(gpus),
        "precision_policy": policy,
    }


def resolve_class_weights(config_weights, fallback_weights, num_classes, labels_dict=None):
    """Resolve explicit auto/manual/off class-weight policy."""
    config_weights = config_weights or {"mode": "auto"}
    if not isinstance(config_weights, dict):
        config_weights = {"mode": "manual", "values": config_weights}

    mode = config_weights.get("mode")
    if mode is None:
        if not bool(config_weights.get("enabled", True)):
            mode = "off"
        elif config_weights.get("values") is None:
            mode = "auto"
        else:
            mode = "manual"
    mode = str(mode).casefold()

    if mode == "off":
        return False, None
    if mode == "auto":
        if fallback_weights is None:
            raise ValueError("Automatic class weights require training class counts")
        resolved_weights = np.asarray(fallback_weights, dtype=np.float32)
    elif mode == "manual":
        values = config_weights.get("values")
        if isinstance(values, dict):
            if not labels_dict:
                raise ValueError("Manual class weights require the canonical label map")
            canonical_names = {
                str(name).casefold(): (str(name), int(index))
                for name, index in labels_dict.items()
            }
            resolved_weights = np.empty(num_classes, dtype=np.float32)
            assigned = set()
            for supplied_name, weight in values.items():
                match = canonical_names.get(str(supplied_name).casefold())
                if match is None:
                    raise ValueError(
                        f"Unknown class weight name: {supplied_name!r}"
                    )
                _, class_index = match
                if class_index in assigned:
                    raise ValueError(
                        f"Duplicate class weight for index {class_index}"
                    )
                assigned.add(class_index)
                resolved_weights[class_index] = float(weight)
            missing = sorted(set(range(num_classes)) - assigned)
            if missing:
                raise ValueError(
                    f"Manual class weights are missing class indices: {missing}"
                )
        elif values is None or len(values) != num_classes:
            size = 0 if values is None else len(values)
            raise ValueError(
                f"class_weights must contain {num_classes} values, got {size}"
            )
        else:
            resolved_weights = np.asarray(values, dtype=np.float32)
    else:
        raise ValueError("class_weights.mode must be auto, manual, or off")

    if resolved_weights.shape != (num_classes,):
        raise ValueError(
            f"class_weights must contain {num_classes} values, "
            f"got {resolved_weights.size}"
        )
    if not np.all(np.isfinite(resolved_weights)) or np.any(resolved_weights <= 0):
        raise ValueError("class_weights values must be finite and greater than zero")

    # Apply overrides if present (e.g. from W&B Sweep)
    if isinstance(config_weights, dict) and "override" in config_weights:
        overrides = config_weights["override"]
        if isinstance(overrides, dict) and resolved_weights is not None:
            resolved_weights = np.array(resolved_weights, copy=True)
            for class_name, multiplier in overrides.items():
                if labels_dict and class_name in labels_dict:
                    class_idx = labels_dict[class_name]
                    resolved_weights[class_idx] *= float(multiplier)
                elif str(class_name).isdigit():
                    class_idx = int(class_name)
                    if 0 <= class_idx < num_classes:
                        resolved_weights[class_idx] *= float(multiplier)

    return True, resolved_weights


def normalize_config(defaults):
    """Consolidate all configuration extractions into a single normalized dictionary."""
    # 1. Load absolute base defaults from configs/defaults.yaml
    defaults_yaml_path = os.path.join(os.path.dirname(__file__), "defaults.yaml")
    base_defaults = load_config(defaults_yaml_path)

    # 2. Merge user override config (defaults) into the base config
    cfg = recursive_merge(base_defaults, defaults or {})

    # 3. Handle derived properties and conversions
    sample_rate = int(cfg["audio"]["sample_rate"])
    duration = float(cfg["audio"]["duration"])
    cfg["audio"]["segment_length"] = int(duration * sample_rate)
    cfg["segment_length"] = cfg["audio"]["segment_length"]

    cfg["preprocess"]["dc_removal"] = bool(
        cfg["preprocess"]["dc_removal"]
    )
    cfg["augment"]["preprocess"] = cfg["preprocess"]

    split_ratios = cfg["dataset"]["split_ratios"]
    cfg["dataset"]["split_list"] = [
        float(split_ratios["train"]),
        float(split_ratios["val"]),
        float(split_ratios["test"]),
    ]

    # Resolve classes list and number of classes
    labels_dict = cfg["labels"]
    if labels_dict:
        num_classes = max(labels_dict.values()) + 1
        classes_list = [""] * num_classes
        for folder_name, class_idx in labels_dict.items():
            if classes_list[class_idx] == "":
                classes_list[class_idx] = folder_name
            else:
                if "Female" in folder_name:
                    classes_list[class_idx] = "Female"
                elif "Male" in folder_name:
                    classes_list[class_idx] = "Male"
        cfg["classes"] = classes_list
        cfg["num_classes"] = num_classes
    else:
        cfg["classes"] = []
        cfg["num_classes"] = 0

    configured_nomos_index = cfg.get("nomos_index")
    cfg["nomos_index"] = (
        int(configured_nomos_index)
        if configured_nomos_index is not None
        else None
    )
    if cfg["nomos_index"] is None:
        for i, name in enumerate(cfg["classes"]):
            compact_name = "".join(c for c in name.casefold() if c.isalnum())
            if compact_name == "nomos":
                cfg["nomos_index"] = i
                break

    return cfg


def generate_experiment_name(cfg, mode="Pretrain"):
    """
    Generate a structured, dynamic experiment name based on configuration parameters.
    Format: [Mode]_[Dataset]_[Loss]_[CW]_[Aug]_[Optimizer_LR]_[BZ]
    """
    import os
    train_path = cfg.get("dataset", {}).get("train_dir") or cfg.get("dataset", {}).get("indoor", "")
    if "indoor" in train_path.lower():
        ds_str = "ds-indoor"
    elif "outdoor" in train_path.lower():
        ds_str = "ds-outdoor"
    elif train_path:
        ds_str = f"ds-{os.path.basename(os.path.normpath(train_path))}"
    else:
        ds_str = "ds-unknown"

    loss_name = cfg["loss"]["name"]
    if "focal" in loss_name.lower():
        loss_str = "loss-Focal"
    elif "crossentropy" in loss_name.lower():
        loss_str = "loss-CE"
    else:
        loss_str = f"loss-{loss_name}"

    cw_mode = cfg["class_weights"].get("mode")
    cw_enabled = (
        str(cw_mode).casefold() != "off"
        if cw_mode is not None
        else bool(cfg["class_weights"].get("enabled", True))
    )
    cw_str = "cw" if cw_enabled else "nocw"

    augment_cfg = cfg.get("augment", {})
    active_augs = []
    for key, val in augment_cfg.items():
        if isinstance(val, dict) and val.get("p", 0.0) > 0.0:
            short_name = key.replace("noise_", "").replace("random_", "")
            active_augs.append(short_name)

    aug_str = "aug-" + "-".join(sorted(active_augs)) if active_augs else "noaug"
    opt_name = cfg["optimizer"]["name"]
    lr = cfg["optimizer"]["learning_rate"]
    opt_str = f"{opt_name}-lr{lr}"
    bz = cfg["train"]["batch_size"]
    bz_str = f"bz{bz}"

    return f"{mode}_{ds_str}_{loss_str}_{cw_str}_{aug_str}_{opt_str}_{bz_str}"


def resolve_experiment_paths(cfg, experiment_name):
    """
    Resolve and return save directories and weight paths for the given experiment name.
    Automatically creates the directories if they don't exist.
    """
    import os
    base_dir = os.path.join(
        cfg["runtime"]["experiments_dir"],
        experiment_name,
    )
    results_dir = os.path.join(base_dir, "results")
    save_path = os.path.join(base_dir, "best_model.weights.h5")

    return {
        "save_dir": base_dir,
        "results_dir": results_dir,
        "save_path": save_path
    }
