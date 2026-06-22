import yaml
import os
import numpy as np


def load_config(path):
    """Load a YAML configuration file."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def extract_audio_settings(defaults):
    """Return normalized audio configuration values from defaults."""
    audio_cfg = defaults.get('audio', {})
    sample_rate = int(audio_cfg.get('sample_rate', 8000))
    duration = float(audio_cfg.get('duration', 0.3))
    segment_length = int(duration * sample_rate)
    overlap = audio_cfg.get('overlap', [0.0, 0.8])

    return {
        'sample_rate': sample_rate,
        'duration': duration,
        'segment_length': segment_length,
        'overlap': overlap,
    }


def extract_train_settings(defaults):
    """Return normalized training configuration values from defaults."""
    train_cfg = defaults.get('train', {})
    val_overlap = float(train_cfg.get('val_overlap', 0.7))
    step_ratio = 1.0 - val_overlap
    
    return {
        'batch_size': int(train_cfg.get('batch_size', 32)),
        'shuffle': bool(train_cfg.get('shuffle', True)),
        'epochs': int(train_cfg.get('epochs', 100)),
        'val_overlap': val_overlap,
        'step_ratio': step_ratio,
        'seed': int(train_cfg.get('seed', 42)),
    }


def extract_dataset_settings(defaults):
    """Return normalized dataset configuration values from defaults."""
    dataset_cfg = defaults.get('dataset', {})
    split_ratios = dataset_cfg.get('split_ratios', {'train': 0.8, 'val': 0.1, 'test': 0.1})
    
    return {
        'indoor': dataset_cfg.get('indoor', 'dataset/MSB/Indoor'),
        'outdoor': dataset_cfg.get('outdoor', 'dataset/MSB/Outdoor'),
        'moslab': dataset_cfg.get('mosLab', 'dataset/Philip'),
        'val_dir': dataset_cfg.get('val_dir'),
        'test_dir': dataset_cfg.get('test_dir'),
        'split_ratios': split_ratios,
        'split_list': [
            float(split_ratios.get('train', 0.8)),
            float(split_ratios.get('val', 0.1)),
            float(split_ratios.get('test', 0.1))
        ]
    }


def extract_augment_settings(defaults):
    """Return normalized augmentation configuration values from defaults."""
    augment_cfg = defaults.get('augment', {})

    def merge_cfg(key, defaults_dict):
        user_val = augment_cfg.get(key, {})
        if user_val is None:
            return defaults_dict
        return {**defaults_dict, **user_val}

    return {
        'noise_banks': augment_cfg.get('noise_banks', []),
        'noise_overlay': merge_cfg('noise_overlay', {
            'p': 0.0,
            'snr_db': [10, 20],
            'envelope_gain': [0.7, 1.0],
            'post_gain_db': [-6.0, 3.0]
        }),
        'pitch_shift': merge_cfg('pitch_shift', {'p': 0.0, 'semitones': [-0.5, 0.5]}),
        'time_shift': merge_cfg('time_shift', {'p': 0.0, 'rate': [-0.1, 0.1]}),
        'random_gain': merge_cfg('random_gain', {'p': 0.0, 'gain_db': [-6, 6]}),
        'gaussian_noise': merge_cfg('gaussian_noise', {'p': 0.0, 'snr_db': [10, 20]}),
        'time_masking': merge_cfg('time_masking', {'p': 0.0, 'num_masks': 1, 'max_mask_size': 400}),
        'pre_emphasis': merge_cfg('pre_emphasis', {'p': 0.0, 'coeff': 0.97}),
        'high_pass': merge_cfg('high_pass', {'p': 0.0, 'fc': 150}),
        'rms_norm': merge_cfg('rms_norm', {'p': 0.0, 'target_rms': 0.05}),
        'train_overlap': augment_cfg.get('train_overlap', [0.0, 0.8]),
        'config': augment_cfg,  # Keep original for any custom needs
    }


def extract_model_settings(defaults):
    """Return normalized model configuration values from defaults."""
    model_cfg = defaults.get('model', {})
    return {
        'output_activation': model_cfg.get('output_activation'), # null in YAML is None
    }


def extract_reproducibility_settings(defaults):
    """Return normalized reproducibility configuration values from defaults."""
    cfg = defaults.get("reproducibility", {})
    if isinstance(cfg, bool):
        cfg = {"enabled": cfg}

    enabled = bool(cfg.get("enabled", False))
    # Try to get seed from reproducibility, then train, then root, fallback to 42
    default_seed = defaults.get("train", {}).get("seed", defaults.get("seed", 42))
    seed = int(cfg.get("seed", default_seed))
    deterministic_ops = bool(cfg.get("deterministic_ops", enabled))
    deterministic_data = bool(cfg.get("deterministic_data", enabled))

    return {
        "enabled": enabled,
        "seed": seed,
        "deterministic_ops": deterministic_ops,
        "deterministic_data": deterministic_data,
    }


def apply_reproducibility_environment(settings):
    """Apply environment variables for reproducibility based on settings."""
    if settings.get("enabled"):
        seed = settings.get("seed", 42)
        os.environ["PYTHONHASHSEED"] = str(seed)
        if settings.get("deterministic_ops"):
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
            os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def resolve_class_weights(config_weights, fallback_weights, num_classes):
    """Resolve class weights from config or fallback values."""
    if config_weights is None:
        return True, fallback_weights

    enabled = True
    values = config_weights

    if isinstance(config_weights, dict) and any(
        key in config_weights for key in ("enabled", "values")
    ):
        enabled = bool(config_weights.get("enabled", True))
        values = config_weights.get("values")

    if not enabled:
        return False, None

    if values is None:
        return True, fallback_weights

    if isinstance(values, dict):
        return True, np.array(
            [float(values.get(i, values.get(str(i), 1.0))) for i in range(num_classes)],
            dtype=np.float32,
        )

    if len(values) != num_classes:
        raise ValueError(f"class_weights must contain {num_classes} values, got {len(values)}")

    return True, np.array(values, dtype=np.float32)


def extract_optimizer_settings(defaults):
    """Return normalized optimizer configuration values from defaults."""
    opt_cfg = defaults.get('optimizer', {'name': 'Adam', 'learning_rate': 0.001})
    if opt_cfg is None:
        opt_cfg = {'name': 'Adam', 'learning_rate': 0.001}
    return opt_cfg


def extract_loss_settings(defaults):
    """Return normalized loss configuration values from defaults."""
    loss_cfg = defaults.get('loss', {'name': 'CategoricalCrossentropy'})
    if loss_cfg is None:
        loss_cfg = {'name': 'CategoricalCrossentropy'}
    return loss_cfg


def extract_callback_settings(defaults):
    """Return normalized callback configuration values from defaults."""
    return defaults.get('callbacks', {})


def normalize_config(defaults):
    """Consolidate all configuration extractions into a single normalized dictionary."""
    if not defaults:
        defaults = {}
        
    normalized = {
        'audio': extract_audio_settings(defaults),
        'train': extract_train_settings(defaults),
        'dataset': extract_dataset_settings(defaults),
        'augment': extract_augment_settings(defaults),
        'model': extract_model_settings(defaults),
        'reproducibility': extract_reproducibility_settings(defaults),
        'labels': defaults.get('labels', {}),
        'class_weights': defaults.get('class_weights'),
        'optimizer': extract_optimizer_settings(defaults),
        'loss': extract_loss_settings(defaults),
        'callbacks': extract_callback_settings(defaults),
    }
    
    # Add derived/helper values at the top level
    normalized['classes'] = list(normalized['labels'].keys())
    normalized['num_classes'] = len(normalized['labels'])
    normalized['segment_length'] = normalized['audio']['segment_length']
    
    # Find No.Mos index
    normalized['nomos_index'] = None
    for i, name in enumerate(normalized['classes']):
        if "No.Mos" in name or "Nomos" in name:
            normalized['nomos_index'] = i
            break
    
    return normalized
