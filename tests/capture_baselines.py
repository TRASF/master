import os
import sys
import json
import numpy as np
import tensorflow as tf

# Force CPU execution for baselines to match test environment
tf.config.set_visible_devices([], 'GPU')

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.mos_config import load_config, normalize_config, apply_reproducibility_environment, resolve_experiment_paths
from src.framework.supervised.dataset import SupervisedDataset
from wingbeat_ml.models import MosSongPlusModel

def capture_baselines():
    # 1. Load configuration and override directories to point to our fixtures
    defaults_path = "configs/defaults.yaml"
    model_cfg_path = "configs/model.yaml"

    defaults_raw = load_config(defaults_path)
    # Force defaults to use our fixture directories
    defaults_raw["dataset"]["train_dir"] = "tests/fixtures/audio_11class"
    defaults_raw["dataset"]["val_dir"] = None
    defaults_raw["dataset"]["test_dir"] = None
    defaults_raw["dataset"]["split_list"] = [0.6, 0.2, 0.2]
    # Disable noise overlay for testing simplicity since we don't have local noise dirs in test run
    if "augment" not in defaults_raw:
        defaults_raw["augment"] = {}
    if "noise_overlay" not in defaults_raw["augment"]:
        defaults_raw["augment"]["noise_overlay"] = {}
    defaults_raw["augment"]["noise_overlay"]["p"] = 0.0

    # Set seed
    defaults_raw["reproducibility"]["seed"] = 45
    defaults_raw["reproducibility"]["deterministic_data"] = True
    defaults_raw["reproducibility"]["deterministic_ops"] = True
    defaults_raw["reproducibility"]["enabled"] = True
    # Disable W&B logging in test run
    defaults_raw["wandb"] = {"enabled": False}

    cfg = normalize_config(defaults_raw)
    model_cfg = load_config(model_cfg_path)

    # Apply reproducibility environment
    apply_reproducibility_environment(cfg["reproducibility"])
    tf.random.set_seed(45)
    np.random.seed(45)

    baseline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "parity/baselines"))
    os.makedirs(baseline_dir, exist_ok=True)

    print("Building SupervisedDataset...")
    ds_builder = SupervisedDataset(
        dataset_dir=cfg["dataset"]["train_dir"],
        val_dir=cfg["dataset"]["val_dir"],
        test_dir=cfg["dataset"]["test_dir"],
        sample_rate=cfg["audio"]["sample_rate"],
        segment_length=cfg["audio"]["segment_length"],
        classes=cfg["classes"],
        noise_dirs=None,  # No noise bank in baseline test
        augment_cfg=cfg["augment"],
        seed=cfg["reproducibility"]["seed"],
        deterministic=cfg["reproducibility"]["deterministic_data"],
        nomos_index=cfg["nomos_index"],
        labels_dict=cfg["labels"]
    )

    # Gather files
    train_paths, train_labels = ds_builder.data_loader.gather_files()

    # Capture 1. File Discovery
    file_discovery = {
        "files": [os.path.basename(p) for p in sorted(train_paths)],
        "labels": [int(l) for _, l in sorted(zip(train_paths, train_labels))]
    }
    with open(os.path.join(baseline_dir, "file_discovery.json"), "w") as f:
        json.dump(file_discovery, f, indent=2)
    print("Saved file_discovery.json")

    # Capture 2. Label Map
    label_map = {
        "classes": cfg["classes"],
        "class_to_idx": cfg["labels"],
        "num_classes": cfg["num_classes"]
    }
    with open(os.path.join(baseline_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    print("Saved label_map.json")

    # Build dataset splits
    train_ds, val_ds, test_ds = ds_builder.build(
        split=cfg["dataset"]["split_list"],
        batch_size=2, # small batch size for tests
        shuffle=False # turn off shuffle to capture clean splits
    )

    # Capture 3. Splits
    splits = {
        "train_paths": [os.path.basename(p) for p in ds_builder.train_paths],
        "train_labels": [int(l) for l in ds_builder.train_labels],
        "val_paths": [os.path.basename(p) for p in ds_builder.val_paths],
        "val_labels": [int(l) for l in ds_builder.val_labels],
        "test_paths": [os.path.basename(p) for p in ds_builder.test_paths],
        "test_labels": [int(l) for l in ds_builder.test_labels]
    }
    with open(os.path.join(baseline_dir, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)
    print("Saved splits.json")

    # Capture 4. Preprocessing Outputs (Clean loaded audio & preprocessed audio)
    preprocessed_dict = {}
    for path, label in zip(train_paths, train_labels):
        name = os.path.basename(path)
        raw_audio = ds_builder.data_loader.load_file(path)
        # Sliced/padded to match segment length (2400)
        sliced_audio = raw_audio[:ds_builder.segment_length]
        if len(sliced_audio) < ds_builder.segment_length:
            sliced_audio = np.pad(sliced_audio, (0, ds_builder.segment_length - len(sliced_audio)))

        audio_tensor = tf.convert_to_tensor(sliced_audio, dtype=tf.float32)
        preprocessed_audio, _ = ds_builder.augmentor.apply_post_processing(
            audio_tensor, tf.constant(label, dtype=tf.int32), augment=False
        )
        preprocessed_dict[f"{name}_raw"] = sliced_audio
        preprocessed_dict[f"{name}_preprocessed"] = preprocessed_audio.numpy()

    np.savez(os.path.join(baseline_dir, "preprocessing_outputs.npz"), **preprocessed_dict)
    print("Saved preprocessing_outputs.npz")

    # Capture 5. Augmentation Outputs (Fixed seed)
    augmented_dict = {}
    for i, (path, label) in enumerate(zip(train_paths[:3], train_labels[:3])):
        name = os.path.basename(path)
        raw_audio = ds_builder.data_loader.load_file(path)
        sliced_audio = raw_audio[:ds_builder.segment_length]
        if len(sliced_audio) < ds_builder.segment_length:
            sliced_audio = np.pad(sliced_audio, (0, ds_builder.segment_length - len(sliced_audio)))

        audio_tensor = tf.convert_to_tensor(sliced_audio, dtype=tf.float32)

        # Enable time shift, random gain, high pass, mixup to check all transforms
        test_augment_cfg = {
            "preprocess": {"dc_removal": True},
            "rms_norm": {"target_rms": 0.05, "min_gain": 0.05, "max_gain": 15.0},
            "high_pass": {"p": 1.0, "fc": 150},
            "time_shift": {"p": 1.0, "rate": [-0.05, 0.05]},
            "random_gain": {"p": 1.0, "gain_db": [-3.0, 3.0]},
            "pitch_shift": {"p": 1.0, "semitones": [-0.2, 0.2]}
        }
        test_augmentor = ds_builder.augmentor.__class__(
            segment_length=cfg["audio"]["segment_length"],
            config=test_augment_cfg,
            seed=45,
            deterministic=True,
            nomos_index=ds_builder.nomos_index
        )

        # Test individual transformations
        hpf_audio = test_augmentor.apply_hpf(audio_tensor)
        time_shifted = test_augmentor.time_shift(audio_tensor, [-0.05, 0.05], tf.constant([45, 1], dtype=tf.int64))
        gained = test_augmentor.random_gain(audio_tensor, [-3.0, 3.0], tf.constant([45, 2], dtype=tf.int64))
        pitch_shifted = test_augmentor.pitch_shift(audio_tensor, [-0.2, 0.2], tf.constant([45, 3], dtype=tf.int64))

        # Save outputs
        augmented_dict[f"{name}_hpf"] = hpf_audio.numpy()
        augmented_dict[f"{name}_timeshift"] = time_shifted.numpy()
        augmented_dict[f"{name}_gain"] = gained.numpy()
        augmented_dict[f"{name}_pitchshift"] = pitch_shifted.numpy()

    np.savez(os.path.join(baseline_dir, "augmentation_outputs.npz"), **augmented_dict)
    print("Saved augmentation_outputs.npz")

    # Capture 6. Model Structure
    print("Building model...")
    import tensorflow.keras as keras
    keras.backend.clear_session()
    keras.utils.set_random_seed(45)
    model_builder = MosSongPlusModel(model_cfg, model_overrides=cfg.get("model"))
    model = model_builder.build(
        input_shape=(cfg["audio"]["segment_length"], 1),
        output_units=cfg["num_classes"],
        output_activation=cfg["model"]["output_activation"]
    )

    model_structure = {
        "name": model.name,
        "layers": []
    }
    for layer in model.layers:
        try:
            out_shape = layer.output_shape
        except AttributeError:
            try:
                out_shape = layer.input_shape
            except AttributeError:
                out_shape = None
        model_structure["layers"].append({
            "name": layer.name,
            "class": layer.__class__.__name__,
            "output_shape": out_shape,
            "trainable": layer.trainable,
            "weight_shapes": [list(w.shape) for w in layer.weights]
        })
    model_structure["total_params"] = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))

    with open(os.path.join(baseline_dir, "model_structure.json"), "w") as f:
        json.dump(model_structure, f, indent=2)
    print("Saved model_structure.json")

    # Capture 7. Initial Predictions
    # Create deterministic input tensor (e.g. ones)
    dummy_input = np.ones((5, cfg["audio"]["segment_length"], 1), dtype=np.float32)
    initial_preds = model.predict(dummy_input)
    np.savez(os.path.join(baseline_dir, "initial_predictions.npz"), preds=initial_preds)
    print("Saved initial_predictions.npz")

    # Capture 8. Metrics Definition
    metrics = {
        "evaluation_metrics": [
            "loss", "accuracy", "macro_f1", "macro_precision", "macro_recall",
            "female_f1", "female_prec", "female_rec",
            "male_f1", "male_prec", "male_rec"
        ]
    }
    with open(os.path.join(baseline_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics.json")

    # Capture 9. Output Paths
    output_paths = {
        "pretrain": resolve_experiment_paths(cfg, "Pretrain_test_exp"),
        "linear_probe": resolve_experiment_paths(cfg, "LP_test_exp"),
        "fine_tune": resolve_experiment_paths(cfg, "FT_test_exp")
    }
    with open(os.path.join(baseline_dir, "output_paths.json"), "w") as f:
        json.dump(output_paths, f, indent=2)
    print("Saved output_paths.json")

    # Capture 10. W&B Keys
    wandb_keys = {
        "init_keys": ["project", "config", "group", "tags", "job_type"],
        "logged_keys": [
            "epoch", "train_loss", "train_accuracy", "learning_rate", "epoch_duration_seconds",
            "val_loss", "val_accuracy", "val_macro_f1", "val_female_f1", "val_male_f1"
        ]
    }
    with open(os.path.join(baseline_dir, "wandb_keys.json"), "w") as f:
        json.dump(wandb_keys, f, indent=2)
    print("Saved wandb_keys.json")

    print("\nBaseline capture completed successfully!")

if __name__ == "__main__":
    capture_baselines()
