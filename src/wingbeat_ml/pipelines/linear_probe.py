"""Canonical linear-probe pipeline."""

import os
import sys
import random
import numpy as np
from wingbeat_ml.config.runtime import load_config, normalize_config, apply_reproducibility_environment, resolve_class_weights, generate_experiment_name, resolve_experiment_paths
import tensorflow as tf
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Dynamic GPU memory allocation enabled for {len(gpus)} GPU(s).")
except Exception as e:
    print(f"Failed to configure dynamic GPU memory allocation: {e}")


def train_linear_probe(defaults_path="configs/defaults.yaml",
                       model_cfg_path="configs/model.yaml",
                       pretrained_weights="models/supervised_mossongplus/best_model.weights.h5",
                       save_path=None,
                       results_dir=None):

    # 1. Load and Normalize Configurations
    defaults_raw = load_config(defaults_path)
    cfg = normalize_config(defaults_raw)
    model_cfg = load_config(model_cfg_path)

    # 2. Handle W&B Sweeps and Configuration Merging
    if cfg.get("wandb", {}).get("enabled", False):
        try:
            import wandb
            wandb.init(project=cfg["wandb"].get("project", "MosSongPlus"), config=cfg)

            # Allow W&B Sweep to overwrite config
            for k, v in wandb.config.items():
                if "." in k:
                    parts = k.split(".")
                    if len(parts) == 2 and parts[0] in cfg:
                        cfg[parts[0]][parts[1]] = v
                    elif len(parts) == 3 and parts[0] in cfg and parts[1] in cfg[parts[0]]:
                        cfg[parts[0]][parts[1]][parts[2]] = v
        except ImportError:
            print("WandB is enabled in config but 'wandb' package is not installed.")

    # 3. Dynamic Experiment Naming & Path Resolution (Run once!)
    exp_name = generate_experiment_name(cfg, mode="LP")
    if cfg.get("wandb", {}).get("enabled", False) and 'wandb' in sys.modules:
        import wandb
        if wandb.run is not None:
            wandb.run.name = exp_name

    resolved_paths = resolve_experiment_paths(cfg, exp_name)
    if save_path is None:
        save_path = resolved_paths["save_path"]
    if results_dir is None:
        results_dir = resolved_paths["results_dir"]

    print(f"Experiment Name: {exp_name}")
    print(f"Saving weights to: {save_path}")
    print(f"Saving results to: {results_dir}")

    apply_reproducibility_environment(cfg["reproducibility"])

    # Ensure classification output activation is softmax if not set
    if cfg["model"].get("output_activation") is None:
        cfg["model"]["output_activation"] = "softmax"

    # Ensure consistency between activation and from_logits
    if cfg["model"]["output_activation"] == "softmax":
        cfg["loss"]["from_logits"] = False
    else:
        cfg["loss"]["from_logits"] = True

    # Set seeds for reproducibility
    if cfg["reproducibility"]["enabled"]:
        seed = cfg["reproducibility"]["seed"]
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print(f"Reproducibility enabled. Seed: {seed}")

    from wingbeat_ml.data.dataset import SupervisedDataset
    from wingbeat_ml.evaluation import ModelEvaluator
    from wingbeat_ml.models import MosSongPlusModel
    from wingbeat_ml.training import LossFactory
    from wingbeat_ml.pipelines.train import run_training

    # 4. Setup Dataset
    print("Setting up datasets...")
    ds_builder = SupervisedDataset(
        dataset_dir=cfg["dataset"].get("train_dir") or cfg["dataset"]["indoor"],
        val_dir=cfg["dataset"]["val_dir"],
        test_dir=cfg["dataset"]["test_dir"],
        sample_rate=cfg["audio"]["sample_rate"],
        segment_length=cfg["audio"]["segment_length"],
        classes=cfg["classes"],
        noise_dirs=cfg["augment"]["noise_banks"],
        augment_cfg=cfg["augment"],
        seed=cfg["reproducibility"]["seed"],
        deterministic=cfg["reproducibility"]["deterministic_data"],
        nomos_index=cfg["nomos_index"]
    )

    train_ds, val_ds, test_ds = ds_builder.build(
        split=cfg["dataset"]["split_list"],
        batch_size=cfg["train"]["batch_size"],
        shuffle=cfg["train"]["shuffle"]
    )

    # 5. Build Model
    print("Building model...")
    model_builder = MosSongPlusModel(model_cfg, model_overrides=cfg.get("model"))
    model = model_builder.build(
        input_shape=(cfg["audio"]["segment_length"], 1),
        output_units=cfg["num_classes"],
        output_activation=cfg["model"]["output_activation"]
    )

    # LOAD PRETRAINED WEIGHTS
    if os.path.exists(pretrained_weights):
        print(f"Loading pre-trained contrastive weights from {pretrained_weights}...")
        model.load_weights(pretrained_weights)
    else:
        print(f"WARNING: Pre-trained weights not found at {pretrained_weights}! Training from scratch.")

    model.summary()

    # 6. Resolve Class Weights
    class_weights_enabled, class_weights = resolve_class_weights(
        cfg["class_weights"],
        ds_builder.class_weights,
        cfg["num_classes"],
        labels_dict=cfg["labels"]
    )

    if class_weights_enabled:
        print(f"Using class weights: {np.round(class_weights, 3).tolist()}")
        cfg["resolved_class_weights"] = class_weights.tolist()
    else:
        cfg["resolved_class_weights"] = None

    # 7. Setup Optimizer and Loss
    loss_fn = LossFactory.get_loss(cfg)
    evaluator = ModelEvaluator(model, cfg["classes"], loss_fn)

    epochs = cfg["train"]["epochs"]
    cfg["training_mode"] = "linear_probe"
    print("\n--- Linear Probing (Training Only Dense Head) ---")
    print(f"Starting linear probe training for {epochs} epochs...")

    def print_epoch(epoch, logs):
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"loss: {logs['train_loss']:.4f} - "
            f"acc: {logs['train_accuracy']:.4f} | "
            f"val_loss: {logs['val_loss']:.4f} - "
            f"val_acc: {logs['val_accuracy']:.4f} | "
            f"val_f1: {logs['val_macro_f1']:.3f} | "
            f"Time: {logs['epoch_duration_seconds']:.1f}s"
        )

    run_training(
        model,
        train_ds,
        cfg,
        evaluate_epoch=lambda: evaluator.evaluate_epoch(val_ds),
        on_epoch_end=print_epoch,
        class_weights=(class_weights if class_weights_enabled else None),
        save_path=save_path,
    )

    # Final Evaluation on Test Set
    print("\nTraining complete. Running final evaluation on test set...")
    if os.path.exists(save_path):
        model.load_weights(save_path)
    test_results = evaluator.evaluate_final_test(test_ds, save_dir=results_dir, return_predictions=True)

    # File-level evaluation
    print("\nRunning file-level evaluation on test set...")
    file_results = evaluator.evaluate_files(
        file_paths=ds_builder.test_paths,
        labels=ds_builder.test_labels,
        load_fn=ds_builder.data_loader.load_file,
        augmentor=ds_builder.augmentor,
        batch_size=cfg["train"]["batch_size"],
        save_dir=results_dir
    )

    print("\nRunning file-level evaluation on training set...")
    train_file_results = evaluator.evaluate_files(
        file_paths=ds_builder.train_paths,
        labels=ds_builder.train_labels,
        load_fn=ds_builder.data_loader.load_file,
        augmentor=ds_builder.augmentor,
        batch_size=cfg["train"]["batch_size"],
        save_dir=results_dir,
        filename="train_file_level_results.yaml"
    )

    # Log/Report Results
    from wingbeat_ml.evaluation import report_results
    report_results(
        model=model,
        test_results=test_results,
        file_results=file_results,
        train_file_results=train_file_results,
        cfg=cfg,
        ds_builder=ds_builder,
        save_path=save_path,
        results_dir=results_dir,
        artifact_name='mossongplus-linearprobe',
        val_ds=val_ds,
        test_ds=test_ds,
        evaluator=evaluator
    )


if __name__ == "__main__":
    train_linear_probe()
