"""Canonical pretraining pipeline."""

import os
import sys
import random
import numpy as np
from wingbeat_ml.config.runtime import load_config, normalize_config, apply_reproducibility_environment, resolve_class_weights, generate_experiment_name, resolve_experiment_paths


def train_supervised(defaults_path="configs/defaults.yaml",
                     model_cfg_path="configs/model.yaml",
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
            wandb_cfg = cfg.get("wandb", {})
            wandb.init(
                project=wandb_cfg.get("project", "MosSongPlus"),
                config=cfg,
                group=wandb_cfg.get("group"),
                tags=wandb_cfg.get("tags"),
                job_type=wandb_cfg.get("job_type"),
            )

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
    base_exp_name = generate_experiment_name(cfg, mode="Pretrain")
    if cfg.get("wandb", {}).get("enabled", False) and 'wandb' in sys.modules:
        import wandb
        if wandb.run is not None:
            hpf_p = cfg.get("augment", {}).get("high_pass", {}).get("p", 0.0)
            seed = cfg.get("reproducibility", {}).get("seed", cfg.get("train", {}).get("seed", "seed"))
            task = cfg.get("wandb", {}).get("group") or f"{cfg.get('num_classes', 'n')}class"
            exp_name = f"{task}_{base_exp_name}_hpf{hpf_p}_seed{seed}"
            wandb.run.name = exp_name
        else:
            exp_name = base_exp_name
    else:
        exp_name = base_exp_name

    resolved_paths = resolve_experiment_paths(cfg, exp_name)
    if save_path is None:
        save_path = resolved_paths["save_path"]
    if results_dir is None:
        results_dir = resolved_paths["results_dir"]

    print(f"Experiment Name: {exp_name}")
    print(f"Saving weights to: {save_path}")
    print(f"Saving results to: {results_dir}")

    # Set seeds for reproducibility
    apply_reproducibility_environment(cfg["reproducibility"])
    if cfg["reproducibility"]["enabled"]:
        seed = cfg["reproducibility"]["seed"]
        random.seed(seed)
        np.random.seed(seed)
        import tensorflow as tf
        tf.random.set_seed(seed)
        print(f"Reproducibility enabled. Seed: {seed}")

    import tensorflow as tf
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Dynamic GPU memory allocation enabled for {len(gpus)} GPU(s).")
    except Exception as e:
        print(f"Failed to configure dynamic GPU memory allocation: {e}")

    from wingbeat_ml.data.dataset import SupervisedDataset
    from wingbeat_ml.training import Trainer
    from src.evaluation.evaluate import ModelEvaluator
    from wingbeat_ml.models import MosSongPlusModel
    from wingbeat_ml.training import OptimizerFactory
    from wingbeat_ml.training import LossFactory
    from wingbeat_ml.training import CallbackFactory

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
        nomos_index=cfg["nomos_index"],
        labels_dict=cfg["labels"]
    )

    train_ds, val_ds, test_ds = ds_builder.build(
        split=cfg["dataset"]["split_list"],
        batch_size=cfg["train"]["batch_size"],
        shuffle=cfg["train"]["shuffle"]
    )

    # 5. Build Model (Let builder apply config overrides internally!)
    print("Building model...")
    model_builder = MosSongPlusModel(model_cfg, model_overrides=cfg.get("model"))
    model = model_builder.build(
        input_shape=(cfg["audio"]["segment_length"], 1),
        output_units=cfg["num_classes"],
        output_activation=cfg["model"]["output_activation"]
    )
    model.summary()

    # 6. Resolve Class Weights
    class_weights_enabled, class_weights = resolve_class_weights(
        cfg["class_weights"],
        ds_builder.class_weights,
        cfg["num_classes"],
        labels_dict=cfg["labels"]
    )

    if class_weights_enabled:
        print(f"Training class counts: {np.bincount(ds_builder.train_labels, minlength=cfg['num_classes']).tolist()}")
        print(f"Using class weights: {np.round(class_weights, 3).tolist()}")
        cfg["resolved_class_weights"] = class_weights.tolist()
    else:
        print("Class weights disabled.")
        cfg["resolved_class_weights"] = None

    # 7. Setup Optimizer and Loss
    optimizer = OptimizerFactory.get_optimizer(cfg)

    # Ensure consistency between activation and from_logits
    if cfg["model"]["output_activation"] is None:
        cfg["loss"]["from_logits"] = True
    elif cfg["model"]["output_activation"] == "softmax":
        cfg["loss"]["from_logits"] = False

    loss_fn = LossFactory.get_loss(cfg)

    # 8. Initialize Train and Evaluator
    print(f"Output activation: {cfg['model']['output_activation']}")
    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        train_ds,
        class_weights=class_weights if class_weights_enabled else None,
    )
    evaluator = ModelEvaluator(model, cfg["classes"], loss_fn)

    # 9. Setup Callbacks (val_x extraction for activation logging removed)
    callbacks = CallbackFactory.get_callbacks(cfg, optimizer, model, save_path)

    # 10. Training Loop
    epochs = cfg["train"]["epochs"]
    print(f"\nStarting training for {epochs} epochs...")

    import time
    for epoch in range(epochs):
        # --- Train ---
        start_time = time.time()
        train_metrics = trainer.train_epoch()
        epoch_time = time.time() - start_time

        # --- Eval ---
        val_metrics = evaluator.evaluate_epoch(val_ds)
        val_male_f1 = val_metrics.get("male_f1", 0.0)
        val_female_f1 = val_metrics.get("female_f1", 0.0)

        # Precision and Recall groups
        val_m_prec = val_metrics.get("male_prec", 0.0)
        val_m_rec = val_metrics.get("male_rec", 0.0)
        val_f_prec = val_metrics.get("female_prec", 0.0)
        val_f_rec = val_metrics.get("female_rec", 0.0)
        examples_per_second = train_metrics["examples"] / epoch_time

        # Print metrics with detailed P/R for both sexes
        print(f"Epoch {epoch+1}/{epochs} - loss: {train_metrics['loss']:.4f} - acc: {train_metrics['accuracy']:.4f} | "
              f"val_loss: {val_metrics['loss']:.4f} - val_acc: {val_metrics['accuracy']:.4f} | "
              f"val_f1: {val_metrics['macro_f1']:.3f} | "
              f"Female (P:{val_f_prec:.2f}, R:{val_f_rec:.2f}, F1:{val_female_f1:.2f}) | "
              f"Male (P:{val_m_prec:.2f}, R:{val_m_rec:.2f}, F1:{val_male_f1:.2f}) | "
              f"Time: {epoch_time:.2f}s | "
              f"Batches: {train_metrics['batches']} | "
              f"Examples: {train_metrics['examples']} | "
              f"Throughput: {examples_per_second:.0f} examples/s")

        # --- Manual Callback Execution ---
        callback_values = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "learning_rate": float(tf.keras.backend.get_value(optimizer.learning_rate)),
            "epoch_duration_seconds": epoch_time,
        }

        # Populate all val_metrics dynamically
        for k, v in val_metrics.items():
            key = f"val_{k}" if not k.startswith("val_") else k
            callback_values[key] = v

        # Checkpoint
        if 'model_checkpoint' in callbacks:
            cb = callbacks['model_checkpoint']
            if cb.save(model, callback_values):
                if hasattr(cb, "monitor") and hasattr(cb.monitor, "monitor"):
                    m_name = cb.monitor.monitor
                elif hasattr(cb, "monitors"):
                    m_name = cb.monitors[0]
                else:
                    m_name = "val_score"
                print(f"  --> Saved best weights to {save_path} ({m_name}={callback_values.get(m_name, 0):.4f})")

        # Reduce LR
        if 'reduce_lr_on_plateau' in callbacks:
            callbacks['reduce_lr_on_plateau'].on_epoch_end(callback_values)

        # Cosine Annealing
        if 'cosine_annealing' in callbacks:
            callbacks['cosine_annealing'].on_epoch_end(callback_values)

        # Wandb
        if 'wandb_logger' in callbacks:
            callbacks['wandb_logger'].on_epoch_end(callback_values)

        # Early Stopping
        if 'early_stopping' in callbacks:
            if callbacks['early_stopping'].check(callback_values):
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break

    # 11. Final Evaluation on Test Set
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
    from src.evaluation.report import report_results
    report_results(
        model=model,
        test_results=test_results,
        file_results=file_results,
        train_file_results=train_file_results,
        cfg=cfg,
        ds_builder=ds_builder,
        save_path=save_path,
        results_dir=results_dir,
        artifact_name='mossongplus-pretrained',
        val_ds=val_ds,
        test_ds=test_ds,
        evaluator=evaluator
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--defaults_path", type=str, default="configs/defaults.yaml")
    parser.add_argument("--model_cfg_path", type=str, default="configs/model.yaml")
    args, unknown = parser.parse_known_args()

    train_supervised(
        defaults_path=args.defaults_path,
        model_cfg_path=args.model_cfg_path
    )
