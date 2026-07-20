import os
import random
import numpy as np
from configs.mos_config import load_config, normalize_config, apply_reproducibility_environment, resolve_class_weights, generate_experiment_name, resolve_experiment_paths
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

    from src.framework.supervised.dataset import SupervisedDataset
    from src.framework.supervised.train_step import Train
    from src.evaluation.evaluate import ModelEvaluator
    from wingbeat_ml.models import MosSongPlusModel
    from src.framework.optimizer import OptimizerFactory
    from src.framework.loss import LossFactory
    from src.framework.callbacks import CallbackFactory

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

    # -------------------------------------------------------------
    # PHASE 1: LINEAR PROBING
    # -------------------------------------------------------------
    print(f"\n--- Linear Probing (Training Only Dense Head) ---")
    for layer in model.layers[:-1]:
        layer.trainable = False
    model.layers[-1].trainable = True

    optimizer = OptimizerFactory.get_optimizer(cfg)
    trainer = Train(model, optimizer, loss_fn, train_ds, class_weights=class_weights if class_weights_enabled else None)
    callbacks = CallbackFactory.get_callbacks(cfg, optimizer, model, save_path)

    print(f"Starting linear probe training for {epochs} epochs...")

    import time
    for epoch in range(epochs):
        # --- Train ---
        start_time = time.time()
        train_metrics = trainer.train_epoch()
        epoch_time = time.time() - start_time

        # --- Eval ---
        val_metrics = evaluator.evaluate_epoch(val_ds)

        print(f"Epoch {epoch+1}/{epochs} - loss: {train_metrics['loss']:.4f} - acc: {train_metrics['accuracy']:.4f} | "
              f"val_loss: {val_metrics['loss']:.4f} - val_acc: {val_metrics['accuracy']:.4f} | "
              f"val_f1: {val_metrics['macro_f1']:.3f} | "
              f"Time: {epoch_time:.1f}s")

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
                print(f"  --> Saved best weights to {save_path} (val_macro_f1={callback_values['val_macro_f1']:.4f})")

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
        artifact_name='mossongplus-linearprobe',
        val_ds=val_ds,
        test_ds=test_ds,
        evaluator=evaluator
    )


if __name__ == "__main__":
    train_linear_probe()
