"""Canonical fine-tuning pipeline."""

import os
from wingbeat_ml.config.runtime import (
    configure_training_runtime,
    generate_experiment_name,
    load_config,
    normalize_config,
    resolve_experiment_paths,
)
from wingbeat_ml.tracking import initialize_training_run


def train_finetune(defaults_path="configs/defaults.yaml",
                   model_cfg_path="configs/model.yaml",
                   pretrained_weights="models/supervised_mossongplus/best_model.weights.h5",
                   save_path=None,
                   results_dir=None):

    # 1. Load and Normalize Configurations
    defaults_raw = load_config(defaults_path)
    cfg = normalize_config(defaults_raw)
    model_cfg = load_config(model_cfg_path)

    # 2. Start optional tracking and apply sweep overrides.
    wandb_run = initialize_training_run(cfg)

    # 3. Dynamic Experiment Naming & Path Resolution (Run once!)
    exp_name = generate_experiment_name(cfg, mode="FT")
    if wandb_run is not None:
        wandb_run.name = exp_name

    resolved_paths = resolve_experiment_paths(cfg, exp_name)
    if save_path is None:
        save_path = resolved_paths["save_path"]
    if results_dir is None:
        results_dir = resolved_paths["results_dir"]

    print(f"Experiment Name: {exp_name}")
    print(f"Saving weights to: {save_path}")
    print(f"Saving results to: {results_dir}")

    # Ensure classification output activation is softmax if not set
    if cfg["model"].get("output_activation") is None:
        cfg["model"]["output_activation"] = "softmax"

    # Ensure consistency between activation and from_logits
    if cfg["model"]["output_activation"] == "softmax":
        cfg["loss"]["from_logits"] = False
    else:
        cfg["loss"]["from_logits"] = True

    configure_training_runtime(cfg["reproducibility"])

    from wingbeat_ml.data.dataset import build_datasets
    from wingbeat_ml.training import Trainer
    from wingbeat_ml.evaluation import ModelEvaluator
    from wingbeat_ml.registry import build_model
    from wingbeat_ml.training import OptimizerFactory
    from wingbeat_ml.training import LossFactory
    from wingbeat_ml.pipelines.evaluate import evaluate_training_run
    from wingbeat_ml.pipelines.train import (
        resolve_training_class_weights,
        run_training,
    )

    # 4. Setup Dataset
    print("Setting up datasets...")
    dataset_dir = (
        cfg["dataset"].get("train_dir")
        or cfg["dataset"]["indoor"]
    )
    ds_builder, train_ds, val_ds, test_ds = build_datasets(
        dataset_dir,
        cfg,
        val_dir=cfg["dataset"]["val_dir"],
        test_dir=cfg["dataset"]["test_dir"],
        return_builder=True,
    )

    # Warmup dataset builder (with high augmentation probability)
    import copy
    warmup_augment_cfg = copy.deepcopy(cfg["augment"])
    warmup_p = cfg["train"].get("warmup_augment_p", 1.0)

    # Force warmup augmentation probabilities for active augmentations
    for key in ["noise_overlay", "random_gain"]:
        if key in warmup_augment_cfg and isinstance(warmup_augment_cfg[key], dict):
            if key == "noise_overlay" and not warmup_augment_cfg.get("noise_banks"):
                continue
            warmup_augment_cfg[key]["p"] = warmup_p

    print(f"Setting up warmup dataset (forcing augmentation p={warmup_p})...")
    warmup_config = copy.deepcopy(cfg)
    warmup_config["augment"] = warmup_augment_cfg
    train_ds_warmup, _, _ = build_datasets(
        dataset_dir,
        warmup_config,
        val_dir=cfg["dataset"]["val_dir"],
        test_dir=cfg["dataset"]["test_dir"],
    )

    # 5. Build Model
    print("Building model...")
    model = build_model(cfg, model_cfg)

    # LOAD PRETRAINED WEIGHTS
    if os.path.exists(pretrained_weights):
        print(f"Loading pre-trained contrastive weights from {pretrained_weights}...")
        model.load_weights(pretrained_weights)
    else:
        print(f"WARNING: Pre-trained weights not found at {pretrained_weights}! Training from scratch.")

    model.summary()

    # 6. Resolve Class Weights
    class_weights = resolve_training_class_weights(cfg, ds_builder)

    # 7. Setup Optimizer and Loss
    loss_fn = LossFactory.get_loss(cfg)
    evaluator = ModelEvaluator(model, cfg["classes"], loss_fn)

    epochs = cfg["train"]["epochs"]
    warmup_epochs = cfg["train"].get("warmup_epochs", 15)

    # -------------------------------------------------------------
    # PHASE 1: LINEAR PROBING (Warmup Dense Head)
    # -------------------------------------------------------------
    print(f"\n--- PHASE 1: Warming up Dense Head for {warmup_epochs} epochs ---")
    for layer in model.layers[:-1]:
        layer.trainable = False
    model.layers[-1].trainable = True

    optimizer_phase1 = OptimizerFactory.get_optimizer(cfg)
    trainer_phase1 = Trainer(
        model,
        optimizer_phase1,
        loss_fn,
        train_ds_warmup,
        class_weights=class_weights,
    )

    for epoch in range(warmup_epochs):
        train_metrics = trainer_phase1.train_epoch()
        val_metrics = evaluator.evaluate_epoch(val_ds)
        print(f"Warmup Epoch {epoch+1}/{warmup_epochs} - loss: {train_metrics['loss']:.4f} - acc: {train_metrics['accuracy']:.4f} | "
              f"val_loss: {val_metrics['loss']:.4f} - val_acc: {val_metrics['accuracy']:.4f}")

        if cfg.get("wandb", {}).get("enabled", False):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "warmup_epoch": epoch,
                        "warmup_train_loss": train_metrics["loss"],
                        "warmup_train_accuracy": train_metrics["accuracy"],
                        "warmup_val_loss": val_metrics["loss"],
                        "warmup_val_accuracy": val_metrics["accuracy"]
                    })
            except ImportError:
                pass

    # -------------------------------------------------------------
    # PHASE 2: FULL FINE-TUNING
    # -------------------------------------------------------------
    print("\n--- PHASE 2: Full Fine-Tuning ---")
    cfg["training_mode"] = "fine_tune"
    cfg["optimizer"]["learning_rate"] = 1e-3
    print(f"Starting full fine-tuning for {epochs} epochs...")

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
        class_weights=class_weights,
        save_path=save_path,
    )

    evaluate_training_run(
        model=model,
        evaluator=evaluator,
        dataset_builder=ds_builder,
        config=cfg,
        checkpoint_path=save_path,
        results_dir=results_dir,
        artifact_name="mossongplus-finetuned",
        validation_dataset=val_ds,
        test_dataset=test_ds,
    )


if __name__ == "__main__":
    train_finetune()
