"""Canonical linear-probe pipeline."""

import os
from wingbeat_ml.config.runtime import (
    configure_training_runtime,
    generate_experiment_name,
    load_config,
    normalize_config,
    resolve_experiment_paths,
)
from wingbeat_ml.tracking import initialize_training_run


def train_linear_probe(defaults_path="configs/defaults.yaml",
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
    exp_name = generate_experiment_name(cfg, mode="LP")
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
    from wingbeat_ml.evaluation import ModelEvaluator
    from wingbeat_ml.registry import build_model
    from wingbeat_ml.training import LossFactory
    from wingbeat_ml.pipelines.evaluate import evaluate_training_run
    from wingbeat_ml.pipelines.train import (
        resolve_training_class_weights,
        run_training,
    )

    # 4. Setup Dataset
    print("Setting up datasets...")
    ds_builder, train_ds, val_ds, test_ds = build_datasets(
        cfg["dataset"].get("train_dir") or cfg["dataset"]["indoor"],
        cfg,
        val_dir=cfg["dataset"]["val_dir"],
        test_dir=cfg["dataset"]["test_dir"],
        return_builder=True,
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
        artifact_name="mossongplus-linearprobe",
        validation_dataset=val_ds,
        test_dataset=test_ds,
    )


if __name__ == "__main__":
    train_linear_probe()
