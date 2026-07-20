"""Canonical pretraining pipeline."""

from wingbeat_ml.config.runtime import (
    configure_training_runtime,
    generate_experiment_name,
    load_config,
    normalize_config,
    resolve_experiment_paths,
)
from wingbeat_ml.tracking import initialize_training_run


def train_supervised(defaults_path="configs/defaults.yaml",
                     model_cfg_path="configs/model.yaml",
                     save_path=None,
                     results_dir=None):

    # 1. Load and Normalize Configurations
    defaults_raw = load_config(defaults_path)
    cfg = normalize_config(defaults_raw)
    model_cfg = load_config(model_cfg_path)

    # 2. Start optional tracking and apply sweep overrides.
    wandb_run = initialize_training_run(cfg)

    # 3. Dynamic Experiment Naming & Path Resolution (Run once!)
    base_exp_name = generate_experiment_name(cfg, mode="Pretrain")
    if wandb_run is not None:
        hpf_p = cfg.get("augment", {}).get("high_pass", {}).get("p", 0.0)
        seed = cfg.get("reproducibility", {}).get(
            "seed",
            cfg.get("train", {}).get("seed", "seed"),
        )
        task = (
            cfg.get("wandb", {}).get("group")
            or f"{cfg.get('num_classes', 'n')}class"
        )
        exp_name = f"{task}_{base_exp_name}_hpf{hpf_p}_seed{seed}"
        wandb_run.name = exp_name
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

    # 5. Build Model (Let builder apply config overrides internally!)
    print("Building model...")
    model = build_model(cfg, model_cfg)
    model.summary()

    # 6. Resolve Class Weights
    class_weights = resolve_training_class_weights(
        cfg,
        ds_builder,
        show_counts=True,
    )

    # 7. Setup Loss and Evaluation
    if cfg["model"]["output_activation"] is None:
        cfg["loss"]["from_logits"] = True
    elif cfg["model"]["output_activation"] == "softmax":
        cfg["loss"]["from_logits"] = False

    loss_fn = LossFactory.get_loss(cfg)
    print(f"Output activation: {cfg['model']['output_activation']}")
    evaluator = ModelEvaluator(model, cfg["classes"], loss_fn)

    # 8. Run the shared epoch loop.
    epochs = cfg["train"]["epochs"]
    print(f"\nStarting training for {epochs} epochs...")

    def print_epoch(epoch, logs):
        duration = logs["epoch_duration_seconds"]
        examples = logs.get("train_examples", 0)
        throughput = examples / duration if duration else 0.0
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"loss: {logs['train_loss']:.4f} - "
            f"acc: {logs['train_accuracy']:.4f} | "
            f"val_loss: {logs['val_loss']:.4f} - "
            f"val_acc: {logs['val_accuracy']:.4f} | "
            f"val_f1: {logs['val_macro_f1']:.3f} | "
            f"Female (P:{logs.get('val_female_prec', 0.0):.2f}, "
            f"R:{logs.get('val_female_rec', 0.0):.2f}, "
            f"F1:{logs.get('val_female_f1', 0.0):.2f}) | "
            f"Male (P:{logs.get('val_male_prec', 0.0):.2f}, "
            f"R:{logs.get('val_male_rec', 0.0):.2f}, "
            f"F1:{logs.get('val_male_f1', 0.0):.2f}) | "
            f"Time: {duration:.2f}s | "
            f"Batches: {logs.get('train_batches', 0)} | "
            f"Examples: {examples} | "
            f"Throughput: {throughput:.0f} examples/s"
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
        artifact_name="mossongplus-pretrained",
        validation_dataset=val_ds,
        test_dataset=test_ds,
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
