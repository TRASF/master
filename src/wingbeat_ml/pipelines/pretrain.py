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
        class_weights=(class_weights if class_weights_enabled else None),
        save_path=save_path,
    )

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
