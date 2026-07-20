"""Canonical pretraining pipeline."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path

from wingbeat_ml.config import (
    load_config as load_layered_config,
    write_resolved_config,
)
from wingbeat_ml.config.runtime import (
    configure_training_runtime,
    generate_experiment_name,
    load_config,
    normalize_config,
    resolve_experiment_paths,
)
from wingbeat_ml.tracking import initialize_training_run


DEFAULT_RUNTIME_ROOT = Path(
    "/media/miru4090s/New Volume2/wingbeat_ml"
)


def _find_project_root(start=None):
    """Find a source checkout containing the canonical configuration."""
    starting_path = Path(start or Path.cwd()).resolve()
    candidates = (starting_path, *starting_path.parents)
    source_root = Path(__file__).resolve().parents[3]

    for candidate in (*candidates, source_root):
        if (
            (candidate / "configs" / "base.yaml").is_file()
            and (
                candidate
                / "configs"
                / "models"
                / "mossong_plus.yaml"
            ).is_file()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not find configs/base.yaml. Run the bare command "
        "from a MosSongPlus source checkout or provide explicit paths."
    )


def prepare_default_pilot(project_root=None, runtime_root=None):
    """Resolve the safe five-epoch configuration for a bare invocation."""
    root = _find_project_root(project_root)
    dataset = (root / "dataset" / "MSB" / "Indoor").resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"Pilot dataset not found: {dataset}")

    runtime = Path(
        runtime_root
        or os.environ.get("WINGBEAT_RUNTIME_ROOT")
        or DEFAULT_RUNTIME_ROOT
    ).resolve()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    execution_root = runtime / "pilots" / timestamp
    config_dir = execution_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    profile_path = config_dir / "profile.yaml"
    profile_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "train_dir": str(dataset),
                    "val_dir": None,
                    "test_dir": None,
                },
                "train": {"epochs": 5, "batch_size": 256},
                "augment": {"noise_overlay": {"p": 0.0}},
                "wandb": {"enabled": False},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    model_path = root / "configs" / "models" / "mossong_plus.yaml"
    resolved = load_layered_config(
        base_path=str(root / "configs" / "base.yaml"),
        model_path=str(model_path),
        experiment_path=str(
            root / "configs" / "experiments" / "pretrain.yaml"
        ),
        profile_path=str(profile_path),
    )
    resolved_path = config_dir / "resolved.yaml"
    write_resolved_config(resolved, str(resolved_path))

    print(f"Zero-argument pilot config: {resolved_path}")
    print(f"Pilot run directory: {execution_root}")
    return resolved_path, model_path, execution_root


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


def main(args=None):
    """Run pretraining, selecting the local pilot when no paths are given."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--defaults_path", type=str)
    parser.add_argument("--model_cfg_path", type=str)
    parsed_args, _ = parser.parse_known_args(args)

    if (
        parsed_args.defaults_path is None
        and parsed_args.model_cfg_path is None
    ):
        defaults_path, model_cfg_path, runtime_root = (
            prepare_default_pilot()
        )
        os.environ["WINGBEAT_RUNTIME_ROOT"] = str(runtime_root)
        os.chdir(runtime_root)
    else:
        defaults_path = (
            parsed_args.defaults_path or "configs/defaults.yaml"
        )
        model_cfg_path = (
            parsed_args.model_cfg_path or "configs/model.yaml"
        )

    train_supervised(
        defaults_path=defaults_path,
        model_cfg_path=model_cfg_path,
    )


if __name__ == "__main__":
    main()
