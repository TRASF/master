import os
import random
import yaml
import numpy as np

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def configure_reproducibility_environment(defaults):
    cfg = defaults.get("reproducibility", {})
    if isinstance(cfg, bool):
        cfg = {"enabled": cfg}

    enabled = bool(cfg.get("enabled", False))
    default_seed = defaults.get("train", {}).get("seed", defaults.get("seed", 42))
    seed = int(cfg.get("seed", default_seed))
    deterministic_ops = bool(cfg.get("deterministic_ops", enabled))
    deterministic_data = bool(cfg.get("deterministic_data", enabled))

    if enabled:
        os.environ["PYTHONHASHSEED"] = str(seed)
        if deterministic_ops:
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
            os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    return {
        "enabled": enabled,
        "seed": seed,
        "deterministic_ops": deterministic_ops,
        "deterministic_data": deterministic_data,
    }


def apply_reproducibility(settings, tf):
    if not settings["enabled"]:
        print("Reproducibility disabled.")
        return

    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    if settings["deterministic_ops"]:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as exc:
            print(f"Warning: TensorFlow op determinism could not be enabled: {exc}")

    print(
        "Reproducibility enabled "
        f"(seed={seed}, deterministic_data={settings['deterministic_data']}, "
        f"deterministic_ops={settings['deterministic_ops']})."
    )


def resolve_class_weights(config_weights, fallback_weights, num_classes):
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

def train_supervised(defaults_path="configs/defaults.yaml",
                     model_cfg_path="configs/model.yaml",
                     save_path="models/supervised_mossongplus/best_model.weights.h5",
                     results_dir="models/supervised_mossongplus/results"):

    # 1. Load Configurations
    defaults = load_config(defaults_path)
    model_cfg = load_config(model_cfg_path)
    reproducibility = configure_reproducibility_environment(defaults)

    import tensorflow as tf
    from src.framework.supervised.dataset import SupervisedDataset
    from src.framework.supervised.train_step import Train
    from src.evaluation.evaluate import ModelEvaluator
    from model.mossongplus import MosSongPlusModel
    from src.framework.optimizer import OptimizerFactory
    from src.framework.loss import LossFactory
    from src.framework.callbacks import CallbackFactory

    apply_reproducibility(reproducibility, tf)

    # Extract sub-configs
    audio_cfg = defaults.get("audio", {})
    train_cfg = defaults.get("train", {})
    dataset_cfg = defaults.get("dataset", {})
    augment_cfg = defaults.get("augment", {})
    model_defaults = defaults.get("model", {})

    # Strictly respect the config file (null in YAML = None in Python)
    output_activation = model_defaults.get("output_activation")

    configured_class_weights = defaults.get("class_weights")
    classes = list(defaults["labels"].keys())

    seed = reproducibility["seed"]
    deterministic_data = reproducibility["enabled"] and reproducibility["deterministic_data"]

    # Pre-calculate derived values
    sample_rate = audio_cfg.get("sample_rate", 8000)
    duration = audio_cfg.get("duration", 0.3)
    segment_length = int(duration * sample_rate)

    # 2. Setup Dataset
    print("Setting up datasets...")
    dataset_dir = dataset_cfg.get("indoor", "dataset/MSB/Indoor")
    val_dir = dataset_cfg.get("val_dir")
    test_dir = dataset_cfg.get("test_dir")
    noise_dirs = augment_cfg.get("noise_banks", [])

    ds_builder = SupervisedDataset(
        dataset_dir=dataset_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        sample_rate=sample_rate,
        segment_length=segment_length,
        classes=classes,
        noise_dirs=noise_dirs,
        augment_cfg=augment_cfg,
        seed=seed,
        deterministic=deterministic_data,
    )

    batch_size = train_cfg.get("batch_size", 32)
    split_ratios = dataset_cfg.get("split_ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
    split_list = [split_ratios.get("train", 0.8), split_ratios.get("val", 0.1), split_ratios.get("test", 0.1)]

    train_ds, val_ds, test_ds = ds_builder.build(
        split=split_list,
        batch_size=batch_size,
        shuffle=train_cfg.get("shuffle", True),
        step_ratio=train_cfg.get("step_ratio", 0.5)
    )

    # 3. Build Model
    print("Building model...")
    num_classes = len(defaults["labels"])
    model_builder = MosSongPlusModel(model_cfg)
    model = model_builder.build(
        input_shape=(segment_length, 1),
        output_units=num_classes,
        output_activation=output_activation
    )
    model.summary()

    # 4. Resolve Class Weights
    class_weights_enabled, class_weights = resolve_class_weights(
        configured_class_weights,
        ds_builder.class_weights,
        num_classes,
    )

    if class_weights_enabled:
        print(f"Training class counts: {np.bincount(ds_builder.train_labels, minlength=num_classes).tolist()}")
        print(f"Using class weights: {np.round(class_weights, 3).tolist()}")
        defaults["resolved_class_weights"] = class_weights.tolist()
    else:
        print("Class weights disabled.")
        defaults["resolved_class_weights"] = None

    # 5. Setup Optimizer and Loss
    optimizer = OptimizerFactory.get_optimizer(defaults)

    # Ensure consistency between activation and from_logits
    if output_activation is None:
        defaults["loss"]["from_logits"] = True
    elif output_activation == "softmax":
        defaults["loss"]["from_logits"] = False

    loss_fn = LossFactory.get_loss(defaults)

    # 6. Initialize Train and Evaluator
    print(f"Output activation: {output_activation}")
    trainer = Train(
        model,
        optimizer,
        loss_fn,
        train_ds,
        class_weights=class_weights if class_weights_enabled else None,
    )
    evaluator = ModelEvaluator(model, classes, loss_fn)

    # 7. Setup Callbacks
    callbacks = CallbackFactory.get_callbacks(defaults, optimizer, model, save_path)

    # 8. Training Loop
    epochs = defaults["train"]["epochs"]
    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(epochs):
        # --- Train ---
        train_metrics = trainer.train_epoch()

        # --- Eval ---
        val_metrics = evaluator.evaluate_epoch(val_ds)
        val_male_f1 = val_metrics.get("male_f1", 0.0)
        val_female_f1 = val_metrics.get("female_f1", 0.0)

        # New: Precision and Recall groups
        val_m_prec = val_metrics.get("male_prec", 0.0)
        val_m_rec = val_metrics.get("male_rec", 0.0)
        val_f_prec = val_metrics.get("female_prec", 0.0)
        val_f_rec = val_metrics.get("female_rec", 0.0)

        # Print metrics with detailed P/R for both sexes
        print(f"Epoch {epoch+1}/{epochs} - loss: {train_metrics['loss']:.4f} - acc: {train_metrics['accuracy']:.4f} | "
              f"val_loss: {val_metrics['loss']:.4f} - val_acc: {val_metrics['accuracy']:.4f} | "
              f"val_f1: {val_metrics['macro_f1']:.3f} | "
              f"Female (P:{val_f_prec:.2f}, R:{val_f_rec:.2f}, F1:{val_female_f1:.2f}) | "
              f"Male (P:{val_m_prec:.2f}, R:{val_m_rec:.2f}, F1:{val_male_f1:.2f})")

        # --- Manual Callback Execution ---
        callback_values = {
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_male_f1": val_male_f1,
            "val_female_f1": val_female_f1,
        }

        # Checkpoint
        if 'model_checkpoint' in callbacks:
            cb = callbacks['model_checkpoint']
            if cb.save(model, callback_values):
                # Robust monitor name retrieval
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

        # Early Stopping
        if 'early_stopping' in callbacks:
            if callbacks['early_stopping'].check(callback_values):
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break

    # 9. Final Evaluation on Test Set
    print("\nTraining complete. Running final evaluation on test set...")
    if os.path.exists(save_path):
        model.load_weights(save_path)
    test_results = evaluator.evaluate_final_test(test_ds, save_dir=results_dir)
    print(f"Final Test Accuracy: {test_results['metrics']['accuracy']:.4f}")
    print(f"Final Test Macro F1: {test_results['metrics']['macro_f1']:.4f}")
    print("Confusion Matrix:")
    print(np.array(test_results["confusion_matrix"]))

    print("\nClassification Report:")
    for label, metrics in test_results["report"].items():
        if label in classes:
            print(f"Class {label:20} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1-score']:.4f}")

if __name__ == "__main__":
    train_supervised()
