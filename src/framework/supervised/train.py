import os
import random
import numpy as np
from configs.mos_config import load_config, normalize_config, apply_reproducibility_environment, resolve_class_weights


def train_supervised(defaults_path="configs/defaults.yaml",
                     model_cfg_path="configs/model.yaml",
                     save_path="models/supervised_mossongplus/best_model.weights.h5",
                     results_dir="models/supervised_mossongplus/results"):

    # 1. Load and Normalize Configurations
    defaults_raw = load_config(defaults_path)
    cfg = normalize_config(defaults_raw)
    model_cfg = load_config(model_cfg_path)
    
    apply_reproducibility_environment(cfg["reproducibility"])

    import tensorflow as tf
    
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
    from model.mossongplus import MosSongPlusModel
    from src.framework.optimizer import OptimizerFactory
    from src.framework.loss import LossFactory
    from src.framework.callbacks import CallbackFactory

    # 2. Setup Dataset
    print("Setting up datasets...")
    ds_builder = SupervisedDataset(
        dataset_dir=cfg["dataset"]["indoor"],
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
        shuffle=cfg["train"]["shuffle"],
        step_ratio=cfg["train"]["step_ratio"]
    )

    # 3. Build Model
    print("Building model...")
    model_builder = MosSongPlusModel(model_cfg)
    model = model_builder.build(
        input_shape=(cfg["audio"]["segment_length"], 1),
        output_units=cfg["num_classes"],
        output_activation=cfg["model"]["output_activation"]
    )
    model.summary()

    # 4. Resolve Class Weights
    class_weights_enabled, class_weights = resolve_class_weights(
        cfg["class_weights"],
        ds_builder.class_weights,
        cfg["num_classes"],
    )

    if class_weights_enabled:
        print(f"Training class counts: {np.bincount(ds_builder.train_labels, minlength=cfg['num_classes']).tolist()}")
        print(f"Using class weights: {np.round(class_weights, 3).tolist()}")
        cfg["resolved_class_weights"] = class_weights.tolist()
    else:
        print("Class weights disabled.")
        cfg["resolved_class_weights"] = None

    # 5. Setup Optimizer and Loss
    optimizer = OptimizerFactory.get_optimizer(cfg)

    # Ensure consistency between activation and from_logits
    if cfg["model"]["output_activation"] is None:
        cfg["loss"]["from_logits"] = True
    elif cfg["model"]["output_activation"] == "softmax":
        cfg["loss"]["from_logits"] = False

    loss_fn = LossFactory.get_loss(cfg)

    # 6. Initialize Train and Evaluator
    print(f"Output activation: {cfg['model']['output_activation']}")
    trainer = Train(
        model,
        optimizer,
        loss_fn,
        train_ds,
        class_weights=class_weights if class_weights_enabled else None,
    )
    evaluator = ModelEvaluator(model, cfg["classes"], loss_fn)

    # 7. Setup Callbacks
    callbacks = CallbackFactory.get_callbacks(cfg, optimizer, model, save_path)

    # 8. Training Loop
    epochs = cfg["train"]["epochs"]
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
        if label in cfg["classes"]:
            print(f"Class {label:20} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1-score']:.4f}")

if __name__ == "__main__":
    train_supervised()
