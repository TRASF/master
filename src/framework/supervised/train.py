import os
import yaml
import tensorflow as tf
import numpy as np
from src.framework.supervised.dataset import SupervisedDataset
from src.framework.supervised.train_step import Train
from src.evaluation.evaluate import ModelEvaluator
from model.mossongplus import MosSongPlusModel

# Import Factories
from src.framework.optimizer import OptimizerFactory
from src.framework.loss import LossFactory
from src.framework.callbacks import CallbackFactory

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_class_weights(config_weights, fallback_weights, num_classes):
    if config_weights is None:
        return fallback_weights

    if isinstance(config_weights, dict):
        return np.array([float(config_weights.get(i, config_weights.get(str(i), 1.0))) for i in range(num_classes)], dtype=np.float32)

    if len(config_weights) != num_classes:
        raise ValueError(f"class_weights must contain {num_classes} values, got {len(config_weights)}")

    return np.array(config_weights, dtype=np.float32)

def train_supervised(defaults_path="configs/defaults.yaml", model_cfg_path="configs/model.yaml", save_path="models/supervised_mossongplus/best_model.weights.h5", results_dir="models/supervised_mossongplus/results"):
    # 1. Load Configurations
    defaults = load_config(defaults_path)
    model_cfg = load_config(model_cfg_path)
    
    # Extract sub-configs for cleaner access
    audio_cfg = defaults.get("audio", {})
    train_cfg = defaults.get("train", {})
    dataset_cfg = defaults.get("dataset", {})
    augment_cfg = defaults.get("augment", {})
    model_defaults = defaults.get("model", {})
    output_activation = model_defaults.get("output_activation", None)
    configured_class_weights = defaults.get("class_weights")
    classes = list(defaults["labels"].keys())
    
    # Pre-calculate derived values
    sample_rate = audio_cfg.get("sample_rate", 8000)
    duration = audio_cfg.get("duration", 0.3)
    segment_length = int(duration * sample_rate)
    
    # 2. Setup Dataset
    print("Setting up datasets...")
    dataset_dir = dataset_cfg.get("indoor", "dataset/MSB/Indoor")
    val_dir = dataset_cfg.get("val_dir") # Dedicated val directory
    test_dir = dataset_cfg.get("test_dir") # Dedicated test directory
    noise_dirs = augment_cfg.get("noise_banks", [])
    
    ds_builder = SupervisedDataset(
        dataset_dir=dataset_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        sample_rate=sample_rate,
        segment_length=segment_length,
        classes=classes,
        noise_dirs=noise_dirs,
        augment_cfg=augment_cfg
    )
    
    batch_size = train_cfg.get("batch_size", 32)
    step_ratio = train_cfg.get("step_ratio", 0.5)
    split_ratios = dataset_cfg.get("split_ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
    split_list = [split_ratios.get("train", 0.8), split_ratios.get("val", 0.1), split_ratios.get("test", 0.1)]

    train_ds, val_ds, test_ds = ds_builder.build(
        split=split_list,
        batch_size=batch_size,
        shuffle=train_cfg.get("shuffle", True),
        step_ratio=step_ratio
    )
    
    # Add Channel Dimension for Conv1D
    train_ds = train_ds.map(lambda x, y: (tf.expand_dims(x, -1), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (tf.expand_dims(x, -1), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (tf.expand_dims(x, -1), y), num_parallel_calls=tf.data.AUTOTUNE)

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
    class_weights = resolve_class_weights(configured_class_weights, ds_builder.class_weights, num_classes)
    print(f"Using class weights: {np.round(class_weights, 3).tolist()}")
    defaults['class_weights'] = class_weights.tolist()

    # 5. Setup Optimizer and Loss using Factories
    optimizer = OptimizerFactory.get_optimizer(defaults)
    loss_fn = LossFactory.get_loss(defaults)

    # 6. Initialize Train and Evaluator
    print(f"Output activation: {output_activation}")
    trainer = Train(model, optimizer, loss_fn, train_ds)
    evaluator = ModelEvaluator(model, classes, loss_fn)

    # 7. Setup Callbacks using Factory
    callbacks = CallbackFactory.get_callbacks(defaults, optimizer, save_path)

    # 8. Training Loop
    epochs = defaults["train"]["epochs"]
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # --- Train ---
        train_metrics = trainer.train_epoch()
        
        # --- Eval ---
        val_metrics = evaluator.evaluate_epoch(val_ds)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{epochs} - "
              f"loss: {train_metrics['loss']:.4f} - acc: {train_metrics['accuracy']:.4f} - "
              f"val_loss: {val_metrics['loss']:.4f} - val_acc: {val_metrics['accuracy']:.4f} - "
              f"val_f1: {val_metrics['macro_f1']:.4f}")

        # --- Manual Callback Execution ---
        callback_values = {
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }

        # Checkpoint
        if 'model_checkpoint' in callbacks:
            cb = callbacks['model_checkpoint']
            if cb.save(model, callback_values):
                monitor_info = ", ".join(
                    f"{m}={callback_values[m]:.4f}" for m in cb.monitors
                )
                print(f"  --> Saved best weights to {save_path} ({monitor_info})")
        
        # Reduce LR
        if 'reduce_lr_on_plateau' in callbacks:
            cb = callbacks['reduce_lr_on_plateau']
            cb.on_epoch_end(callback_values[cb.monitor])
            
        # Early Stopping
        if 'early_stopping' in callbacks:
            cb = callbacks['early_stopping']
            if cb.check(callback_values[cb.monitor]):
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break

    # 8. Final Evaluation on Test Set
    print("\nTraining complete. Running final evaluation on test set...")
    if os.path.exists(save_path):
        model.load_weights(save_path)
    test_results = evaluator.evaluate_final_test(test_ds, save_dir=results_dir)
    print(f"Final Test Accuracy: {test_results['metrics']['accuracy']:.4f}")
    print(f"Final Test Macro F1: {test_results['metrics']['macro_f1']:.4f}")
    print("Confusion Matrix:")
    print(np.array(test_results["confusion_matrix"]))

    print("Classification Report:")
    for label, metrics in test_results["report"].items():
        if label in classes:
            print(f"Class {label} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1-score']:.4f}")

if __name__ == "__main__":
    train_supervised()
