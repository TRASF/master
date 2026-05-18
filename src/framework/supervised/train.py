import os
import yaml
import argparse
import numpy as np
import tensorflow as tf
from model.mossongplus import MosSongPlusModel
from src.framework.supervised.dataset import SupervisedDataset
from src.framework.optimizer import OptimizerFactory
from src.framework.loss import LossFactory
from src.framework.callbacks import CallbackFactory

def main():
    parser = argparse.ArgumentParser(description="Train the MosSongPlus supervised model.")
    parser.add_argument("--config", type=str, default="configs/defaults.yaml", help="Path to default config.")
    parser.add_argument("--model_config", type=str, default="configs/model.yaml", help="Path to model config.")
    parser.add_argument("--save_dir", type=str, default="models/supervised_mossongplus", help="Directory to save models.")
    parser.add_argument("--datasets", type=str, default='MSB/indoor', help="Comma-separated list of datasets to use (e.g., 'MSB/indoor,Philip/lab'). If None, uses everything configured.")
    args = parser.parse_args()

    # Load configurations
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Instantiate the model
    print("Building model...")
    model_factory = MosSongPlusModel(model_yaml_path=args.model_config, defaults_yaml_path=args.config)
    model = model_factory.build()
    
    # Compile the model
    print("Compiling model...")
    optimizer = OptimizerFactory.get_optimizer(args.config)
    loss = LossFactory.get_loss(args.config)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.summary()

    # Create dataset manager
    dataset_manager = SupervisedDataset(config_path=args.config)

    # Determine paths and dataset splits
    batch_size = config.get('training', {}).get('batch_size', 32)
    dataset_root = config.get('dataset', {}).get('root', 'dataset')
    
    dataset_selection = args.datasets.split(',') if args.datasets else None
    
    # Create a unique dataset path if subsetting
    tf_dataset_name = 'tf_supervised'
    if dataset_selection:
        suffix = '_'.join([s.replace('/', '_') for s in dataset_selection])
        tf_dataset_name += f"_{suffix}"
        
    tf_dataset_path = os.path.join(dataset_root, tf_dataset_name)
    
    if not os.path.exists(tf_dataset_path):
        print(f"Generating dataset at {tf_dataset_path}...")
        dataset_manager.generate_and_save(
            raw_data_path=dataset_root, 
            save_path=tf_dataset_path,
            dataset_selection=dataset_selection
        )

    print("Loading datasets...")
    train_ds = dataset_manager.load_for_training(tf_dataset_path, split='train', batch_size=batch_size, shuffle=True)
    val_ds = dataset_manager.load_for_training(tf_dataset_path, split='val', batch_size=batch_size, shuffle=False)

    # Calculate class weights if enabled
    class_weight = None
    if config.get('training', {}).get('use_class_weights', False):
        print("Calculating balanced class weights...")
        raw_samples = dataset_manager._get_raw_samples(dataset_selection)
        all_labels = [s[1] for s in raw_samples]
        
        from sklearn.utils.class_weight import compute_class_weight
        unique_labels = np.unique(all_labels)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_labels,
            y=all_labels
        )
        class_weight = dict(zip(unique_labels, weights))
        print(f"Computed weights for {len(unique_labels)} classes.")
        # Map back to labels if any classes are missing in current subset
        # Keras expects weights for all labels in class_dict (0 to num_classes-1)
        full_class_weight = {i: 1.0 for i in range(len(config['class_dict']))}
        for label, weight in class_weight.items():
            full_class_weight[int(label)] = float(weight)
        class_weight = full_class_weight

    # Get callbacks
    print("Setting up callbacks...")
    os.makedirs(args.save_dir, exist_ok=True)
    callbacks = CallbackFactory.get_callbacks(config_path=args.config, model_save_path=args.save_dir)

    # Start training
    epochs = config.get('training', {}).get('epochs', 100)
    print(f"Starting training for {epochs} epochs...")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight
    )

    # Save the final model
    final_model_path = os.path.join(args.save_dir, "final_model.keras")
    model.save(final_model_path)
    print(f"Training completed. Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()
