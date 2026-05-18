import os
import yaml
import tensorflow as tf

from typing import List, Optional

class CallbackFactory:
    """
    Global factory for modular Keras callbacks.
    """
    
    @staticmethod
    def get_callbacks(config_path: str = "configs/defaults.yaml", model_save_path: str = "models/supervised_mossongplus") -> List[tf.keras.callbacks.Callback]:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        train_config = config.get('training', {})
        callback_config = train_config.get('callbacks', {})
        
        callbacks = []
        
        # 1. Early Stopping
        if callback_config.get('early_stopping', {}).get('enabled', True):
            es_params = callback_config.get('early_stopping', {
                'monitor': 'val_loss',
                'patience': 10,
                'restore_best_weights': True
            })
            # Remove 'enabled' key before passing to Keras
            es_params.pop('enabled', None)
            callbacks.append(tf.keras.callbacks.EarlyStopping(**es_params))
            
        # 2. Model Checkpoint
        if callback_config.get('checkpoint', {}).get('enabled', True):
            cp_params = callback_config.get('checkpoint', {
                'monitor': 'val_accuracy',
                'save_best_only': True
            })
            cp_params.pop('enabled', None)
            
            # Ensure filepath is set
            if 'filepath' not in cp_params:
                cp_params['filepath'] = os.path.join(model_save_path, "checkpoint.keras")
                
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(**cp_params))
            
        # 3. TensorBoard (Disabled by default if not installed, or can be enabled via config)
        if callback_config.get('tensorboard', {}).get('enabled', False):
            tb_params = callback_config.get('tensorboard', {})
            tb_params.pop('enabled', None)
            
            if 'log_dir' not in tb_params:
                tb_params['log_dir'] = os.path.join(model_save_path, "logs")
                
            try:
                callbacks.append(tf.keras.callbacks.TensorBoard(**tb_params))
            except Exception as e:
                print(f"Warning: Could not initialize TensorBoard callback: {e}")

        # 4. CSV Logger
        if callback_config.get('csv_logger', {}).get('enabled', True):
            csv_params = callback_config.get('csv_logger', {})
            csv_params.pop('enabled', None)
            
            if 'filename' not in csv_params:
                csv_params['filename'] = os.path.join(model_save_path, "training_log.csv")
                
            callbacks.append(tf.keras.callbacks.CSVLogger(**csv_params))

        # 5. Reduce LR on Plateau
        if callback_config.get('reduce_lr', {}).get('enabled', True):
            lr_params = callback_config.get('reduce_lr', {
                'monitor': 'val_loss',
                'factor': 0.2,
                'patience': 15,
                'min_lr': 0.00001
            })
            lr_params.pop('enabled', None)
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(**lr_params))

        return callbacks
