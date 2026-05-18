import tensorflow as tf

import yaml

class LossFactory:
    """Global factory for modular loss functions."""
    
    @staticmethod
    def get_loss(config_path: str = "configs/defaults.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Get the training config, fallback to defaults if missing
        train_config = config.get('training', {})
        loss_config = train_config.get('loss', {'name': 'SparseCategoricalCrossentropy', 'from_logits': False})
        
        # Extract the class name
        loss_name = loss_config.pop('name')
        
        # Dynamically fetch the class from tf.keras.losses
        LossClass = getattr(tf.keras.losses, loss_name)
        
        # Instantiate with remaining arguments
        return LossClass(**loss_config)
