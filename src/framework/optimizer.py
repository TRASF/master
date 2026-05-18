import tensorflow as tf

import yaml

class OptimizerFactory:
    """Global factory for modular optimizers."""
    
    @staticmethod
    def get_optimizer(config_path: str = "configs/defaults.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get the training config, fallback to defaults if missing
        train_config = config.get('training', {})
        opt_config = train_config.get('optimizer', {'name': 'Adam', 'learning_rate': 0.003})
        
        # Extract the class name
        opt_name = opt_config.pop('name')
        
        # Dynamically fetch the class from tf.keras.optimizers (e.g., tf.keras.optimizers.Adam)
        OptimizerClass = getattr(tf.keras.optimizers, opt_name)
        
        # Instantiate with remaining arguments (e.g., learning_rate, momentum)
        return OptimizerClass(**opt_config)
