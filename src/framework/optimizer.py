import tensorflow as tf
import keras

class OptimizerFactory:
    @staticmethod
    def get_optimizer(config: dict):
        """
        Retrieves the optimizer based on the 'optimizer' section of the config.
        """
        opt_config = config.get('optimizer')
        opt_params = opt_config.copy()
        
        # Extract the name to find the class, the rest are kwargs
        opt_name = opt_params.pop('name')
        
        # Map common names if necessary (e.g., AdamW is in keras.optimizers)
        try:
            OptimizerClass = getattr(keras.optimizers, opt_name)
            return OptimizerClass(**opt_params)
        except AttributeError:
            raise ValueError(f"Optimizer '{opt_name}' not found in keras.optimizers")
