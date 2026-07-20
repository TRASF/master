import tensorflow as tf

class OptimizerFactory:
    @staticmethod
    def get_optimizer(config: dict):
        opt_cfg = config.get('optimizer', {}).copy()
        name = opt_cfg.pop('name', 'Adam')
        return tf.keras.optimizers.get({"class_name": name, "config": opt_cfg})


__all__ = ["OptimizerFactory"]
