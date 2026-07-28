import tensorflow as tf


def build_optimizer(config=None):
    """Build one Keras optimizer from its configuration section."""
    from wingbeat_ml.config.schema import AppConfig, OptimizerConfig

    if isinstance(config, AppConfig):
        opt_obj = config.optimizer
    elif isinstance(config, OptimizerConfig):
        opt_obj = config
    else:
        opt_obj = config

    if isinstance(opt_obj, (OptimizerConfig, AppConfig)):
        optimizer_config = opt_obj.model_dump() if hasattr(opt_obj, "model_dump") else opt_obj.optimizer.model_dump()
    elif isinstance(opt_obj, dict):
        optimizer_config = dict(opt_obj)
    else:
        optimizer_config = {}

    name = optimizer_config.pop("name", "Adam")
    return tf.keras.optimizers.get(
        {"class_name": name, "config": optimizer_config}
    )


__all__ = ["build_optimizer"]
