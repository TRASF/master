import tensorflow as tf


def build_optimizer(config=None):
    """Build one Keras optimizer from its configuration section."""
    optimizer_config = dict(config or {})
    name = optimizer_config.pop("name", "Adam")
    return tf.keras.optimizers.get(
        {"class_name": name, "config": optimizer_config}
    )


__all__ = ["build_optimizer"]
