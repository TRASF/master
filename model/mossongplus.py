import tensorflow as tf

class MosSongPlusModel:
    def __init__(self, model_config):
        self.model_config = model_config
        self.config = model_config.get("model", {}).get("mossongplus")
        if not self.config:
            raise ValueError("Invalid model configuration. Ensure 'mossongplus' exists under 'model' in your YAML.")

    def build(self, input_shape, output_units, output_activation=None):

        inputs = tf.keras.layers.Input(shape=input_shape)
        x = inputs

        # 1. Convolutional Layers
        for conv_cfg in self.config.get("conv", []):
            x = tf.keras.layers.Conv1D(**conv_cfg)(x)

        # 2. Max Pooling Layers
        for pool_cfg in self.config.get("maxpool", []):
            x = tf.keras.layers.MaxPooling1D(**pool_cfg)(x)

        # 3. Flatten
        flatten_cfg = self.config.get("flatten")
        # Handle YAML list [True]
        should_flatten = flatten_cfg[0] if isinstance(flatten_cfg, list) and flatten_cfg else flatten_cfg
        if should_flatten:
            x = tf.keras.layers.Flatten()(x)

        # 4. Dropout
        dropout_cfgs = self.config.get("dropout", [])
        for d_cfg in dropout_cfgs:
            rate = d_cfg.get("rate", 0.5) if isinstance(d_cfg, dict) else d_cfg
            x = tf.keras.layers.Dropout(rate=rate)(x)

        # 5. Dense Layers
        for dense_cfg in self.config.get("dense", []):
            x = tf.keras.layers.Dense(**dense_cfg)(x)

        # 6. Final Output Layer (e.g., for classification)
        if output_units > 0:
            x = tf.keras.layers.Dense(units=output_units, activation=output_activation)(x)

        return tf.keras.Model(inputs=inputs, outputs=x, name="MosSongPlus")