import tensorflow as tf


class MosSongPlusModel:
    def __init__(self, model_config):
        self.model_config = model_config
        self.config = model_config.get("model", {}).get("mossongplus")

        if not self.config:
            raise ValueError(
                "Invalid model configuration. Ensure 'mossongplus' exists under 'model' in your YAML."
            )

    def build(self, input_shape, output_units, output_activation="softmax"):
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = inputs

        for conv_cfg in self.config.get("conv", []):
            cfg = conv_cfg.copy()
            activation = cfg.pop("activation", None)
            use_batch_norm = cfg.pop("batch_norm", False)

            if cfg.get("padding") == "linear":
                cfg["padding"] = "valid"

            if not use_batch_norm and activation:
                # OPTIMIZATION: Embed activation directly if no BN is present
                cfg["activation"] = activation
                x = tf.keras.layers.Conv1D(**cfg)(x)
            else:
                # Standard sequence: Conv (Linear) -> BN -> Activation
                x = tf.keras.layers.Conv1D(**cfg)(x)
                if use_batch_norm:
                    bn_cfg = use_batch_norm if isinstance(use_batch_norm, dict) else {}
                    x = tf.keras.layers.BatchNormalization(**bn_cfg)(x)
                if activation:
                    x = tf.keras.layers.Activation(activation)(x)

        for pool_cfg in self.config.get("maxpool", []):
            x = tf.keras.layers.MaxPooling1D(**pool_cfg)(x)

        if self.config.get("global_avg_pool", False):
            x = tf.keras.layers.GlobalAveragePooling1D()(x)

        if self.config.get("global_max_pool", False):
            x = tf.keras.layers.GlobalMaxPooling1D()(x)

        if self.config.get("flatten", False):
            x = tf.keras.layers.Flatten()(x)

        for d_cfg in self.config.get("dropout", []):
            rate = d_cfg.get("rate", 0.5) if isinstance(d_cfg, dict) else float(d_cfg)
            x = tf.keras.layers.Dropout(rate)(x)


        for dense_cfg in self.config.get("dense", []):
            cfg = dense_cfg.copy()
            activation = cfg.pop("activation", None)
            use_batch_norm = cfg.pop("batch_norm", False)
            if not use_batch_norm and activation:
                # OPTIMIZATION: Embed activation directly if no BN is present
                cfg["activation"] = activation
                x = tf.keras.layers.Dense(**cfg)(x)
            else:
                # Standard sequence: Dense (Linear) -> BN -> Activation
                x = tf.keras.layers.Dense(**cfg)(x)
                if use_batch_norm:
                    bn_cfg = use_batch_norm if isinstance(use_batch_norm, dict) else {}
                    x = tf.keras.layers.BatchNormalization(**bn_cfg)(x)
                if activation:
                    x = tf.keras.layers.Activation(activation)(x)


        # 6. Output Layer
        x = tf.keras.layers.Dense(
            units=output_units,
            activation=output_activation,
        )(x)

        return tf.keras.Model(inputs=inputs, outputs=x, name="MosquitoSongPlus")
