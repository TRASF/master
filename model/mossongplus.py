import tensorflow.keras as keras


class MosSongPlusModel:
    def __init__(self, model_config):
        self.model_config = model_config
        self.config = model_config.get("model", {}).get("mossongplus")

        if not self.config:
            raise ValueError(
                "Invalid model configuration. Ensure 'mossongplus' exists under 'model' in your YAML."
            )

    def build(self, input_shape, output_units, output_activation="softmax", batch_size=None):
        if batch_size is not None:
            inputs = keras.layers.Input(batch_shape=(batch_size,) + input_shape)
        else:
            inputs = keras.layers.Input(shape=input_shape)
        x = inputs

        # Support for sequential layer list
        if "layers" in self.config:
            x = self._build_sequential(x, self.config["layers"])
        else:
            x = self._build_legacy(x)

        if self.config.get("global_avg_pool", False):
            x = keras.layers.GlobalAveragePooling1D()(x)

        if self.config.get("global_max_pool", False):
            x = keras.layers.GlobalMaxPooling1D()(x)

        if self.config.get("flatten", False):
            x = keras.layers.Flatten()(x)

        # Output Layer
        x = keras.layers.Dense(
            units=output_units,
            activation=output_activation
        )(x)

        return keras.Model(inputs=inputs, outputs=x, name="MosquitoSongPlus")

    def _add_standard_layer(self, x, layer_class, layer_def):
        activation = layer_def.get("activation")
        use_batch_norm = layer_def.get("batch_norm", False)

        # Extract only relevant Keras kwargs dynamically by filtering out metadata
        keras_kwargs = {
            k: v for k, v in layer_def.items()
            if k not in ("type", "activation", "batch_norm")
        }

        # Resolve custom mapping details (e.g. padding='linear' -> 'valid')
        if keras_kwargs.get("padding") == "linear":
            keras_kwargs["padding"] = "valid"

        if use_batch_norm:
            # Flow: Layer -> BatchNormalization -> Activation
            x = layer_class(**keras_kwargs)(x)
            bn_cfg = use_batch_norm if isinstance(use_batch_norm, dict) else {}
            x = keras.layers.BatchNormalization(**bn_cfg)(x)
            if activation:
                x = keras.layers.Activation(activation)(x)
        else:
            # Flow: Layer with inline activation
            x = layer_class(activation=activation, **keras_kwargs)(x)

        return x

    def _build_sequential(self, x, layers_config):
        for layer_def in layers_config:
            layer_type = layer_def.get("type")

            if layer_type == "conv1d":
                x = self._add_standard_layer(x, keras.layers.Conv1D, layer_def)

            elif layer_type == "dense":
                x = self._add_standard_layer(x, keras.layers.Dense, layer_def)

            elif layer_type == "maxpool1d":
                cfg = {k: v for k, v in layer_def.items() if k != "type"}
                x = keras.layers.MaxPooling1D(**cfg)(x)

            elif layer_type == "flatten":
                x = keras.layers.Flatten()(x)

            elif layer_type == "global_avg_pool":
                x = keras.layers.GlobalAveragePooling1D()(x)

            elif layer_type == "global_max_pool":
                x = keras.layers.GlobalMaxPooling1D()(x)

            elif layer_type == "dropout":
                rate = layer_def.get("rate", 0.5)
                x = keras.layers.Dropout(rate)(x)

            else:
                 raise ValueError(f"Unsupported layer type: {layer_type}")
        return x

    def _build_legacy(self, x):
        for conv_cfg in self.config.get("conv", []):
            x = self._add_standard_layer(x, keras.layers.Conv1D, conv_cfg)

        for pool_cfg in self.config.get("maxpool", []):
            x = keras.layers.MaxPooling1D(**pool_cfg)(x)

        for d_cfg in self.config.get("dropout", []):
            rate = d_cfg.get("rate", 0.5) if isinstance(d_cfg, dict) else float(d_cfg)
            x = keras.layers.Dropout(rate)(x)

        for dense_cfg in self.config.get("dense", []):
            x = self._add_standard_layer(x, keras.layers.Dense, dense_cfg)

        return x
