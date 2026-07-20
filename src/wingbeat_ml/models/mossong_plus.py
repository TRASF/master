import tensorflow.keras as keras


class MosSongPlusModel:
    def __init__(self, model_config, model_overrides=None):
        self.model_config = model_config
        model_section = model_config.get("model", {})
        self.config = (
            model_section.get("mossong_plus")
            or model_section.get("mossongplus")
        )
        self.overrides = model_overrides or {}

        if not self.config:
            raise ValueError(
                "Invalid model configuration: expected "
                "'model.mossong_plus' or legacy 'model.mossongplus'."
            )

    def build(self, input_shape, output_units, output_activation="softmax", batch_size=None):
        if batch_size is not None:
            inputs = keras.layers.Input(batch_shape=(batch_size,) + input_shape)
        else:
            inputs = keras.layers.Input(shape=input_shape)
        x = inputs

        # Build sequential layer list
        if "layers" not in self.config:
            raise ValueError("Expected 'layers' in model configuration.")

        x = self._build_sequential(x, self.config["layers"])

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
        bn_momentum = self.overrides.get("bn_momentum")
        dense_l2 = self.overrides.get("dense_l2")
        conv_l2 = self.overrides.get("conv_l2")

        conv_idx = 1
        dense_idx = 1

        for layer_def in layers_config:
            layer_type = layer_def.get("type")
            layer_def = dict(layer_def)  # Copy to avoid modifying model_config dict in-place

            if layer_type == "conv1d":
                opt_key = f"bn_conv{conv_idx}"
                if opt_key in self.overrides:
                    enabled = bool(self.overrides[opt_key])
                    if enabled and bn_momentum is not None:
                        layer_def["batch_norm"] = {"momentum": float(bn_momentum)}
                    else:
                        layer_def["batch_norm"] = enabled
                elif layer_def.get("batch_norm") and bn_momentum is not None:
                    layer_def["batch_norm"] = {"momentum": float(bn_momentum)}

                if conv_l2 is not None and float(conv_l2) > 0:
                    layer_def["kernel_regularizer"] = keras.regularizers.l2(float(conv_l2))

                x = self._add_standard_layer(x, keras.layers.Conv1D, layer_def)
                conv_idx += 1

            elif layer_type == "dense":
                opt_key = f"bn_dense{dense_idx}"
                if opt_key in self.overrides:
                    enabled = bool(self.overrides[opt_key])
                    if enabled and bn_momentum is not None:
                        layer_def["batch_norm"] = {"momentum": float(bn_momentum)}
                    else:
                        layer_def["batch_norm"] = enabled
                elif layer_def.get("batch_norm") and bn_momentum is not None:
                    layer_def["batch_norm"] = {"momentum": float(bn_momentum)}

                if dense_l2 is not None and float(dense_l2) > 0:
                    layer_def["kernel_regularizer"] = keras.regularizers.l2(float(dense_l2))

                x = self._add_standard_layer(x, keras.layers.Dense, layer_def)
                dense_idx += 1

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


__all__ = ["MosSongPlusModel"]
