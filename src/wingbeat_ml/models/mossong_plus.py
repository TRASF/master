from __future__ import annotations

from typing import Any

import tensorflow.keras as keras

from wingbeat_ml.models.layers import (
    RepConv1D,
    SincConv1D,
)


class MosSongPlusModel:
    """
    Config-driven MosquitoSong+ model builder.

    Supported trainable layers
    --------------------------
    Standard / built-in:
        - conv1d
        - separable_conv1d
        - depthwise_conv1d
        - dense

    Custom:
        - sincconv1d
        - repconv1d

    Supported utility layers
    ------------------------
        - maxpool1d
        - flatten
        - global_avg_pool
        - global_max_pool
        - dropout

    Notes
    -----
    Grouped convolution:
        type: conv1d
        groups: 4

    Pointwise convolution:
        type: conv1d
        kernel_size: 1

    Depthwise convolution:
        Does NOT accept `filters`.

        Output channels are:

            input_channels * depth_multiplier

        Use a pointwise Conv1D afterward when channel mixing or
        channel-count changes are required.

    BatchNorm policy:
        Standard Conv/Dense/Sinc layers use:

            Layer -> BatchNorm -> Activation

        RepConv1D is handled separately because it owns its
        internal Conv+BN branches.

    Bias policy:
        Bias is disabled automatically when an external
        BatchNormalization immediately follows the layer.
    """

    # ------------------------------------------------------------------
    # Generic trainable layers.
    #
    # RepConv1D is intentionally NOT placed here because it owns its own
    # internal branch normalization and activation behavior.
    # ------------------------------------------------------------------

    _TRAINABLE_LAYERS = {
        "conv1d": (
            keras.layers.Conv1D,
            ("kernel_regularizer",),
        ),
        "separable_conv1d": (
            keras.layers.SeparableConv1D,
            (
                "depthwise_regularizer",
                "pointwise_regularizer",
            ),
        ),
        "depthwise_conv1d": (
            keras.layers.DepthwiseConv1D,
            ("depthwise_regularizer",),
        ),
        "sincconv1d": (
            SincConv1D,
            (),
        ),
    }

    def __init__(
        self,
        model_config: Any,
        model_overrides: dict[str, Any] | None = None,
    ):
        from wingbeat_ml.config.schema import (
            AppConfig,
            ModelConfig,
        )

        # --------------------------------------------------------------
        # Normalize input configuration.
        # --------------------------------------------------------------

        if isinstance(model_config, AppConfig):
            model_dict = model_config.model.model_dump()

        elif isinstance(model_config, ModelConfig):
            model_dict = model_config.model_dump()

        elif isinstance(model_config, dict):
            model_dict = model_config

        else:
            raise TypeError(
                "model_config must be AppConfig, "
                "ModelConfig, or dict."
            )

        # --------------------------------------------------------------
        # Support:
        #
        # model:
        #   mossong_plus:
        #
        # and the legacy:
        #
        # mossongplus:
        # --------------------------------------------------------------

        model_section = (
            model_dict.get("model", {})
            if isinstance(
                model_dict.get("model"),
                dict,
            )
            else model_dict
        )

        self.model_cfg = (
            model_section.get("mossong_plus")
            or model_section.get("mossongplus")
            or model_section
        )

        self.overrides = (
            dict(model_overrides)
            if model_overrides
            else {}
        )

        if not isinstance(
            self.model_cfg,
            dict,
        ):
            raise ValueError(
                "Expected 'model.mossong_plus' "
                "or legacy 'model.mossongplus'."
            )

    # ==================================================================
    # Public API
    # ==================================================================

    def build(
        self,
        input_shape: tuple[int, ...],
        output_units: int,
        output_activation: str | None = "softmax",
        batch_size: int | None = None,
    ) -> keras.Model:
        """
        Construct a Functional Keras model from configuration.
        """

        if output_units <= 0:
            raise ValueError(
                "output_units must be > 0."
            )

        if batch_size is None:
            inputs = keras.layers.Input(
                shape=input_shape,
                name="input",
            )
        else:
            inputs = keras.layers.Input(
                batch_shape=(
                    batch_size,
                )
                + input_shape,
                name="input",
            )

        layers_config = self.model_cfg.get(
            "layers"
        )

        if not layers_config:
            raise ValueError(
                "Expected non-empty "
                "'layers' configuration."
            )

        x = self._build_sequential(
            inputs,
            layers_config,
        )

        # --------------------------------------------------------------
        # Keep output in float32 for mixed-precision stability.
        # --------------------------------------------------------------

        outputs = keras.layers.Dense(
            units=int(output_units),
            activation=output_activation,
            dtype="float32",
            name="output",
        )(x)

        return keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="MosquitoSongPlus",
        )

    # ==================================================================
    # Model construction
    # ==================================================================

    def _build_sequential(
        self,
        x,
        layers_config: list[dict[str, Any]],
    ):
        conv_idx = 1
        dense_idx = 1

        for layer_position, raw_def in enumerate(
            layers_config,
            start=1,
        ):
            if not isinstance(
                raw_def,
                dict,
            ):
                raise TypeError(
                    "Each layer definition must "
                    "be a dictionary. "
                    f"Layer {layer_position}: "
                    f"{raw_def!r}"
                )

            layer_def = dict(raw_def)

            layer_type = layer_def.pop(
                "type",
                None,
            )

            if not layer_type:
                raise ValueError(
                    "Layer definition missing "
                    f"'type': {raw_def}"
                )

            layer_type = str(
                layer_type
            ).lower()

            # ==========================================================
            # Generic convolutions
            # ==========================================================

            if layer_type in self._TRAINABLE_LAYERS:
                self._validate_trainable_layer(
                    layer_type=layer_type,
                    layer_def=layer_def,
                )

                layer_class, regularizer_keys = (
                    self._TRAINABLE_LAYERS[
                        layer_type
                    ]
                )

                x = self._add_trainable_layer(
                    x=x,
                    layer_class=layer_class,
                    layer_def=layer_def,
                    bn_override_key=(
                        f"bn_conv{conv_idx}"
                    ),
                    l2_value=self.overrides.get(
                        "conv_l2"
                    ),
                    regularizer_keys=(
                        regularizer_keys
                    ),
                )

                conv_idx += 1
                continue

            # ==========================================================
            # RepConv
            # ==========================================================

            if layer_type == "repconv1d":
                self._validate_repconv_layer(
                    layer_def
                )

                x = self._add_repconv_layer(
                    x=x,
                    layer_def=layer_def,
                    bn_override_key=(
                        f"bn_conv{conv_idx}"
                    ),
                )

                conv_idx += 1
                continue

            # ==========================================================
            # Dense
            # ==========================================================

            if layer_type == "dense":
                x = self._add_trainable_layer(
                    x=x,
                    layer_class=(
                        keras.layers.Dense
                    ),
                    layer_def=layer_def,
                    bn_override_key=(
                        f"bn_dense{dense_idx}"
                    ),
                    l2_value=self.overrides.get(
                        "dense_l2"
                    ),
                    regularizer_keys=(
                        "kernel_regularizer",
                    ),
                )

                dense_idx += 1
                continue

            # ==========================================================
            # Pooling
            # ==========================================================

            if layer_type == "maxpool1d":
                x = keras.layers.MaxPooling1D(
                    **layer_def
                )(x)

                continue

            # ==========================================================
            # Flatten
            # ==========================================================

            if layer_type == "flatten":
                self._reject_unexpected_config(
                    layer_type,
                    layer_def,
                )

                x = keras.layers.Flatten()(x)

                continue

            # ==========================================================
            # Global average pooling
            # ==========================================================

            if layer_type == "global_avg_pool":
                self._reject_unexpected_config(
                    layer_type,
                    layer_def,
                )

                x = (
                    keras.layers
                    .GlobalAveragePooling1D()
                )(x)

                continue

            # ==========================================================
            # Global max pooling
            # ==========================================================

            if layer_type == "global_max_pool":
                self._reject_unexpected_config(
                    layer_type,
                    layer_def,
                )

                x = (
                    keras.layers
                    .GlobalMaxPooling1D()
                )(x)

                continue

            # ==========================================================
            # Dropout
            # ==========================================================

            if layer_type == "dropout":
                unknown = set(
                    layer_def
                ) - {"rate"}

                if unknown:
                    raise ValueError(
                        "Unsupported dropout "
                        f"arguments: {sorted(unknown)}"
                    )

                rate = float(
                    layer_def.get(
                        "rate",
                        0.5,
                    )
                )

                if not 0.0 <= rate < 1.0:
                    raise ValueError(
                        "dropout rate must satisfy "
                        "0 <= rate < 1."
                    )

                x = keras.layers.Dropout(
                    rate=rate
                )(x)

                continue

            raise ValueError(
                "Unsupported layer type: "
                f"{layer_type!r}"
            )

        return x

    # ==================================================================
    # Generic Conv / Dense / Sinc builder
    # ==================================================================

    def _add_trainable_layer(
        self,
        *,
        x,
        layer_class,
        layer_def: dict[str, Any],
        bn_override_key: str,
        l2_value: float | None,
        regularizer_keys: tuple[str, ...],
    ):
        """
        Build a generic layer using:

            Trainable layer
                ->
            optional BatchNorm
                ->
            optional Activation
        """

        config = dict(layer_def)

        activation = config.pop(
            "activation",
            None,
        )

        configured_batch_norm = config.pop(
            "batch_norm",
            False,
        )

        # --------------------------------------------------------------
        # Legacy alias.
        # --------------------------------------------------------------

        if config.get(
            "padding"
        ) == "linear":
            config["padding"] = "valid"

        # --------------------------------------------------------------
        # Resolve BN runtime overrides.
        # --------------------------------------------------------------

        batch_norm = self._resolve_batch_norm(
            configured=(
                configured_batch_norm
            ),
            override_key=(
                bn_override_key
            ),
        )

        # --------------------------------------------------------------
        # Conv/Dense/Sinc bias is unnecessary when immediately followed
        # by BatchNormalization.
        # --------------------------------------------------------------

        if batch_norm:
            config.setdefault(
                "use_bias",
                False,
            )

        # --------------------------------------------------------------
        # Apply correct L2 argument for each layer family.
        # --------------------------------------------------------------

        self._apply_l2_regularization(
            config=config,
            l2_value=l2_value,
            regularizer_keys=(
                regularizer_keys
            ),
        )

        # --------------------------------------------------------------
        # Layer
        # --------------------------------------------------------------

        x = layer_class(
            **config
        )(x)

        # --------------------------------------------------------------
        # BatchNorm
        # --------------------------------------------------------------

        if batch_norm:
            bn_config = (
                dict(batch_norm)
                if isinstance(
                    batch_norm,
                    dict,
                )
                else {}
            )

            x = (
                keras.layers
                .BatchNormalization(
                    **bn_config
                )
            )(x)

        # --------------------------------------------------------------
        # Activation
        # --------------------------------------------------------------

        if activation:
            x = keras.layers.Activation(
                activation
            )(x)

        return x

    # ==================================================================
    # RepConv builder
    # ==================================================================

    def _add_repconv_layer(
        self,
        *,
        x,
        layer_def: dict[str, Any],
        bn_override_key: str,
    ):
        """
        Build RepConv1D.

        RepConv differs from ordinary layers because BatchNorm lives
        INSIDE each training branch and is fused into the deployment
        convolution later.

        Therefore we must not append an external BatchNormalization
        after RepConv1D.
        """

        config = dict(layer_def)

        # --------------------------------------------------------------
        # RepConv owns activation internally.
        # --------------------------------------------------------------

        activation = config.pop(
            "activation",
            "relu",
        )

        configured_batch_norm = config.pop(
            "batch_norm",
            True,
        )

        # --------------------------------------------------------------
        # Legacy alias.
        # --------------------------------------------------------------

        if config.get(
            "padding"
        ) == "linear":
            config["padding"] = "valid"

        batch_norm = self._resolve_batch_norm(
            configured=(
                configured_batch_norm
            ),
            override_key=(
                bn_override_key
            ),
        )

        use_batch_norm = bool(
            batch_norm
        )

        # --------------------------------------------------------------
        # Extract BN parameters that RepConv branches should use.
        # --------------------------------------------------------------

        bn_momentum = 0.99
        bn_epsilon = 1e-3

        if isinstance(
            batch_norm,
            dict,
        ):
            if "momentum" in batch_norm:
                bn_momentum = float(
                    batch_norm["momentum"]
                )

            if "epsilon" in batch_norm:
                bn_epsilon = float(
                    batch_norm["epsilon"]
                )

        elif self.overrides.get(
            "bn_momentum"
        ) is not None:
            bn_momentum = float(
                self.overrides[
                    "bn_momentum"
                ]
            )

        # --------------------------------------------------------------
        # L2.
        # --------------------------------------------------------------

        kernel_regularizer = None

        conv_l2 = self.overrides.get(
            "conv_l2"
        )

        if conv_l2 is not None:
            conv_l2 = float(
                conv_l2
            )

            if conv_l2 > 0:
                kernel_regularizer = (
                    keras.regularizers.l2(
                        conv_l2
                    )
                )

        # --------------------------------------------------------------
        # Explicit config values take precedence where useful.
        # --------------------------------------------------------------

        if (
            "bn_momentum"
            in config
        ):
            bn_momentum = float(
                config.pop(
                    "bn_momentum"
                )
            )

        if (
            "bn_epsilon"
            in config
        ):
            bn_epsilon = float(
                config.pop(
                    "bn_epsilon"
                )
            )

        if (
            "kernel_regularizer"
            in config
        ):
            kernel_regularizer = (
                config.pop(
                    "kernel_regularizer"
                )
            )

        # --------------------------------------------------------------
        # Build custom block.
        # --------------------------------------------------------------

        return RepConv1D(
            activation=activation,
            use_batch_norm=(
                use_batch_norm
            ),
            bn_momentum=(
                bn_momentum
            ),
            bn_epsilon=(
                bn_epsilon
            ),
            kernel_regularizer=(
                kernel_regularizer
            ),
            **config,
        )(x)

    # ==================================================================
    # Validation
    # ==================================================================

    @staticmethod
    def _validate_trainable_layer(
        *,
        layer_type: str,
        layer_def: dict[str, Any],
    ):
        """
        Catch common configuration mistakes before Keras construction.
        """

        # --------------------------------------------------------------
        # DepthwiseConv1D has no `filters`.
        # --------------------------------------------------------------

        if (
            layer_type
            == "depthwise_conv1d"
            and "filters"
            in layer_def
        ):
            raise ValueError(
                "DepthwiseConv1D does not accept "
                "'filters'. Its output channels are "
                "input_channels * depth_multiplier. "
                "Remove 'filters' and add a "
                "pointwise Conv1D(kernel_size=1) "
                "after the depthwise layer if you "
                "need to change channel count."
            )

        # --------------------------------------------------------------
        # Sinc frontend should generally operate on raw waveform.
        # Prevent accidental unsupported regularizer arguments.
        # --------------------------------------------------------------

        if layer_type == "sincconv1d":
            unsupported = {
                "kernel_regularizer",
                "depthwise_regularizer",
                "pointwise_regularizer",
            }.intersection(
                layer_def
            )

            if unsupported:
                raise ValueError(
                    "SincConv1D does not support "
                    "these regularizer arguments: "
                    f"{sorted(unsupported)}"
                )

        # --------------------------------------------------------------
        # Keras convolution families reject stride>1 + dilation>1.
        # Validate early for a clearer configuration error.
        # --------------------------------------------------------------

        if layer_type in {
            "conv1d",
            "separable_conv1d",
            "depthwise_conv1d",
            "sincconv1d",
        }:
            strides = layer_def.get(
                "strides",
                1,
            )

            dilation = layer_def.get(
                "dilation_rate",
                1,
            )

            if isinstance(
                strides,
                (tuple, list),
            ):
                stride_value = max(
                    int(v)
                    for v in strides
                )
            else:
                stride_value = int(
                    strides
                )

            if isinstance(
                dilation,
                (tuple, list),
            ):
                dilation_value = max(
                    int(v)
                    for v in dilation
                )
            else:
                dilation_value = int(
                    dilation
                )

            if (
                stride_value > 1
                and dilation_value > 1
            ):
                raise ValueError(
                    f"{layer_type}: strides > 1 "
                    "cannot be combined with "
                    "dilation_rate > 1."
                )

    @staticmethod
    def _validate_repconv_layer(
        layer_def: dict[str, Any],
    ):
        """
        RepConv-specific validation.
        """

        if "filters" not in layer_def:
            raise ValueError(
                "repconv1d requires 'filters'."
            )

        if "kernel_size" not in layer_def:
            raise ValueError(
                "repconv1d requires "
                "'kernel_size'."
            )

        branches = int(
            layer_def.get(
                "branches",
                2,
            )
        )

        if branches <= 0:
            raise ValueError(
                "repconv1d branches "
                "must be > 0."
            )

        strides = int(
            layer_def.get(
                "strides",
                1,
            )
        )

        dilation = int(
            layer_def.get(
                "dilation_rate",
                1,
            )
        )

        if (
            strides > 1
            and dilation > 1
        ):
            raise ValueError(
                "repconv1d: strides > 1 "
                "cannot be combined with "
                "dilation_rate > 1."
            )

    @staticmethod
    def _reject_unexpected_config(
        layer_type: str,
        layer_def: dict[str, Any],
    ):
        if layer_def:
            raise ValueError(
                f"{layer_type!r} does not "
                "accept configuration values: "
                f"{sorted(layer_def)}"
            )

    # ==================================================================
    # Regularization
    # ==================================================================

    @staticmethod
    def _apply_l2_regularization(
        *,
        config: dict[str, Any],
        l2_value: float | None,
        regularizer_keys: tuple[str, ...],
    ):
        """
        Apply layer-family-specific L2 regularization.

        Examples:

        Conv1D:
            kernel_regularizer

        SeparableConv1D:
            depthwise_regularizer
            pointwise_regularizer

        DepthwiseConv1D:
            depthwise_regularizer
        """

        if (
            l2_value is None
            or not regularizer_keys
        ):
            return

        value = float(
            l2_value
        )

        if value <= 0:
            return

        regularizer = (
            keras.regularizers.l2(
                value
            )
        )

        for key in regularizer_keys:
            config.setdefault(
                key,
                regularizer,
            )

    # ==================================================================
    # BatchNorm configuration
    # ==================================================================

    def _resolve_batch_norm(
        self,
        *,
        configured: bool | dict[str, Any],
        override_key: str,
    ) -> bool | dict[str, Any]:
        """
        Resolve per-layer BatchNorm state using:

        1. explicit runtime override;
        2. YAML configuration;
        3. global BN momentum override.
        """

        bn_momentum = self.overrides.get(
            "bn_momentum"
        )

        # --------------------------------------------------------------
        # Runtime override has highest priority.
        # --------------------------------------------------------------

        if (
            override_key
            in self.overrides
        ):
            enabled = bool(
                self.overrides[
                    override_key
                ]
            )

            if not enabled:
                return False

            if bn_momentum is None:
                return True

            return {
                "momentum": float(
                    bn_momentum
                )
            }

        # --------------------------------------------------------------
        # Disabled in configuration.
        # --------------------------------------------------------------

        if not configured:
            return False

        # --------------------------------------------------------------
        # Detailed BN configuration.
        # --------------------------------------------------------------

        if isinstance(
            configured,
            dict,
        ):
            config = dict(
                configured
            )

            if bn_momentum is not None:
                config["momentum"] = float(
                    bn_momentum
                )

            return config

        # --------------------------------------------------------------
        # Boolean True.
        # --------------------------------------------------------------

        if bn_momentum is not None:
            return {
                "momentum": float(
                    bn_momentum
                )
            }

        return True


__all__ = [
    "MosSongPlusModel",
]
