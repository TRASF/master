from __future__ import annotations

from typing import Any

import tensorflow.keras as keras

from wingbeat_ml.classification.models.layers import (
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
            )
        else:
            inputs = keras.layers.Input(
                batch_shape=(
                    batch_size,
                )
                + input_shape,
            )

        layers_config = self.model_cfg.get("layers")

        if layers_config is None:
            raise ValueError(
                "Expected 'layers' configuration."
            )

        if layers_config:
            x = self._build_sequential(
                inputs,
                layers_config,
            )
        else:
            x = inputs

        if len(x.shape) > 2:
            if x.shape[-1] == 1:
                x = keras.layers.Reshape((-1,))(x)
            else:
                x = keras.layers.Flatten()(x)

        # --------------------------------------------------------------
        # Keep output in float32 for mixed-precision stability.
        # --------------------------------------------------------------

        outputs = keras.layers.Dense(
            units=int(output_units),
            activation=output_activation,
            dtype="float32",
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
        import inspect
        from wingbeat_ml.config.schema import parse_layer_config
        from wingbeat_ml.classification.models.registry import LAYER_REGISTRY

        context = {
            "overrides": self.overrides,
            "conv_idx": 1,
            "dense_idx": 1,
        }

        for layer_position, raw_def in enumerate(layers_config, start=1):
            layer_config = parse_layer_config(raw_def)
            factory = LAYER_REGISTRY.get(layer_config.type)

            if callable(factory):
                sig = inspect.signature(factory)
                if len(sig.parameters) == 3:
                    x = factory(x, layer_config, context)
                elif len(sig.parameters) == 2:
                    x = factory(x, layer_config)
                else:
                    spec = layer_config.model_dump() if hasattr(layer_config, "model_dump") else dict(layer_config)
                    spec.pop("type", None)
                    x = factory(**spec)(x)
            else:
                spec = layer_config.model_dump() if hasattr(layer_config, "model_dump") else dict(layer_config)
                spec.pop("type", None)
                x = factory(**spec)(x)

        return x

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
            override_key in self.overrides
            and self.overrides[override_key] is not None
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
