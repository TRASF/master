from __future__ import annotations

from typing import Any

import tensorflow.keras as keras

from wingbeat_ml.models.layers import RepConv1D, SincConv1D
from wingbeat_ml.models.initializers import (
    FIRBandpassInitializer,
)

# ----------------------------------------------------------------------
# Extension points
#
# Adding a normal/custom layer should usually require only:
#   1. import the class
#   2. add one entry here
#
# RepConv1D stays special because its BatchNorm lives inside the block.
# ----------------------------------------------------------------------

LAYER_TYPES = {
    "conv1d": keras.layers.Conv1D,
    "separable_conv1d": keras.layers.SeparableConv1D,
    "depthwise_conv1d": keras.layers.DepthwiseConv1D,
    "dense": keras.layers.Dense,
    "sincconv1d": SincConv1D,
    "maxpool1d": keras.layers.MaxPooling1D,
    "avgpool1d": keras.layers.AveragePooling1D,
    "flatten": keras.layers.Flatten,
    "global_avg_pool": keras.layers.GlobalAveragePooling1D,
    "global_max_pool": keras.layers.GlobalMaxPooling1D,
    "dropout": keras.layers.Dropout,
    "leaky_relu": keras.layers.LeakyReLU,
    "leakyrelu": keras.layers.LeakyReLU,
    "relu": keras.layers.ReLU,
}

TRAINABLE_TYPES = {
    "conv1d",
    "separable_conv1d",
    "depthwise_conv1d",
    "dense",
    "sincconv1d",
}

REGULARIZER_KEYS = {
    "conv1d": ("kernel_regularizer",),
    "dense": ("kernel_regularizer",),
    "separable_conv1d": (
        "depthwise_regularizer",
        "pointwise_regularizer",
    ),
    "depthwise_conv1d": ("depthwise_regularizer",),
    "sincconv1d": (),
}

INITIALIZER_TYPES = {
    "fir_bandpass": FIRBandpassInitializer,
}

class MosSongPlusModel:
    """
    Small config-driven 1D model builder.

    The YAML defines the architecture. Python only handles:
      - layer lookup
      - optional BatchNorm placement
      - project-wide L2 overrides
      - optional custom initializers
      - RepConv1D's special training/deployment structure

    To add another ordinary layer:
        add it to LAYER_TYPES.

    To add another trainable layer:
        also add it to TRAINABLE_TYPES and REGULARIZER_KEYS.

    To add another sequential architecture:
        normally only change the YAML.
    """

    def __init__(
        self,
        model_config: Any,
        model_overrides: dict[str, Any] | None = None,
    ):
        self.model_cfg = self._get_model_config(model_config)
        self.overrides = dict(model_overrides or {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        input_shape: tuple[int, ...],
        output_units: int,
        output_activation: str | None = "softmax",
        batch_size: int | None = None,
    ) -> keras.Model:
        if output_units <= 0:
            raise ValueError("output_units must be > 0")

        if batch_size is None:
            inputs = keras.layers.Input(
                shape=input_shape,
                name="input",
            )
        else:
            inputs = keras.layers.Input(
                batch_shape=(batch_size, *input_shape),
                name="input",
            )

        layers = self.model_cfg.get("layers")
        if not layers:
            raise ValueError("Expected a non-empty 'layers' configuration")

        x = inputs
        conv_index = 0
        dense_index = 0

        for position, raw_spec in enumerate(layers, start=1):
            spec = dict(raw_spec)
            layer_type = str(spec.pop("type", "")).lower()

            if not layer_type:
                raise ValueError(f"Layer {position} is missing 'type'")

            if spec.get("padding") == "linear":
                spec["padding"] = "valid"

            if layer_type == "repconv1d":
                conv_index += 1
                x = self._build_repconv(
                    x,
                    spec,
                    bn_key=f"bn_conv{conv_index}",
                )
                continue

            layer_class = LAYER_TYPES.get(layer_type)
            if layer_class is None:
                raise ValueError(
                    f"Unsupported layer type {layer_type!r}. "
                    f"Available: {sorted((*LAYER_TYPES, 'repconv1d'))}"
                )

            if layer_type in TRAINABLE_TYPES:
                if layer_type == "dense":
                    dense_index += 1
                    bn_key = f"bn_dense{dense_index}"
                    l2_value = self.overrides.get("dense_l2")
                else:
                    conv_index += 1
                    bn_key = f"bn_conv{conv_index}"
                    l2_value = self.overrides.get("conv_l2")

                x = self._build_trainable(
                    x,
                    layer_type,
                    layer_class,
                    spec,
                    bn_key=bn_key,
                    l2_value=l2_value,
                )
            else:
                # Keras already validates arguments such as pool size,
                # dropout rate, etc., so do not duplicate that logic here.
                x = layer_class(**spec)(x)

        outputs = keras.layers.Dense(
            int(output_units),
            activation=output_activation,
            dtype="float32",
            name="output",
        )(x)

        return keras.Model(
            inputs,
            outputs,
            name="MosquitoSongPlus",
        )

    # ------------------------------------------------------------------
    # Normal Conv / Dense / Sinc path
    # ------------------------------------------------------------------

    def _build_trainable(
        self,
        x,
        layer_type: str,
        layer_class,
        spec: dict[str, Any],
        *,
        bn_key: str,
        l2_value: float | None,
    ):
        spec = dict(spec)

        activation = spec.pop("activation", None)
        configured_bn = spec.pop("batch_norm", False)
        bn_config = self._batch_norm_config(configured_bn, bn_key)

        self._apply_initializer(spec)
        self._apply_l2(
            spec,
            REGULARIZER_KEYS[layer_type],
            l2_value,
        )

        if bn_config is not None:
            # Existing project convention: BatchNorm follows the layer.
            spec.setdefault("use_bias", False)

        x = layer_class(**spec)(x)

        if bn_config is not None:
            x = keras.layers.BatchNormalization(**bn_config)(x)

        if activation:
            x = keras.layers.Activation(activation)(x)

        return x

    # ------------------------------------------------------------------
    # RepConv1D is the one special case
    # ------------------------------------------------------------------

    def _build_repconv(
        self,
        x,
        spec: dict[str, Any],
        *,
        bn_key: str,
    ):
        spec = dict(spec)

        activation = spec.pop("activation", "relu")
        configured_bn = spec.pop("batch_norm", True)
        bn_config = self._batch_norm_config(configured_bn, bn_key)

        bn_momentum = float(spec.pop("bn_momentum", 0.99))
        bn_epsilon = float(spec.pop("bn_epsilon", 1e-3))

        if bn_config:
            bn_momentum = float(
                bn_config.get("momentum", bn_momentum)
            )
            bn_epsilon = float(
                bn_config.get("epsilon", bn_epsilon)
            )

        kernel_regularizer = spec.pop("kernel_regularizer", None)

        conv_l2 = self.overrides.get("conv_l2")
        if kernel_regularizer is None and conv_l2 is not None:
            value = float(conv_l2)
            if value > 0:
                kernel_regularizer = keras.regularizers.l2(value)

        return RepConv1D(
            activation=activation,
            use_batch_norm=bn_config is not None,
            bn_momentum=bn_momentum,
            bn_epsilon=bn_epsilon,
            kernel_regularizer=kernel_regularizer,
            **spec,
        )(x)

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _batch_norm_config(
        self,
        configured: bool | dict[str, Any],
        override_key: str,
    ) -> dict[str, Any] | None:
        override = self.overrides.get(override_key)
        if override is not None:
            enabled = bool(override)
        else:
            enabled = bool(configured)

        if not enabled:
            return None

        config = dict(configured) if isinstance(configured, dict) else {}

        if self.overrides.get("bn_momentum") is not None:
            config["momentum"] = float(self.overrides["bn_momentum"])

        return config

    @staticmethod
    def _apply_l2(
        spec: dict[str, Any],
        keys: tuple[str, ...],
        l2_value: float | None,
    ) -> None:
        if l2_value is None or not keys:
            return

        value = float(l2_value)
        if value <= 0:
            return

        regularizer = keras.regularizers.l2(value)

        for key in keys:
            spec.setdefault(key, regularizer)

    @staticmethod
    def _apply_initializer(spec: dict[str, Any]) -> None:
        """
        Supports both normal Keras initializers:

            kernel_initializer: he_normal

        and small project-local initializer specs:

            kernel_initializer:
              type: fir_bandpass
              sample_rate: 8000
              min_freq: 300
              max_freq: 3800
        """
        initializer = spec.get("kernel_initializer")

        if not isinstance(initializer, dict):
            return

        config = dict(initializer)
        initializer_type = str(config.pop("type", "")).lower()

        initializer_class = INITIALIZER_TYPES.get(initializer_type)
        if initializer_class is None:
            raise ValueError(
                f"Unknown custom initializer {initializer_type!r}. "
                f"Available: {sorted(INITIALIZER_TYPES)}"
            )

        spec["kernel_initializer"] = initializer_class(**config)

    @staticmethod
    def _get_model_config(model_config: Any) -> dict[str, Any]:
        """
        Accept AppConfig, ModelConfig, or a plain dict without coupling the
        builder to those classes at import time.
        """
        if hasattr(model_config, "model_dump"):
            model_dict = model_config.model_dump()
        elif isinstance(model_config, dict):
            model_dict = model_config
        else:
            raise TypeError(
                "model_config must be a Pydantic config model or dict"
            )

        model_section = model_dict.get("model", model_dict)

        if not isinstance(model_section, dict):
            raise ValueError("'model' must be a mapping")

        config = (
            model_section.get("mossong_plus")
            or model_section.get("mossongplus")
            or model_section
        )

        if not isinstance(config, dict):
            raise ValueError("MosSongPlus configuration must be a mapping")

        return config


__all__ = ["MosSongPlusModel"]
