"""Model and Layer registry definitions."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict
import tensorflow.keras as keras

from wingbeat_ml.registry import Registry, MODEL_BUILDERS
LAYER_REGISTRY = Registry[Any]("layer")
from wingbeat_ml.models.layers import RepConv1D, SincConv1D
from wingbeat_ml.config.schema import (
    Conv1DLayerConfig,
    DepthwiseConv1DLayerConfig,
    SeparableConv1DLayerConfig,
    SincConv1DLayerConfig,
    RepConv1DLayerConfig,
    DenseLayerConfig,
    FlattenLayerConfig,
    GlobalAvgPoolLayerConfig,
    GlobalMaxPoolLayerConfig,
    MaxPool1DLayerConfig,
    AvgPool1DLayerConfig,
    DropoutLayerConfig,
    ReLULayerConfig,
    ActivationLayerConfig,
    BatchNormLayerConfig,
    ConcatLayerConfig,
)


def register_layer(name: str):
    """Decorator to register a custom layer factory or class."""
    return LAYER_REGISTRY.register(name)


def register_model_builder(name: str):
    """Decorator to register a custom model builder."""
    return MODEL_BUILDERS.register(name)


# Register standard model builder
from wingbeat_ml.models.mossong_plus import MosSongPlusModel
MODEL_BUILDERS.register("mossong_plus", MosSongPlusModel)


# ------------------------------------------------------------------
# Layer Construction Helpers
# ------------------------------------------------------------------

def resolve_initializer(spec: dict[str, Any]) -> None:
    initializer = spec.get("kernel_initializer")
    if isinstance(initializer, dict):
        config = dict(initializer)
        init_type = str(config.pop("type", "")).lower()
        if init_type == "fir_bandpass":
            from wingbeat_ml.models.initializers.firbandpass import FIRBandpassInitializer
            spec["kernel_initializer"] = FIRBandpassInitializer(**config)
        else:
            raise ValueError(f"Unknown custom initializer {init_type!r}")


def resolve_batch_norm(
    configured: bool | dict[str, Any],
    override_key: str,
    overrides: dict[str, Any],
) -> dict[str, Any] | None:
    if override_key in overrides and overrides[override_key] is not None:
        enabled = bool(overrides.get(override_key))
    else:
        enabled = bool(configured)
    if not enabled:
        return None
    config = dict(configured) if isinstance(configured, dict) else {}
    bn_mom = overrides.get("bn_momentum")
    if bn_mom is not None:
        config["momentum"] = float(bn_mom)
    return config


def apply_l2(
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


def _clean_spec(cfg: Any) -> dict[str, Any]:
    spec = cfg.model_dump() if hasattr(cfg, "model_dump") else dict(cfg)
    spec.pop("type", None)
    kwargs = spec.pop("kwargs", {})
    if isinstance(kwargs, dict):
        spec.update(kwargs)
    return spec


# ------------------------------------------------------------------
# Registered Layer Factories
# ------------------------------------------------------------------

@register_layer("conv1d")
def build_conv1d(x, cfg: Conv1DLayerConfig, context: dict[str, Any]):
    spec = _clean_spec(cfg)

    spec.pop("l2_reg", None)
    spec.pop("fir_init", None)
    spec.pop("bn_conv", None)
    spec.pop("separable", None)

    overrides = context.get("overrides", {})
    conv_idx = context.get("conv_idx", 1)

    activation = spec.pop("activation", None)
    configured_bn = spec.pop("batch_norm", False)
    bn_config = resolve_batch_norm(configured_bn, f"bn_conv{conv_idx}", overrides)

    if spec.get("padding") == "linear":
        spec["padding"] = "valid"

    resolve_initializer(spec)
    apply_l2(spec, ("kernel_regularizer",), overrides.get("conv_l2"))

    if bn_config is not None:
        x = keras.layers.Conv1D(**spec)(x)
        x = keras.layers.BatchNormalization(**bn_config)(x)
        if activation:
            x = keras.layers.Activation(activation)(x)
    else:
        x = keras.layers.Conv1D(activation=activation, **spec)(x)

    context["conv_idx"] = conv_idx + 1
    return x


@register_layer("separable_conv1d")
def build_separable_conv1d(x, cfg: SeparableConv1DLayerConfig, context: dict[str, Any]):
    spec = _clean_spec(cfg)

    overrides = context.get("overrides", {})
    conv_idx = context.get("conv_idx", 1)

    activation = spec.pop("activation", None)
    configured_bn = spec.pop("batch_norm", False)
    bn_config = resolve_batch_norm(configured_bn, f"bn_conv{conv_idx}", overrides)

    apply_l2(spec, ("depthwise_regularizer", "pointwise_regularizer"), overrides.get("conv_l2"))

    if bn_config is not None:
        x = keras.layers.SeparableConv1D(**spec)(x)
        x = keras.layers.BatchNormalization(**bn_config)(x)
        if activation:
            x = keras.layers.Activation(activation)(x)
    else:
        x = keras.layers.SeparableConv1D(activation=activation, **spec)(x)

    context["conv_idx"] = conv_idx + 1
    return x


@register_layer("depthwise_conv1d")
def build_depthwise_conv1d(x, cfg: DepthwiseConv1DLayerConfig, context: dict[str, Any]):
    spec = _clean_spec(cfg)

    overrides = context.get("overrides", {})
    conv_idx = context.get("conv_idx", 1)

    activation = spec.pop("activation", None)
    configured_bn = spec.pop("batch_norm", False)
    bn_config = resolve_batch_norm(configured_bn, f"bn_conv{conv_idx}", overrides)

    apply_l2(spec, ("depthwise_regularizer",), overrides.get("conv_l2"))

    if bn_config is not None:
        x = keras.layers.DepthwiseConv1D(**spec)(x)
        x = keras.layers.BatchNormalization(**bn_config)(x)
        if activation:
            x = keras.layers.Activation(activation)(x)
    else:
        x = keras.layers.DepthwiseConv1D(activation=activation, **spec)(x)

    context["conv_idx"] = conv_idx + 1
    return x


@register_layer("sincconv1d")
def build_sincconv1d(x, cfg: SincConv1DLayerConfig, context: dict[str, Any]):
    spec = _clean_spec(cfg)

    overrides = context.get("overrides", {})
    conv_idx = context.get("conv_idx", 1)

    activation = spec.pop("activation", None)
    configured_bn = spec.pop("batch_norm", False)
    bn_config = resolve_batch_norm(configured_bn, f"bn_conv{conv_idx}", overrides)

    if bn_config is not None:
        x = SincConv1D(**spec)(x)
        x = keras.layers.BatchNormalization(**bn_config)(x)
        if activation:
            x = keras.layers.Activation(activation)(x)
    else:
        x = SincConv1D(**spec)(x)
        if activation:
            x = keras.layers.Activation(activation)(x)

    context["conv_idx"] = conv_idx + 1
    return x


@register_layer("repconv1d")
def build_repconv1d(x, cfg: RepConv1DLayerConfig, context: dict[str, Any]):
    spec = _clean_spec(cfg)

    overrides = context.get("overrides", {})
    conv_idx = context.get("conv_idx", 1)

    activation = spec.pop("activation", "relu")
    configured_bn = spec.pop("batch_norm", True)
    bn_config = resolve_batch_norm(configured_bn, f"bn_conv{conv_idx}", overrides)

    bn_momentum = float(spec.pop("bn_momentum", 0.99))
    bn_epsilon = float(spec.pop("bn_epsilon", 1e-3))
    if bn_config:
        bn_momentum = float(bn_config.get("momentum", bn_momentum))
        bn_epsilon = float(bn_config.get("epsilon", bn_epsilon))

    kernel_regularizer = spec.pop("kernel_regularizer", None)
    conv_l2 = overrides.get("conv_l2")
    if kernel_regularizer is None and conv_l2 is not None:
        value = float(conv_l2)
        if value > 0:
            kernel_regularizer = keras.regularizers.l2(value)

    x = RepConv1D(
        activation=activation,
        use_batch_norm=bn_config is not None,
        bn_momentum=bn_momentum,
        bn_epsilon=bn_epsilon,
        kernel_regularizer=kernel_regularizer,
        **spec,
    )(x)

    context["conv_idx"] = conv_idx + 1
    return x


@register_layer("dense")
def build_dense(x, cfg: DenseLayerConfig, context: dict[str, Any]):
    spec = _clean_spec(cfg)

    overrides = context.get("overrides", {})
    dense_idx = context.get("dense_idx", 1)

    activation = spec.pop("activation", None)
    configured_bn = spec.pop("batch_norm", False)
    bn_config = resolve_batch_norm(configured_bn, f"bn_dense{dense_idx}", overrides)

    apply_l2(spec, ("kernel_regularizer",), overrides.get("dense_l2"))

    if bn_config is not None:
        x = keras.layers.Dense(**spec)(x)
        x = keras.layers.BatchNormalization(**bn_config)(x)
        if activation:
            x = keras.layers.Activation(activation)(x)
    else:
        x = keras.layers.Dense(activation=activation, **spec)(x)

    context["dense_idx"] = dense_idx + 1
    return x


@register_layer("flatten")
def build_flatten(x, cfg: FlattenLayerConfig, context: dict[str, Any]):
    spec = _clean_spec(cfg)
    return keras.layers.Flatten(**spec)(x)


@register_layer("global_avg_pool")
@register_layer("global_avg_pool1d")
@register_layer("global_average_pooling1d")
def build_global_avg_pool(x, cfg: GlobalAvgPoolLayerConfig, context: dict[str, Any]):
    spec = _clean_spec(cfg)
    return keras.layers.GlobalAveragePooling1D(**spec)(x)


@register_layer("global_max_pool")
@register_layer("global_max_pool1d")
@register_layer("global_max_pooling1d")
def build_global_max_pool(x, cfg: GlobalMaxPoolLayerConfig, context: dict[str, Any]):
    spec = _clean_spec(cfg)
    return keras.layers.GlobalMaxPooling1D(**spec)(x)


@register_layer("maxpool1d")
@register_layer("max_pooling1d")
def build_maxpool1d(x, cfg: MaxPool1DLayerConfig, context: dict[str, Any]):
    spec = _clean_spec(cfg)
    return keras.layers.MaxPooling1D(**spec)(x)


@register_layer("avgpool1d")
@register_layer("avg_pooling1d")
def build_avgpool1d(x, cfg: AvgPool1DLayerConfig, context: dict[str, Any]):
    spec = _clean_spec(cfg)
    return keras.layers.AveragePooling1D(**spec)(x)


@register_layer("dropout")
def build_dropout(x, cfg: DropoutLayerConfig, context: dict[str, Any]):
    spec = _clean_spec(cfg)
    return keras.layers.Dropout(**spec)(x)


@register_layer("relu")
def build_relu(x, cfg: ReLULayerConfig, context: dict[str, Any]):
    spec = _clean_spec(cfg)
    return keras.layers.ReLU(**spec)(x)


@register_layer("activation")
def build_activation(x, cfg: ActivationLayerConfig, context: dict[str, Any]):
    spec = _clean_spec(cfg)
    return keras.layers.Activation(**spec)(x)


@register_layer("batch_norm")
@register_layer("batch_normalization")
def build_batch_norm(x, cfg: BatchNormLayerConfig, context: dict[str, Any]):
    spec = _clean_spec(cfg)
    return keras.layers.BatchNormalization(**spec)(x)


def _apply_layer_def(raw_def: dict[str, Any] | Any, x: Any, context: dict[str, Any]) -> Any:
    from wingbeat_ml.config.schema import parse_layer_config
    layer_config = parse_layer_config(raw_def) if isinstance(raw_def, dict) else raw_def
    factory = LAYER_REGISTRY.get(layer_config.type)
    if callable(factory):
        sig = inspect.signature(factory)
        if len(sig.parameters) == 3:
            return factory(x, layer_config, context)
        elif len(sig.parameters) == 2:
            return factory(x, layer_config)
        else:
            spec = _clean_spec(layer_config)
            return factory(**spec)(x)
    else:
        spec = _clean_spec(layer_config)
        return factory(**spec)(x)


@register_layer("concat")
@register_layer("concatenate")
@register_layer("group")
def build_concat(x, cfg: ConcatLayerConfig, context: dict[str, Any]):
    spec = _clean_spec(cfg)
    axis = spec.pop("axis", -1)
    layers_spec = spec.pop("layers", [])

    branch_outputs = []
    for branch in layers_spec:
        if isinstance(branch, list):
            branch_x = x
            for item in branch:
                branch_x = _apply_layer_def(item, branch_x, context)
            branch_outputs.append(branch_x)
        else:
            branch_outputs.append(_apply_layer_def(branch, x, context))

    if len(branch_outputs) == 1:
        return branch_outputs[0]

    return keras.layers.Concatenate(axis=axis)(branch_outputs)


__all__ = [
    "MODEL_BUILDERS",
    "LAYER_REGISTRY",
    "register_layer",
    "register_model_builder",
]
