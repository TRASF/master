"""Model analysis module: analytical 1D receptive field calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True)
class ReceptiveFieldResult:
    samples: int | None
    milliseconds: float | None
    output_jump_samples: int | None
    supported: bool
    unsupported_layers: Tuple[str, ...] = ()


def compute_receptive_field(
    model: Any, sample_rate: float = 8000.0
) -> ReceptiveFieldResult:
    """Compute analytical receptive field for a 1D sequential model."""
    layers = getattr(model, "layers", [])
    if not layers:
        return ReceptiveFieldResult(
            samples=None,
            milliseconds=None,
            output_jump_samples=None,
            supported=False,
            unsupported_layers=(),
        )

    rf = 1
    jump = 1
    is_supported = True
    unsupported: list[str] = []

    for layer in layers:
        cls_name = layer.__class__.__name__

        if cls_name in (
            "Conv1D",
            "DepthwiseConv1D",
            "SeparableConv1D",
            "SincConv1D",
            "RepConv1D",
            "MaxPooling1D",
            "AveragePooling1D",
        ):
            k_size = getattr(layer, "kernel_size", None) or getattr(layer, "pool_size", None)
            if isinstance(k_size, (tuple, list)):
                k_size = k_size[0]
            if k_size is None:
                is_supported = False
                unsupported.append(f"{layer.name} ({cls_name})")
                break

            stride = getattr(layer, "strides", 1)
            if isinstance(stride, (tuple, list)):
                stride = stride[0]
            stride = stride or 1

            dilation = getattr(layer, "dilation_rate", 1)
            if isinstance(dilation, (tuple, list)):
                dilation = dilation[0]
            dilation = dilation or 1

            effective_kernel = dilation * (k_size - 1) + 1
            rf = rf + (effective_kernel - 1) * jump
            jump = jump * stride
        elif cls_name in (
            "InputLayer",
            "BatchNormalization",
            "LayerNormalization",
            "Activation",
            "ReLU",
            "LeakyReLU",
            "Dropout",
            "SpatialDropout1D",
            "ZeroPadding1D",
            "Reshape",
            "Flatten",
            "Dense",
            "GlobalAveragePooling1D",
            "GlobalMaxPooling1D",
            "Add",
            "Concatenate",
        ):
            pass
        else:
            is_supported = False
            unsupported.append(f"{layer.name} ({cls_name})")
            break

    if not is_supported or rf <= 0:
        return ReceptiveFieldResult(
            samples=None,
            milliseconds=None,
            output_jump_samples=None,
            supported=False,
            unsupported_layers=tuple(unsupported),
        )

    rf_ms = (rf / sample_rate) * 1000.0 if sample_rate > 0 else None
    return ReceptiveFieldResult(
        samples=rf,
        milliseconds=rf_ms,
        output_jump_samples=jump,
        supported=True,
        unsupported_layers=(),
    )


__all__ = [
    "ReceptiveFieldResult",
    "compute_receptive_field",
]
