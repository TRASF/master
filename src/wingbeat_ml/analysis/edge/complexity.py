"""Edge analysis module: parameter, memory, MAC, and hardware suitability analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from wingbeat_ml.config.schema import EdgeAnalysisResult
from wingbeat_ml.analysis.model.receptive_field import (
    ReceptiveFieldResult,
    compute_receptive_field,
)


@dataclass
class ModelComplexityResult:
    parameters: int
    trainable_parameters: int
    weight_bytes: int
    macs: int | None
    largest_activation_bytes: int | None
    estimated_adjacent_activation_bytes: int | None
    tensor_arena_bytes: int | None
    peak_activation_bytes: int | None
    receptive_field_samples: int | None
    receptive_field_ms: float | None
    layer_details: List[Dict[str, Any]] = field(default_factory=list)


def _get_layer_params_and_macs(layer: Any, out_shape: Any = None) -> Tuple[int, int, int | None, Any]:
    """Extract (total_params, trainable_params, macs, out_shape) for a layer."""
    cls_name = layer.__class__.__name__
    weights = getattr(layer, "weights", [])
    total_p = sum(int(np.prod(w.shape)) for w in weights if hasattr(w, "shape"))
    
    trainable_weights = getattr(layer, "trainable_weights", [])
    trainable_p = sum(int(np.prod(w.shape)) for w in trainable_weights if hasattr(w, "shape"))

    if out_shape is None:
        out_shape = getattr(layer, "output_shape", None)

    if isinstance(out_shape, list) and len(out_shape) > 0:
        out_shape = out_shape[0]

    out_len = 1
    out_ch = 1
    if isinstance(out_shape, tuple) and len(out_shape) >= 2:
        out_len = out_shape[1] if out_shape[1] is not None else 1
        if len(out_shape) >= 3:
            out_ch = out_shape[2] if out_shape[2] is not None else 1

    macs: int | None = None

    kernel_weight = getattr(layer, "kernel", None)
    if kernel_weight is None and hasattr(layer, "weights") and len(layer.weights) > 0:
        kernel_weight = layer.weights[0]

    if cls_name in ("Conv1D", "SincConv1D", "RepConv1D"):
        if kernel_weight is not None and hasattr(kernel_weight, "shape"):
            w_shape = kernel_weight.shape
            if len(w_shape) == 3:
                k_size, in_c, o_c = w_shape
                groups = getattr(layer, "groups", 1) or 1
                macs = int(out_len * o_c * k_size * (in_c / groups))
    elif cls_name == "DepthwiseConv1D":
        dw_kernel = getattr(layer, "depthwise_kernel", None) or kernel_weight
        if dw_kernel is not None and hasattr(dw_kernel, "shape"):
            w_shape = dw_kernel.shape
            if len(w_shape) == 3:
                k_size, in_c, depth_mult = w_shape
                macs = int(out_len * in_c * depth_mult * k_size)
    elif cls_name == "SeparableConv1D":
        dw_k = getattr(layer, "depthwise_kernel", None)
        pw_k = getattr(layer, "pointwise_kernel", None)
        if dw_k is not None and pw_k is not None and hasattr(dw_k, "shape") and hasattr(pw_k, "shape"):
            dw_shape = dw_k.shape
            pw_shape = pw_k.shape
            if len(dw_shape) == 3 and len(pw_shape) == 3:
                k_size, in_c, depth_mult = dw_shape
                _, _, o_c = pw_shape
                dw_macs = out_len * in_c * depth_mult * k_size
                pw_macs = out_len * in_c * depth_mult * o_c
                macs = int(dw_macs + pw_macs)
    elif cls_name == "Dense":
        if kernel_weight is not None and hasattr(kernel_weight, "shape"):
            w_shape = kernel_weight.shape
            if len(w_shape) == 2:
                in_dim, out_dim = w_shape
                macs = int(in_dim * out_dim)

    return total_p, trainable_p, macs, out_shape


def analyze_edge_complexity(
    model: Any, sample_rate: float = 8000.0
) -> EdgeAnalysisResult:
    """Analyze a Keras or TFLite model for MCU hardware constraints.

    Memory definitions:
    - largest_activation_bytes: Maximum single layer output activation tensor in bytes.
    - peak_activation_bytes: Peak live working memory (sum of 2 adjacent layer output tensors for double-buffering).
    - model_bytes: Total parameter weight storage size in bytes (float32 assumption).
    """
    if not hasattr(model, "count_params"):
        return EdgeAnalysisResult(
            parameters=0,
            model_bytes=0,
            macs=0,
            peak_activation_bytes=None,
            trainable_parameters=0,
            largest_activation_bytes=None,
            receptive_field_samples=None,
            receptive_field_ms=None,
            layer_details=[],
        )

    if hasattr(model, "built") and not getattr(model, "built", True):
        inp_shape = getattr(model, "input_shape", None)
        if inp_shape:
            try:
                model.build(inp_shape)
            except Exception:
                pass

    total_params = int(model.count_params())
    trainable_params = sum(
        int(np.prod(w.shape)) for w in getattr(model, "trainable_weights", []) if hasattr(w, "shape")
    )
    weight_bytes = total_params * 4  # float32 assumption

    total_macs = 0
    largest_act_bytes = 0
    prev_act_bytes = 0
    peak_act_bytes = 0
    layer_details: List[Dict[str, Any]] = []

    curr_shape = getattr(model, "input_shape", None)
    if isinstance(curr_shape, list) and len(curr_shape) > 0:
        curr_shape = curr_shape[0]

    for layer in getattr(model, "layers", []):
        out_shape = getattr(layer, "output_shape", None)
        if out_shape is None and curr_shape is not None and hasattr(layer, "compute_output_shape"):
            try:
                out_shape = layer.compute_output_shape(curr_shape)
            except Exception:
                pass

        if out_shape is not None:
            curr_shape = out_shape

        t_p, tr_p, l_macs, out_shape = _get_layer_params_and_macs(layer, out_shape=out_shape)
        if l_macs is not None:
            total_macs += l_macs

        act_bytes = 0
        if isinstance(out_shape, tuple) and len(out_shape) >= 2:
            dim = int(np.prod([s for s in out_shape[1:] if s is not None]))
            dtype_str = str(getattr(layer, "dtype", "float32"))
            dtype_policy_str = str(getattr(layer, "dtype_policy", ""))
            is_int8 = getattr(layer, "is_int8", False) or "int8" in dtype_str or "int8" in dtype_policy_str
            is_f16 = "float16" in dtype_str or "float16" in dtype_policy_str
            dtype_size = 1 if is_int8 else (2 if is_f16 else 4)
            act_bytes = dim * dtype_size

        largest_act_bytes = max(largest_act_bytes, act_bytes)
        peak_act_bytes = max(peak_act_bytes, act_bytes + prev_act_bytes)
        prev_act_bytes = act_bytes

        layer_details.append(
            {
                "layer_name": layer.name,
                "layer_type": layer.__class__.__name__,
                "parameters": t_p,
                "trainable_parameters": tr_p,
                "macs": l_macs,
                "activation_bytes": act_bytes,
                "output_shape": str(out_shape),
            }
        )

    rf_result = compute_receptive_field(model, sample_rate)

    return EdgeAnalysisResult(
        parameters=total_params,
        model_bytes=weight_bytes,
        macs=total_macs,
        largest_activation_bytes=largest_act_bytes if largest_act_bytes > 0 else None,
        estimated_adjacent_activation_bytes=peak_act_bytes if peak_act_bytes > 0 else None,
        tensor_arena_bytes=None,  # Reserved for actual TFLite Micro planner measurement
        peak_activation_bytes=peak_act_bytes if peak_act_bytes > 0 else None,
        trainable_parameters=trainable_params,
        receptive_field_samples=rf_result.samples if rf_result.supported else None,
        receptive_field_ms=rf_result.milliseconds if rf_result.supported else None,
        layer_details=layer_details,
    )


__all__ = [
    "analyze_edge_complexity",
    "compute_receptive_field",
    "ReceptiveFieldResult",
    "ModelComplexityResult",
]
