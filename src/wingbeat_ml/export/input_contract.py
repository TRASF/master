"""Canonical source of truth for the MosSongPlus deployment input contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None


@dataclass(frozen=True)
class DeploymentInputContract:
    """Canonical versioned specification for audio input preprocessing and model quantization."""

    version: str = "1.0.0"
    sample_rate_hz: int = 8000
    frame_length_samples: int = 2400
    duration: float = 0.3
    channels: int = 1
    tensor_layout: str = "[1, T, C]"
    input_dtype: str = "int8"
    pcm_scaling: str = "float32 normalized sample in [-1.0, 1.0]"
    dc_removal: bool = True
    rms_gating: bool = False
    min_raw_rms_gate: float = 0.0005
    normalization_method: str = "rms_normalize"
    target_rms: float = 0.05
    rms_min_gain: float = 0.1
    rms_max_gain: float = 10.0
    fixed_range_amplitude: float = 0.03
    clipping_behavior: str = "clip float32 to [-1.0, 1.0]"
    model_input_shape: Tuple[int, int, int] = (1, 2400, 1)
    int8_scale: Optional[float] = None
    int8_zero_point: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["model_input_shape"] = list(self.model_input_shape)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_config(
        cls,
        config: Any,
        model_input_shape: Tuple[int, int, int] | None = None,
        int8_scale: float | None = None,
        int8_zero_point: int | None = None,
    ) -> DeploymentInputContract:
        sample_rate = 8000
        segment_length = 2400
        duration = 0.3
        dc_removal = True
        target_rms = 0.05
        min_gain = 0.1
        max_gain = 10.0
        fixed_range_amp = 0.03

        if hasattr(config, "audio"):
            sample_rate = getattr(config.audio, "sample_rate", sample_rate)
            segment_length = getattr(config.audio, "segment_length", segment_length)
            duration = getattr(config.audio, "duration", duration)
        elif isinstance(config, dict) and "audio" in config and isinstance(config["audio"], dict):
            sample_rate = config["audio"].get("sample_rate", sample_rate)
            segment_length = config["audio"].get("segment_length", segment_length)
            duration = config["audio"].get("duration", duration)

        if hasattr(config, "preprocess"):
            dc_removal = getattr(config.preprocess, "dc_removal", dc_removal)
        elif isinstance(config, dict) and "preprocess" in config and isinstance(config["preprocess"], dict):
            dc_removal = config["preprocess"].get("dc_removal", dc_removal)

        if hasattr(config, "augment") and hasattr(config.augment, "rms_norm"):
            target_rms = getattr(config.augment.rms_norm, "target_rms", target_rms)
            min_gain = getattr(config.augment.rms_norm, "min_gain", min_gain)
            max_gain = getattr(config.augment.rms_norm, "max_gain", max_gain)

        if model_input_shape is None:
            model_input_shape = (1, int(segment_length), 1)

        return cls(
            sample_rate_hz=int(sample_rate),
            frame_length_samples=int(segment_length),
            duration=float(duration),
            dc_removal=bool(dc_removal),
            target_rms=float(target_rms),
            rms_min_gain=float(min_gain),
            rms_max_gain=float(max_gain),
            fixed_range_amplitude=float(fixed_range_amp),
            model_input_shape=model_input_shape,
            int8_scale=int8_scale,
            int8_zero_point=int8_zero_point,
        )


def resolve_deployment_shape(
    model: Any | None = None,
    input_shape: Sequence[int] | None = None,
    config: Any | None = None,
) -> Tuple[int, int, int]:
    """Resolve static 3D deployment shape (1, T, C) and enforce agreement across configuration and model.

    Raises:
        ValueError: if model signature and configuration/requested shape disagree.
    """
    model_t = None
    model_c = None
    if model is not None and hasattr(model, "input_shape") and model.input_shape is not None:
        ms = model.input_shape
        if isinstance(ms, list):
            ms = ms[0]
        if isinstance(ms, tuple) and len(ms) >= 2:
            model_t = ms[1] if len(ms) >= 2 else None
            model_c = ms[2] if len(ms) >= 3 else 1

    config_t = None
    if config is not None:
        if hasattr(config, "audio") and hasattr(config.audio, "segment_length"):
            config_t = config.audio.segment_length
        elif isinstance(config, dict) and "audio" in config and isinstance(config["audio"], dict):
            config_t = config["audio"].get("segment_length")

    req_t = None
    req_c = None
    if input_shape is not None:
        if len(input_shape) == 3:
            req_t = input_shape[1]
            req_c = input_shape[2]
        elif len(input_shape) == 2:
            req_t = input_shape[0]
            req_c = input_shape[1]

    # Validate agreement between model, config, and requested shape
    t_sources = [("model", model_t), ("config", config_t), ("requested", req_t)]
    specified_t = [(src, val) for src, val in t_sources if val is not None]

    if len(specified_t) > 1:
        first_src, first_val = specified_t[0]
        for src, val in specified_t[1:]:
            if val != first_val:
                raise ValueError(
                    f"Deployment shape mismatch: {first_src} temporal length {first_val} "
                    f"disagrees with {src} temporal length {val}."
                )

    resolved_t = specified_t[0][1] if specified_t else 2400
    resolved_c = req_c or model_c or 1
    return (1, int(resolved_t), int(resolved_c))


def preprocess_audio_canonical(
    waveform: np.ndarray,
    dc_removal: bool = True,
    normalization_method: str = "rms_normalize",
    target_rms: float = 0.05,
    min_gain: float = 0.1,
    max_gain: float = 10.0,
    fixed_range_amplitude: float = 0.03,
    enable_raw_rms_gate: bool = False,
    min_raw_rms_gate: float = 0.0005,
) -> np.ndarray:
    """Canonical Python implementation of audio preprocessing pipeline.

    Operates on a 1D float32 waveform and outputs float32 waveform scaled/clipped to [-1.0, 1.0].
    """
    x = np.array(waveform, dtype=np.float32, copy=True)
    if x.size == 0:
        return x

    if dc_removal:
        x -= np.mean(x)

    raw_rms = np.sqrt(np.mean(np.square(x)))

    if enable_raw_rms_gate and raw_rms < min_raw_rms_gate:
        return np.zeros_like(x)

    if normalization_method == "rms_normalize":
        gain = target_rms / (raw_rms + 1e-8)
        gain = np.clip(gain, min_gain, max_gain)
        x = x * gain
    elif normalization_method == "fixed_range":
        gain = 1.0 / (fixed_range_amplitude if fixed_range_amplitude > 0 else 0.03)
        x = x * gain

    return np.clip(x, -1.0, 1.0).astype(np.float32)


def quantize_float_to_int8(
    x: np.ndarray,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    """Quantize reference float tensor to INT8 using exact TFLite scale and zero_point."""
    if scale <= 0:
        raise ValueError(f"Invalid INT8 scale: {scale}")
    inv_scale = 1.0 / scale
    q = np.round(x * inv_scale + zero_point)
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize_int8_to_float(
    q: np.ndarray,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    """Dequantize INT8 tensor back to float32."""
    return (q.astype(np.float32) - float(zero_point)) * float(scale)
