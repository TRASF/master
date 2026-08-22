"""Canonical source of truth for MosSongPlus deployment contracts and artifact metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DeploymentInputContract:
    """Canonical specification for audio input preprocessing and model quantization."""

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


ModelContract = DeploymentInputContract


@dataclass(frozen=True)
class DeploymentOutputContract:
    num_classes: int
    class_names: Tuple[str, ...]
    output_dtype: str = "int8"
    int8_scale: Optional[float] = None
    int8_zero_point: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "class_names": list(self.class_names),
            "output_dtype": self.output_dtype,
            "int8_scale": self.int8_scale,
            "int8_zero_point": self.int8_zero_point,
        }


@dataclass(frozen=True)
class DeploymentArtifact:
    format: str
    model_path: Path
    size_bytes: int
    input_contract: DeploymentInputContract
    output_contract: DeploymentOutputContract
    quantization: Dict[str, Any] | None = None
    artifact_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "model_path": str(self.model_path),
            "size_bytes": self.size_bytes,
            "input_contract": self.input_contract.to_dict(),
            "output_contract": self.output_contract.to_dict(),
            "quantization": self.quantization,
            "artifact_hash": self.artifact_hash,
        }


def quantize_float_to_int8(
    x: np.ndarray, scale: float, zero_point: int
) -> np.ndarray:
    q = np.round(np.asarray(x, dtype=np.float32) / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize_int8_to_float(
    q: np.ndarray, scale: float, zero_point: int
) -> np.ndarray:
    return (np.asarray(q, dtype=np.float32) - zero_point) * scale


def preprocess_audio_canonical(
    audio: np.ndarray,
    dc_removal: bool = True,
    normalization_method: str = "rms_normalize",
    target_rms: float = 0.05,
    min_gain: float = 0.1,
    max_gain: float = 10.0,
    fixed_range_amplitude: float = 0.03,
) -> np.ndarray:
    sig = np.squeeze(np.asarray(audio, dtype=np.float32))
    if dc_removal:
        sig = sig - np.mean(sig)
    if normalization_method == "fixed_range":
        if fixed_range_amplitude > 0:
            sig = sig / fixed_range_amplitude
    elif normalization_method == "rms_normalize":
        rms = float(np.sqrt(np.mean(sig ** 2)))
        if rms > 1e-8:
            gain = np.clip(target_rms / rms, min_gain, max_gain)
            sig = sig * gain
    return np.clip(sig, -1.0, 1.0)


def resolve_deployment_shape(
    model: Any,
    input_shape: Sequence[int] | None = None,
    config: Any | None = None,
) -> Tuple[int, int, int]:
    model_shape = getattr(model, "input_shape", None)
    if isinstance(model_shape, list) and len(model_shape) > 0:
        model_shape = model_shape[0]

    model_t_c = None
    if isinstance(model_shape, tuple) and len(model_shape) >= 2:
        m_batch = 1
        m_t = model_shape[1] if len(model_shape) >= 2 and model_shape[1] is not None else 2400
        m_c = model_shape[2] if len(model_shape) >= 3 and model_shape[2] is not None else 1
        model_t_c = (m_batch, int(m_t), int(m_c))

    if input_shape is not None:
        if len(input_shape) == 2:
            resolved = (1, int(input_shape[0]), int(input_shape[1]))
        elif len(input_shape) == 3:
            resolved = (int(input_shape[0] or 1), int(input_shape[1]), int(input_shape[2]))
        else:
            resolved = (1, 2400, 1)

        if model_t_c is not None and resolved[1:] != model_t_c[1:]:
            raise ValueError(
                f"Deployment shape mismatch: provided input shape {resolved} "
                f"does not match model shape {model_t_c}."
            )
        return resolved

    if model_t_c is not None:
        return model_t_c

    return (1, 2400, 1)


__all__ = [
    "DeploymentInputContract",
    "ModelContract",
    "DeploymentOutputContract",
    "DeploymentArtifact",
    "quantize_float_to_int8",
    "dequantize_int8_to_float",
    "preprocess_audio_canonical",
    "resolve_deployment_shape",
]
