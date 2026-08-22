"""ESP32/TFLite Micro deployment bundle generation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence, Any, Dict

import numpy as np
import tensorflow as tf


def export_input_quantization_header(
    tflite_path: str | Path,
    out_h_path: str | Path,
    amplitude_range: float,
) -> Path:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    scale, zero_point = input_detail["quantization"]

    if input_detail["dtype"] != np.int8 or scale <= 0:
        raise ValueError(
            "Expected an int8 TFLite input with a positive quantization scale."
        )
    if not np.isfinite(amplitude_range) or amplitude_range <= 0:
        raise ValueError("Input amplitude range must be positive and finite.")

    out_h_path = Path(out_h_path)
    out_h_path.parent.mkdir(parents=True, exist_ok=True)
    out_h_path.write_text(
        "#pragma once\n\n"
        "// Generated from the quantized TFLite model.\n"
        f"#define MODEL_INPUT_SCALE {scale:.10g}f\n"
        f"#define MODEL_INPUT_ZERO_POINT {zero_point}\n"
        f"#define MODEL_INPUT_AMPLITUDE_RANGE {amplitude_range:.10g}f\n",
        encoding="utf-8",
    )
    print(f"Exported input quantization header: {out_h_path}")
    return out_h_path


def export_ota_config_json(
    tflite_path: str | Path,
    out_json_path: str | Path,
    amplitude_range: float = 0.03,
    sample_rate: int = 8000,
    segment_length: int = 2400,
    detection_threshold: float = 0.60,
    class_names: Sequence[str] | None = None,
    dc_removal: bool = True,
    high_pass_filter: bool = True,
    high_pass_cutoff_hz: float = 150.0,
    pre_emphasis: bool = False,
    pre_emphasis_coeff: float = 0.97,
    rms_normalization: bool = True,
) -> Path:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    scale, zero_point = input_detail["quantization"]

    out_json_path = Path(out_json_path)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    config_data = {
        "model_version": time.strftime("%Y%m%d_%H%M%S"),
        "audio": {
            "sample_rate": sample_rate,
            "segment_length": segment_length,
            "amplitude_range": amplitude_range,
            "hop_length_ms": 150,
        },
        "preprocessing_overrides": {
            "dc_removal": dc_removal,
            "high_pass_filter": high_pass_filter,
            "high_pass_cutoff_hz": high_pass_cutoff_hz,
            "pre_emphasis": pre_emphasis,
            "pre_emphasis_coeff": pre_emphasis_coeff,
            "rms_normalization": rms_normalization,
            "target_rms": amplitude_range,
        },
        "quantization": {
            "input_scale": float(scale),
            "input_zero_point": int(zero_point),
        },
        "inference": {
            "detection_threshold": detection_threshold,
            "min_frequency_hz": 150.0,
            "max_frequency_hz": 1500.0,
        },
        "classes": list(class_names) if class_names else [],
    }

    out_json_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
    print(f"Exported OTA config JSON: {out_json_path}")
    return out_json_path


def export_tflite_to_c_header(
    tflite_path: str | Path,
    out_h_path: str | Path,
    array_name: str = "g_mossong_plus_model_data",
) -> Path:
    tflite_bytes = Path(tflite_path).read_bytes()
    out_h_path = Path(out_h_path)
    out_h_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "#pragma once\n",
        "#include <cstdint>\n",
        f"alignas(16) __attribute__((aligned(16))) const unsigned char {array_name}[] = {{",
    ]

    bytes_per_line = 12
    for i in range(0, len(tflite_bytes), bytes_per_line):
        chunk = tflite_bytes[i:i + bytes_per_line]
        hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"  {hex_str},")

    lines.append("};\n")
    lines.append(
        f"const unsigned int {array_name}_len = {len(tflite_bytes)};\n"
    )

    out_h_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported C header: {out_h_path} ({len(tflite_bytes)} bytes)")
    return out_h_path


def write_esp32_readme(out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    readme_path = out_dir / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(
        "# MosSongPlus ESP32 Deployment Bundle\n\n"
        "This bundle contains deployment artifacts:\n"
        "- `model.tflite`\n"
        "- `model_data.h`\n"
        "- `input_quantization.h`\n"
        "- `ota_config.json`\n",
        encoding="utf-8",
    )
    return readme_path


def create_export_bundle(
    tflite_path: str | Path,
    out_dir: str | Path,
    amplitude_range: float = 0.03,
    class_names: Sequence[str] | None = None,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h_q = export_input_quantization_header(
        tflite_path, out_dir / "input_quantization.h", amplitude_range
    )
    c_h = export_tflite_to_c_header(
        tflite_path, out_dir / "model_data.h"
    )
    ota = export_ota_config_json(
        tflite_path, out_dir / "ota_config.json", amplitude_range, class_names=class_names
    )
    readme = write_esp32_readme(out_dir)

    return {
        "input_quantization_header": h_q,
        "c_header": c_h,
        "ota_config": ota,
        "readme": readme,
    }


__all__ = [
    "export_input_quantization_header",
    "export_ota_config_json",
    "export_tflite_to_c_header",
    "write_esp32_readme",
    "create_export_bundle",
]
