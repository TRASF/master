"""Deployment artifact generation: TFLite conversion & quantization."""

from __future__ import annotations

import contextlib
import csv
import io
import math
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import numpy as np
import tensorflow as tf

from wingbeat_ml.deployment.contracts import resolve_deployment_shape


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_tflite_model(model_bytes: bytes, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model_bytes)
    print(f"Saved: {path} ({path.stat().st_size / 1024:.1f} KB)")
    return path


def make_representative_dataset(
    val_ds: tf.data.Dataset,
    max_samples: int = 500,
    seed: int = 42,
    expected_shape: Sequence[int] | None = None,
) -> Callable:
    """Representative dataset generator for full INT8 calibration."""
    rep_ds = (
        val_ds
        .unbatch()
        .shuffle(10000, seed=seed, reshuffle_each_iteration=False)
        .batch(1)
        .take(max_samples)
    )

    def representative_dataset():
        for x, _ in rep_ds:
            x = tf.cast(x, tf.float32)

            if expected_shape is not None and list(x.shape) != list(expected_shape):
                raise ValueError(f"Expected input shape {list(expected_shape)}, got {list(x.shape)}")
            elif x.shape.rank != 3:
                raise ValueError(f"Expected rank-3 input shape [1, T, C], got {x.shape}")

            min_value = float(tf.reduce_min(x))
            max_value = float(tf.reduce_max(x))
            if min_value < -1.0001 or max_value > 1.0001:
                raise ValueError(
                    "Representative sample is outside model range: "
                    f"[{min_value}, {max_value}]"
                )

            yield [x]

    return representative_dataset


create_representative_dataset = make_representative_dataset


def convert_float_tflite(
    keras_model: tf.keras.Model,
    out_path: str | Path,
    input_shape: Sequence[int] | None = None,
    config: Any | None = None,
) -> Path:
    shape = resolve_deployment_shape(keras_model, input_shape=input_shape, config=config)
    run_model = tf.function(lambda x: keras_model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(shape, tf.float32)
    )
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()
    return save_tflite_model(tflite_model, out_path)


def convert_dynamic_range_tflite(
    keras_model: tf.keras.Model,
    out_path: str | Path,
    input_shape: Sequence[int] | None = None,
    config: Any | None = None,
) -> Path:
    shape = resolve_deployment_shape(keras_model, input_shape=input_shape, config=config)
    run_model = tf.function(lambda x: keras_model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(shape, tf.float32)
    )
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    return save_tflite_model(tflite_model, out_path)


def convert_full_int8_tflite(
    keras_model: tf.keras.Model,
    representative_dataset: Callable,
    out_path: str | Path,
    input_shape: Sequence[int] | None = None,
    config: Any | None = None,
) -> Path:
    """Main ESP32-S3 / TFLite Micro INT8 conversion target."""
    shape = resolve_deployment_shape(keras_model, input_shape=input_shape, config=config)
    run_model = tf.function(lambda x: keras_model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(shape, tf.float32)
    )
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    return save_tflite_model(tflite_model, out_path)


def convert_int16x8_tflite_experiment(
    keras_model: tf.keras.Model,
    representative_dataset: Callable,
    out_path: str | Path,
) -> Path:
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    ]

    tflite_model = converter.convert()
    return save_tflite_model(tflite_model, out_path)


def inspect_tflite_io(tflite_path: str | Path) -> Dict:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    ops = []
    if hasattr(interpreter, "_get_ops_details"):
        try:
            ops = sorted(list({op["op_name"] for op in interpreter._get_ops_details()}))
        except Exception:
            pass

    inp_info = {
        "name": inp["name"],
        "shape": inp["shape"].tolist(),
        "dtype": inp["dtype"],
        "quantization": {
            "scale": float(inp["quantization"][0]),
            "zero_point": int(inp["quantization"][1]),
        },
    }
    out_info = {
        "name": out["name"],
        "shape": out["shape"].tolist(),
        "dtype": out["dtype"],
        "quantization": {
            "scale": float(out["quantization"][0]),
            "zero_point": int(out["quantization"][1]),
        },
    }

    return {
        "path": str(tflite_path),
        "size_bytes": Path(tflite_path).stat().st_size,
        "input": inp_info,
        "output": out_info,
        "inputs": [inp_info],
        "outputs": [out_info],
        "operators": ops,
    }


def dump_tflite_analyzer(tflite_path: str | Path, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = f"TFLite Model Analyzer Output for {tflite_path}\n"
    out_path.write_text(summary, encoding="utf-8")
    return out_path


def run_quantization_debugger(
    float_model_path: str | Path,
    quant_model_path: str | Path,
    representative_dataset: Callable,
    out_csv_path: str | Path,
) -> Path:
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv_path.write_text("op_name,mean_squared_error,scale,rmse_over_scale\n", encoding="utf-8")
    return out_csv_path


convert_int8_tflite = convert_full_int8_tflite

__all__ = [
    "ensure_dir",
    "save_tflite_model",
    "make_representative_dataset",
    "create_representative_dataset",
    "convert_float_tflite",
    "convert_dynamic_range_tflite",
    "convert_full_int8_tflite",
    "convert_int8_tflite",
    "convert_int16x8_tflite_experiment",
    "inspect_tflite_io",
    "dump_tflite_analyzer",
    "run_quantization_debugger",
]
