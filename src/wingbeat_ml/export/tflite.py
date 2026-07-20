"""TFLite conversion and quantization diagnostics."""

from __future__ import annotations

import contextlib
import csv
import io
import math
from pathlib import Path
from typing import Callable, Dict, Sequence

import numpy as np
import tensorflow as tf


def _append_rmse_over_scale(path):
    """Append the derived debugger column using the CSV library."""
    path = Path(path)

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())

    if (
        "mean_squared_error" not in fieldnames
        or "scale" not in fieldnames
    ):
        return

    if "rmse_over_scale" not in fieldnames:
        fieldnames.append("rmse_over_scale")

    for row in rows:
        scale = float(row["scale"])
        mse = float(row["mean_squared_error"])
        row["rmse_over_scale"] = (
            math.sqrt(mse) / scale if scale else ""
        )

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
) -> Callable:
    """
    Representative dataset for full-int8 calibration.

    Expected val_ds item:
        x: [batch, 2400, 1], float32
        y: one-hot or integer label

    Output required by TFLite converter:
        yield [x_single_batch]
        where x_single_batch is [1, 2400, 1], float32
    """

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

            # Defensive shape/range checks. The dataset must already be in
            # model space: DC removed, scaled by the configured amplitude range,
            # and clipped to [-1, 1].
            if x.shape.rank != 3:
                raise ValueError(f"Expected rank-3 input [1, 2400, 1], got {x.shape}")

            min_value = float(tf.reduce_min(x))
            max_value = float(tf.reduce_max(x))
            if min_value < -1.0001 or max_value > 1.0001:
                raise ValueError(
                    "Representative sample is outside model range: "
                    f"[{min_value}, {max_value}]"
                )

            yield [x]

    return representative_dataset


def convert_float_tflite(
    keras_model: tf.keras.Model,
    out_path: str | Path,
) -> Path:
    run_model = tf.function(lambda x: keras_model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, 2400, 1], tf.float32)
    )
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()
    return save_tflite_model(tflite_model, out_path)


def convert_dynamic_range_tflite(
    keras_model: tf.keras.Model,
    out_path: str | Path,
) -> Path:
    """
    Weight-focused post-training quantization.
    Useful as an intermediate diagnostic, not the final ESP32 target.
    """
    run_model = tf.function(lambda x: keras_model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, 2400, 1], tf.float32)
    )
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    return save_tflite_model(tflite_model, out_path)


def convert_full_int8_tflite(
    keras_model: tf.keras.Model,
    representative_dataset: Callable,
    out_path: str | Path,
) -> Path:
    """
    Main ESP32-S3 / TFLite Micro target.

    Produces:
        int8 input
        int8 output
        int8 built-in ops only, if conversion succeeds
    """
    run_model = tf.function(lambda x: keras_model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, 2400, 1], tf.float32)
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
    """
    Experimental fallback for raw-audio if full-int8 hurts accuracy badly.

    Not the first ESP32 target.
    Use only as an accuracy experiment.
    """
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

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\nTFLite input details")
    for d in input_details:
        print({
            "name": d["name"],
            "shape": d["shape"].tolist(),
            "dtype": str(d["dtype"]),
            "quantization": d["quantization"],
        })

    print("\nTFLite output details")
    for d in output_details:
        print({
            "name": d["name"],
            "shape": d["shape"].tolist(),
            "dtype": str(d["dtype"]),
            "quantization": d["quantization"],
        })

    return {
        "inputs": input_details,
        "outputs": output_details,
    }


def dump_tflite_analyzer(
    tflite_path: str | Path,
    out_txt: str | Path,
) -> Path:
    out_txt = Path(out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        tf.lite.experimental.Analyzer.analyze(model_path=str(tflite_path))

    out_txt.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"Saved analyzer report: {out_txt}")
    return out_txt


def run_quantization_debugger(
    keras_model: tf.keras.Model,
    representative_dataset: Callable,
    out_csv: str | Path,
) -> Path:
    """
    Layer-wise quantization error report.

    This is analysis only. It does not replace normal evaluation on test_ds.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    def mean_abs_error(diff):
        return np.mean(np.abs(diff))

    def safe_correlation(float_tensor, quant_tensor, scale, zero_point):
        # Cast to float64 for better numerical stability and prevent standard overflow
        float_tensor = float_tensor.flatten().astype(np.float64)
        quant_tensor = quant_tensor.flatten().astype(np.float64)

        dequant = (quant_tensor - zero_point) * scale

        if float_tensor.size < 2:
            return 0.0

        # Safely handle non-finite numbers (NaN/Inf)
        if not np.all(np.isfinite(float_tensor)) or not np.all(np.isfinite(dequant)):
            return 0.0

        with np.errstate(invalid='ignore', over='ignore'):
            std_float = np.std(float_tensor)
            std_dequant = np.std(dequant)

            if std_float < 1e-12 or std_dequant < 1e-12:
                return 0.0

            corr = np.corrcoef(float_tensor, dequant)
            if corr is not None and corr.shape == (2, 2):
                val = corr[0, 1]
                return float(val) if np.isfinite(val) else 0.0
        return 0.0

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    debug_options = tf.lite.experimental.QuantizationDebugOptions(
        layer_debug_metrics={
            "mean_abs_error": mean_abs_error,
        },
        layer_direct_compare_metrics={
            "correlation": safe_correlation,
        },
    )

    debugger = tf.lite.experimental.QuantizationDebugger(
        converter=converter,
        debug_dataset=representative_dataset,
        debug_options=debug_options,
    )

    debugger.run()

    with out_csv.open("w", encoding="utf-8") as f:
        debugger.layer_statistics_dump(f)

    _append_rmse_over_scale(out_csv)

    print(f"Saved quantization debugger report: {out_csv}")
    return out_csv


def convert_selective_quantized_tflite_for_analysis(
    keras_model: tf.keras.Model,
    representative_dataset: Callable,
    denylisted_nodes: Sequence[str],
    out_path: str | Path,
) -> Path:
    """
    Mixed float/int quantization for diagnosis.

    Do not assume this is ESP32-ready.
    Use this to identify which layers are sensitive.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    debug_options = tf.lite.experimental.QuantizationDebugOptions(
        denylisted_nodes=list(denylisted_nodes)
    )

    debugger = tf.lite.experimental.QuantizationDebugger(
        converter=converter,
        debug_dataset=representative_dataset,
        debug_options=debug_options,
    )

    selective_model = debugger.get_nondebug_quantized_model()
    return save_tflite_model(selective_model, out_path)
