from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


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


def _label_to_int(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)

    if y.ndim >= 2 and y.shape[-1] > 1:
        return np.argmax(y, axis=-1).astype(np.int64)

    return y.reshape(-1).astype(np.int64)


def predict_keras_dataset(
    keras_model: tf.keras.Model,
    ds: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_all = []
    y_pred_all = []
    score_all = []

    for x_batch, y_batch in ds:
        scores = keras_model(x_batch, training=False).numpy()

        y_true = _label_to_int(y_batch.numpy())
        y_pred = np.argmax(scores, axis=-1).astype(np.int64)

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        score_all.append(scores)

    return (
        np.concatenate(y_true_all),
        np.concatenate(y_pred_all),
        np.concatenate(score_all),
    )


def _quantize_input(x: np.ndarray, input_detail: Dict) -> np.ndarray:
    dtype = input_detail["dtype"]

    if dtype == np.float32:
        return x.astype(np.float32)

    scale, zero_point = input_detail["quantization"]

    if scale == 0:
        raise ValueError("Input quantization scale is 0. Invalid quantized model.")

    q = np.round(x / scale + zero_point)

    if dtype == np.int8:
        q = np.clip(q, -128, 127).astype(np.int8)
    elif dtype == np.uint8:
        q = np.clip(q, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported input dtype: {dtype}")

    return q




def dequantize_input(q: np.ndarray, input_detail: Dict) -> np.ndarray:
    scale, zero_point = input_detail["quantization"]
    if scale == 0:
        raise ValueError("Input quantization scale is 0. Invalid quantized model.")
    return (q.astype(np.float32) - zero_point) * scale


def predict_keras_with_input_qdq(
    keras_model: tf.keras.Model,
    ds: tf.data.Dataset,
    int8_tflite_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    interpreter = tf.lite.Interpreter(model_path=str(int8_tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]

    y_true_all = []
    y_pred_all = []
    score_all = []

    for x_batch, y_batch in ds:
        x = x_batch.numpy().astype(np.float32)
        q = _quantize_input(x, input_detail)
        x_qdq = dequantize_input(q, input_detail)

        scores = keras_model(x_qdq, training=False).numpy()

        y_true_all.append(_label_to_int(y_batch.numpy()))
        y_pred_all.append(np.argmax(scores, axis=-1).astype(np.int64))
        score_all.append(scores)

    return (
        np.concatenate(y_true_all),
        np.concatenate(y_pred_all),
        np.concatenate(score_all),
    )


def _dequantize_output(y: np.ndarray, output_detail: Dict) -> np.ndarray:
    dtype = output_detail["dtype"]

    if dtype == np.float32:
        return y.astype(np.float32)

    scale, zero_point = output_detail["quantization"]

    if scale == 0:
        raise ValueError("Output quantization scale is 0. Invalid quantized model.")

    return (y.astype(np.float32) - zero_point) * scale


def predict_tflite_dataset(
    tflite_path: str | Path,
    ds: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs TFLite model sample-by-sample.

    This intentionally uses batch=1 because the ESP32 model will also run one
    300 ms window at a time.
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    y_true_all = []
    y_pred_all = []
    score_all = []

    single_ds = ds.unbatch().batch(1)

    for x_batch, y_batch in single_ds:
        x = x_batch.numpy().astype(np.float32)

        expected_shape = input_detail["shape"]

        # If the converted model has a fixed batch size of 1, this should match.
        # If it has dynamic batch, this is still usually okay.
        if list(x.shape) != list(expected_shape):
            try:
                interpreter.resize_tensor_input(input_detail["index"], x.shape)
                interpreter.allocate_tensors()
                input_detail = interpreter.get_input_details()[0]
                output_detail = interpreter.get_output_details()[0]
            except Exception as e:
                raise ValueError(
                    f"TFLite input shape mismatch. Got {x.shape}, expected {expected_shape}."
                ) from e

        x_q = _quantize_input(x, input_detail)

        interpreter.set_tensor(input_detail["index"], x_q)
        interpreter.invoke()

        y_raw = interpreter.get_tensor(output_detail["index"])
        scores = _dequantize_output(y_raw, output_detail)

        y_true = _label_to_int(y_batch.numpy())
        y_pred = np.argmax(scores, axis=-1).astype(np.int64)

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        score_all.append(scores)

    return (
        np.concatenate(y_true_all),
        np.concatenate(y_pred_all),
        np.concatenate(score_all),
    )


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    out_prefix: Optional[str | Path] = None,
) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)

    result = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report_dict,
        "confusion_matrix": cm,
    }

    if out_prefix is not None:
        out_prefix = Path(out_prefix)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(report_dict).T.to_csv(f"{out_prefix}_classification_report.csv")
        pd.DataFrame(cm).to_csv(f"{out_prefix}_confusion_matrix.csv", index=False)

    return result


def evaluate_keras_model(
    keras_model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: Optional[Sequence[str]] = None,
    out_prefix: Optional[str | Path] = None,
) -> Dict:
    y_true, y_pred, scores = predict_keras_dataset(keras_model, test_ds)
    result = evaluate_predictions(y_true, y_pred, class_names, out_prefix)
    result["scores"] = scores
    return result


def evaluate_keras_input_qdq_model(
    keras_model: tf.keras.Model,
    int8_tflite_path: str | Path,
    test_ds: tf.data.Dataset,
    class_names: Optional[Sequence[str]] = None,
    out_prefix: Optional[str | Path] = None,
) -> Dict:
    y_true, y_pred, scores = predict_keras_with_input_qdq(
        keras_model, test_ds, int8_tflite_path
    )
    return evaluate_predictions(y_true, y_pred, scores, class_names, out_prefix)


def evaluate_tflite_model(
    tflite_path: str | Path,
    test_ds: tf.data.Dataset,
    class_names: Optional[Sequence[str]] = None,
    out_prefix: Optional[str | Path] = None,
) -> Dict:
    y_true, y_pred, scores = predict_tflite_dataset(tflite_path, test_ds)
    result = evaluate_predictions(y_true, y_pred, class_names, out_prefix)
    result["scores"] = scores
    return result


def compare_model_pair_agreement(
    reference_scores: np.ndarray,
    candidate_scores: np.ndarray,
) -> Dict:
    ref_top1 = np.argmax(reference_scores, axis=-1)
    cand_top1 = np.argmax(candidate_scores, axis=-1)

    top1_agreement = np.mean(ref_top1 == cand_top1)

    ref_top3 = np.argsort(reference_scores, axis=-1)[:, -3:]
    cand_top3 = np.argsort(candidate_scores, axis=-1)[:, -3:]

    top3_agreement = np.mean([
        len(set(r.tolist()).intersection(set(c.tolist()))) > 0
        for r, c in zip(ref_top3, cand_top3)
    ])

    return {
        "top1_agreement": float(top1_agreement),
        "top3_overlap": float(top3_agreement),
    }


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

    layer_stats = pd.read_csv(out_csv)

    if "mean_squared_error" in layer_stats.columns and "scale" in layer_stats.columns:
        layer_stats["rmse_over_scale"] = (
            np.sqrt(layer_stats["mean_squared_error"]) / layer_stats["scale"]
        )
        layer_stats.to_csv(out_csv, index=False)

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


def run_basic_quantization_suite(
    keras_model: tf.keras.Model,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    out_dir: str | Path,
    class_names: Optional[Sequence[str]] = None,
    rep_samples: int = 500,
    seed: int = 42,
    input_amplitude_range: float = 0.03,
    allow_dummy_calibration: bool = False,
) -> Dict:
    out_dir = ensure_dir(out_dir)

    if val_ds is not None:
        representative_dataset = make_representative_dataset(
            val_ds=val_ds,
            max_samples=rep_samples,
            seed=seed,
        )
    elif allow_dummy_calibration:
        print("WARNING: Using dummy representative data. Output is not deployable.")
        def representative_dataset():
            np.random.seed(seed)
            for _ in range(rep_samples):
                x = np.random.normal(0.0, 0.05, (1, 2400, 1)).astype(np.float32)
                x = np.clip(x, -1.0, 1.0)
                yield [x]
    else:
        raise RuntimeError(
            "Full INT8 deployment requires a real representative dataset. "
            "Pass --allow_dummy_calibration only for non-deployable smoke tests."
        )

    paths = {}

    paths["float"] = convert_float_tflite(
        keras_model,
        out_dir / "model_float32.tflite",
    )

    paths["dynamic"] = convert_dynamic_range_tflite(
        keras_model,
        out_dir / "model_dynamic_range.tflite",
    )

    paths["int8"] = convert_full_int8_tflite(
        keras_model,
        representative_dataset,
        out_dir / "model_full_int8.tflite",
    )

    # Export to C headers for ESP32/TFLite Micro deployment
    paths["int8_header"] = export_tflite_to_c_header(
        paths["int8"],
        out_dir / "model_full_int8.h",
        array_name="g_model",
    )

    paths["float_header"] = export_tflite_to_c_header(
        paths["float"],
        out_dir / "model_float32.h",
        array_name="g_model_float32",
    )

    paths["input_quant_config"] = export_input_quantization_header(
        paths["int8"],
        out_dir / "model_input_quantization.h",
        amplitude_range=input_amplitude_range,
    )

    # Write deployment guide
    write_esp32_readme(out_dir, array_name="g_model")

    results = {}

    if test_ds is not None:
        print("\nEvaluating Keras float model...")
        results["keras_float"] = evaluate_keras_model(
            keras_model,
            test_ds,
            class_names=class_names,
            out_prefix=out_dir / "keras_float",
        )

        print("\nEvaluating float TFLite model...")
        results["tflite_float"] = evaluate_tflite_model(
            paths["float"],
            test_ds,
            class_names=class_names,
            out_prefix=out_dir / "tflite_float",
        )

        print("\nEvaluating dynamic-range TFLite model...")
        results["tflite_dynamic"] = evaluate_tflite_model(
            paths["dynamic"],
            test_ds,
            class_names=class_names,
            out_prefix=out_dir / "tflite_dynamic",
        )

        print("\nEvaluating Keras model with input QDQ...")
        results["keras_input_qdq"] = evaluate_keras_input_qdq_model(
            keras_model,
            paths["int8"],
            test_ds,
            class_names=class_names,
            out_prefix=out_dir / "keras_input_qdq",
        )

        print("\nEvaluating full-int8 TFLite model...")
        results["tflite_int8"] = evaluate_tflite_model(
            paths["int8"],
            test_ds,
            class_names=class_names,
            out_prefix=out_dir / "tflite_int8",
        )

        summary_rows = []

        for name, result in results.items():
            summary_rows.append({
                "model": name,
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "weighted_f1": result["weighted_f1"],
            })

        summary = pd.DataFrame(summary_rows)

        # Agreement against Keras float.
        for name in ["tflite_float", "tflite_dynamic", "keras_input_qdq", "tflite_int8"]:
            agreement = compare_model_pair_agreement(
                results["keras_float"]["scores"],
                results[name]["scores"],
            )

            summary.loc[summary["model"] == name, "top1_agreement_with_keras"] = agreement["top1_agreement"]
            summary.loc[summary["model"] == name, "top3_overlap_with_keras"] = agreement["top3_overlap"]

        summary_path = out_dir / "quantization_summary.csv"
        summary.to_csv(summary_path, index=False)

        print("\nQuantization summary")
        print(summary.to_string(index=False))
        print(f"\nSaved summary: {summary_path}")
    else:
        print("\nSkipping evaluation as dataset is not available.")
        summary = None
        summary_path = None

    print("\nInspecting int8 model I/O...")
    inspect_tflite_io(paths["int8"])

    dump_tflite_analyzer(
        paths["int8"],
        out_dir / "model_full_int8_analyzer.txt",
    )

    if val_ds is not None:
        run_quantization_debugger(
            keras_model,
            representative_dataset,
            out_dir / "quantization_debugger.csv",
        )

    return {
        "paths": paths,
        "results": results,
        "summary": summary,
        "summary_path": summary_path,
    }


def export_tflite_to_c_header(
    tflite_path: str | Path,
    out_h_path: str | Path,
    array_name: str = "g_model",
) -> Path:
    """
    Exports TFLite model bytes to a C/C++ header file containing a byte array.
    This is required for ESP32/TFLite Micro deployment where the model is compiled into the binary.
    """
    tflite_path = Path(tflite_path)
    out_h_path = Path(out_h_path)
    out_h_path.parent.mkdir(parents=True, exist_ok=True)

    model_bytes = tflite_path.read_bytes()
    num_bytes = len(model_bytes)

    lines = [
        "// This file is automatically generated. Do not modify.",
        f"#ifndef {array_name.upper()}_H",
        f"#define {array_name.upper()}_H",
        "",
        "// Align to 16 bytes for alignment requirements of TFLite Micro interpreter",
        "#if defined(__GNUC__) || defined(__clang__)",
        "  #define TFLITE_ALIGN __attribute__((aligned(16)))",
        "#else",
        "  #define TFLITE_ALIGN alignas(16)",
        "#endif",
        "",
        f"const unsigned int {array_name}_len = {num_bytes};",
        f"const unsigned char {array_name}[] TFLITE_ALIGN = {{",
    ]

    # Format bytes in chunks of 12 per line for readability
    chunk_size = 12
    for i in range(0, num_bytes, chunk_size):
        chunk = model_bytes[i:i + chunk_size]
        hex_strs = [f"0x{b:02x}" for b in chunk]
        line = "    " + ", ".join(hex_strs)
        if i + chunk_size < num_bytes:
            line += ","
        lines.append(line)

    lines.extend([
        "};",
        "",
        "#endif // " + f"{array_name.upper()}_H",
    ])

    out_h_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Exported C header: {out_h_path} ({num_bytes} bytes)")
    return out_h_path


def write_esp32_readme(out_dir: Path, array_name: str = "g_model") -> Path:
    readme_path = out_dir / "README_ESP32.md"
    content = f"""# ESP32 TFLite Micro Deployment Guide

This directory contains the quantized TFLite model and the corresponding C/C++ header array ready to be compiled directly into your ESP-IDF or Arduino project.

## Files Generated
- `model_full_int8.tflite`: Full integer quantized model.
- `model_full_int8.h`: C++ header file containing the model as a byte array (`{array_name}`).
- `quantization_summary.csv`: Evaluation metrics comparing Keras float, TFLite float, dynamic range, and full int8.
- `model_full_int8_analyzer.txt`: Detail layer-by-layer structure and execution profiles.

## C++ ESP32 Integration Example

To run inference on your ESP32-S3 or other compatible microcontroller, use the following template:

```cpp
#include "model_full_int8.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Adjust tensor arena size as needed based on model complexity (e.g. 60-120 KB)
constexpr int kTensorArenaSize = 100 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {{
    // 1. Initialize TFLite model from header array
    model = tflite::GetModel({array_name});
    if (model->version() != TFLITE_SCHEMA_VERSION) {{
        // Handle error: Model schema mismatch
        return;
    }}

    // 2. Load all ops resolver (use MicroMutableOpResolver to save flash size in production)
    static tflite::AllOpsResolver resolver;

    // 3. Instantiate interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize
    );
    interpreter = &static_interpreter;

    // 4. Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {{
        // Handle allocation failure
        return;
    }}

    // 5. Cache input/output tensor pointers
    input = interpreter->input(0);
    output = interpreter->output(0);
}}

void loop() {{
    // Fill input->data.int8 with quantized input values:
    // For raw input x (float):
    // input->data.int8[i] = (int8_t)round(x[i] / input->params.scale + input->params.zero_point);

    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {{
        // Handle invocation failure
        return;
    }}

    // Read output->data.int8:
    // Dequantize output if you need float scores:
    // float score = (output->data.int8[class_idx] - output->params.zero_point) * output->params.scale;
}}
```
"""
    readme_path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"Created ESP32 deployment instructions: {readme_path}")
    return readme_path


if __name__ == "__main__":
    import argparse
    import random

    parser = argparse.ArgumentParser(description="Quantization and TFLite compilation suite for MosSongPlus")
    parser.add_argument("--defaults_path", type=str, default="configs/defaults.yaml", help="Path to defaults.yaml config")
    parser.add_argument("--model_cfg_path", type=str, default="configs/model.yaml", help="Path to model.yaml config")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to trained Keras model weights checkpoint (.h5)")
    parser.add_argument("--out_dir", type=str, default="quantization_output", help="Directory to save quantized models and C headers")
    parser.add_argument("--rep_samples", type=int, default=500, help="Number of samples to calibrate quantization")
    parser.add_argument("--input_amplitude_range", type=float, default=None, help="Fixed normalized waveform half-range before clipping")
    parser.add_argument("--allow_dummy_calibration", action="store_true", help="Allow non-deployable dummy calibration if dataset loading fails")
    args = parser.parse_args()

    # Load and Normalize configs
    try:
        # Import local utilities
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

        from configs.mos_config import load_config, normalize_config, apply_reproducibility_environment
        from model.mossongplus import MosSongPlusModel
        from src.framework.supervised.dataset import SupervisedDataset
    except ImportError as e:
        print(f"Error importing required modules. Make sure you run this script from the workspace root: {e}")
        sys.exit(1)

    defaults_raw = load_config(args.defaults_path)
    cfg = normalize_config(defaults_raw)
    model_cfg = load_config(args.model_cfg_path)

    apply_reproducibility_environment(cfg["reproducibility"])
    if cfg["reproducibility"]["enabled"]:
        seed = cfg["reproducibility"]["seed"]
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print(f"Reproducibility enabled. Seed: {seed}")

    print("Loading data...")
    try:
        ds_builder = SupervisedDataset(
            dataset_dir=cfg["dataset"].get("train_dir") or cfg["dataset"]["indoor"],
            val_dir=cfg["dataset"]["val_dir"],
            test_dir=cfg["dataset"]["test_dir"],
            sample_rate=cfg["audio"]["sample_rate"],
            segment_length=cfg["audio"]["segment_length"],
            classes=cfg["classes"],
            noise_dirs=cfg["augment"]["noise_banks"],
            augment_cfg=cfg["augment"],
            seed=cfg["reproducibility"]["seed"],
            deterministic=cfg["reproducibility"]["deterministic_data"],
            nomos_index=cfg["nomos_index"],
            labels_dict=cfg["labels"]
        )

        train_ds, val_ds, test_ds = ds_builder.build(
            split=cfg["dataset"]["split_list"],
            batch_size=cfg["train"]["batch_size"],
            shuffle=cfg["train"]["shuffle"]
        )
    except Exception as e:
        if args.allow_dummy_calibration:
            print(f"Warning: Failed to load dataset: {e}. Proceeding with non-deployable dummy calibration...")
            train_ds = val_ds = test_ds = None
        else:
            raise RuntimeError(
                "Dataset loading failed. Full INT8 deployment requires a real "
                "representative dataset."
            ) from e

    print("Building Keras model and loading weights...")
    model_builder = MosSongPlusModel(model_cfg)
    keras_model = model_builder.build(
        input_shape=(cfg["audio"]["segment_length"], 1),
        output_units=cfg["num_classes"],
        output_activation=cfg["model"]["output_activation"],
        batch_size=1
    )
    weights_path = args.weights_path
    if not os.path.exists(weights_path):
        if "\n" in weights_path or "\r" in weights_path:
            cleaned_path = weights_path.replace("\r", "").replace("\n", "").strip()
            raise FileNotFoundError(
                f"\n\n[Error] The weights file could not be found:\n'{weights_path}'\n\n"
                "It looks like there are newline characters (\\n) in your --weights_path argument.\n"
                "This usually happens when copy-pasting a wrapped directory path from a narrow terminal screen.\n"
                "Please run the command again, ensuring the entire path is copy-pasted as a single line without newlines or line wraps.\n"
                f"Attempted clean-up suggestion (verify spaces): '{cleaned_path}'\n"
            )
        else:
            raise FileNotFoundError(f"Weights file not found at: '{weights_path}'")

    keras_model.load_weights(weights_path)
    print(f"Weights successfully loaded from: {weights_path}")

    quant_cfg = cfg.get("quantization", {})
    input_amplitude_range = (
        args.input_amplitude_range
        if args.input_amplitude_range is not None
        else quant_cfg.get("input_amplitude_range", 0.03)
    )

    # Run the quantization suite
    run_basic_quantization_suite(
        keras_model=keras_model,
        val_ds=val_ds,
        test_ds=test_ds,
        out_dir=args.out_dir,
        class_names=cfg["classes"],
        rep_samples=args.rep_samples,
        seed=cfg["reproducibility"]["seed"],
        input_amplitude_range=input_amplitude_range,
        allow_dummy_calibration=args.allow_dummy_calibration,
    )
