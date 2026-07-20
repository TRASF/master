"""Keras and TFLite numerical verification."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def _write_classification_report(report, path):
    labels = list(report)
    columns = sorted({
        key
        for values in report.values()
        if isinstance(values, dict)
        for key in values
    })

    with Path(path).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["label", *columns],
        )
        writer.writeheader()

        for label in labels:
            values = report[label]
            if isinstance(values, dict):
                writer.writerow({"label": label, **values})
            else:
                writer.writerow({
                    "label": label,
                    columns[0] if columns else "value": values,
                })


def _write_confusion_matrix(matrix, path):
    with Path(path).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        csv.writer(stream).writerows(np.asarray(matrix).tolist())


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

        _write_classification_report(
            report_dict,
            f"{out_prefix}_classification_report.csv",
        )
        _write_confusion_matrix(
            cm,
            f"{out_prefix}_confusion_matrix.csv",
        )

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
    result = evaluate_predictions(y_true, y_pred, class_names, out_prefix)
    result["scores"] = scores
    return result


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
