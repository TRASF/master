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
    q = np.round(x / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def _dequantize_output(y: np.ndarray, output_detail: Dict) -> np.ndarray:
    dtype = output_detail["dtype"]

    if dtype == np.float32:
        return y.astype(np.float32)

    scale, zero_point = output_detail["quantization"]
    return (y.astype(np.float32) - zero_point) * scale


def predict_tflite_dataset(
    tflite_path: str | Path,
    ds: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    y_true_all = []
    y_pred_all = []
    score_all = []

    for x_batch, y_batch in ds:
        x_np = x_batch.numpy()
        y_true = _label_to_int(y_batch.numpy())

        batch_size = x_np.shape[0]
        batch_scores = []

        for i in range(batch_size):
            sample = x_np[i:i + 1]
            q_sample = _quantize_input(sample, input_detail)

            interpreter.set_tensor(input_detail["index"], q_sample)
            interpreter.invoke()

            raw_out = interpreter.get_tensor(output_detail["index"])
            deq_out = _dequantize_output(raw_out, output_detail)
            batch_scores.append(deq_out[0])

        batch_scores = np.stack(batch_scores, axis=0)
        y_pred = np.argmax(batch_scores, axis=-1).astype(np.int64)

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        score_all.append(batch_scores)

    return (
        np.concatenate(y_true_all),
        np.concatenate(y_pred_all),
        np.concatenate(score_all),
    )


def predict_keras_input_qdq_dataset(
    keras_model: tf.keras.Model,
    tflite_path: str | Path,
    ds: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]

    y_true_all = []
    y_pred_all = []
    score_all = []

    for x_batch, y_batch in ds:
        x_np = x_batch.numpy()
        y_true = _label_to_int(y_batch.numpy())

        q_sample = _quantize_input(x_np, input_detail)
        deq_sample = _dequantize_output(q_sample, input_detail)

        scores = keras_model(deq_sample, training=False).numpy()
        y_pred = np.argmax(scores, axis=-1).astype(np.int64)

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        score_all.append(scores)

    return (
        np.concatenate(y_true_all),
        np.concatenate(y_pred_all),
        np.concatenate(score_all),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Sequence[str] | None = None,
) -> Dict:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    weighted_f1 = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    cm = confusion_matrix(y_true, y_pred)

    if target_names is not None:
        report = classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
    else:
        report = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def evaluate_keras_model(
    keras_model: tf.keras.Model,
    ds: tf.data.Dataset,
    target_names: Sequence[str] | None = None,
    out_dir: str | Path | None = None,
) -> Dict:
    y_true, y_pred, scores = predict_keras_dataset(keras_model, ds)
    metrics = compute_metrics(y_true, y_pred, target_names=target_names)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_classification_report(
            metrics["classification_report"],
            out_dir / "keras_classification_report.csv",
        )
        _write_confusion_matrix(
            metrics["confusion_matrix"],
            out_dir / "keras_confusion_matrix.csv",
        )

    return metrics


def evaluate_tflite_model(
    tflite_path: str | Path,
    ds: tf.data.Dataset,
    target_names: Sequence[str] | None = None,
    out_dir: str | Path | None = None,
    prefix: str = "tflite",
) -> Dict:
    y_true, y_pred, scores = predict_tflite_dataset(tflite_path, ds)
    metrics = compute_metrics(y_true, y_pred, target_names=target_names)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_classification_report(
            metrics["classification_report"],
            out_dir / f"{prefix}_classification_report.csv",
        )
        _write_confusion_matrix(
            metrics["confusion_matrix"],
            out_dir / f"{prefix}_confusion_matrix.csv",
        )

    return metrics


def evaluate_keras_input_qdq_model(
    keras_model: tf.keras.Model,
    tflite_path: str | Path,
    ds: tf.data.Dataset,
    target_names: Sequence[str] | None = None,
    out_dir: str | Path | None = None,
) -> Dict:
    y_true, y_pred, scores = predict_keras_input_qdq_dataset(
        keras_model, tflite_path, ds
    )
    metrics = compute_metrics(y_true, y_pred, target_names=target_names)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_classification_report(
            metrics["classification_report"],
            out_dir / "keras_input_qdq_classification_report.csv",
        )
        _write_confusion_matrix(
            metrics["confusion_matrix"],
            out_dir / "keras_input_qdq_confusion_matrix.csv",
        )

    return metrics


def compare_model_pair_agreement(
    y_pred_base: np.ndarray,
    y_pred_candidate: np.ndarray,
    scores_base: np.ndarray | None = None,
    scores_candidate: np.ndarray | None = None,
    top_k: int = 3,
) -> Dict:
    y_base = np.asarray(y_pred_base)
    y_cand = np.asarray(y_pred_candidate)

    if y_base.ndim >= 2:
        if scores_base is None:
            scores_base = y_base
        y_base = np.argmax(y_base, axis=-1)

    if y_cand.ndim >= 2:
        if scores_candidate is None:
            scores_candidate = y_cand
        y_cand = np.argmax(y_cand, axis=-1)

    top1_agreement = float(accuracy_score(y_base, y_cand))

    top_k_overlap = 1.0
    if scores_base is not None and scores_candidate is not None:
        top_k_base = np.argsort(scores_base, axis=-1)[:, -top_k:]
        top_k_candidate = np.argsort(scores_candidate, axis=-1)[:, -top_k:]

        overlap_count = 0
        total = len(y_base)

        for i in range(total):
            s_base = set(top_k_base[i])
            s_cand = set(top_k_candidate[i])
            if len(s_base.intersection(s_cand)) > 0:
                overlap_count += 1

        top_k_overlap = float(overlap_count / total) if total > 0 else 0.0

    return {
        "top1_agreement": top1_agreement,
        "top3_overlap": top_k_overlap,
    }


def verify_tflite_against_keras(
    keras_model: tf.keras.Model,
    tflite_path: str | Path,
    ds: tf.data.Dataset,
    atol: float = 1e-3,
    class_names: Sequence[str] | None = None,
) -> Dict:
    y_true_k, y_pred_k, scores_k = predict_keras_dataset(keras_model, ds)
    y_true_t, y_pred_t, scores_t = predict_tflite_dataset(tflite_path, ds)

    agreement = compare_model_pair_agreement(
        y_pred_k, y_pred_t, scores_k, scores_t
    )
    max_abs_diff = float(np.max(np.abs(scores_k - scores_t)))

    return {
        "verified": bool(max_abs_diff <= atol or agreement["top1_agreement"] >= 0.95),
        "max_abs_diff": max_abs_diff,
        "top1_agreement": agreement["top1_agreement"],
        "top3_overlap": agreement["top3_overlap"],
    }


__all__ = [
    "predict_keras_dataset",
    "predict_tflite_dataset",
    "predict_keras_input_qdq_dataset",
    "compute_metrics",
    "evaluate_keras_model",
    "evaluate_tflite_model",
    "evaluate_keras_input_qdq_model",
    "compare_model_pair_agreement",
    "verify_tflite_against_keras",
]
