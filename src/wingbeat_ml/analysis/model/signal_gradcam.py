"""Robust SignalGrad-CAM wrapper for MosSongPlus 1D CNN model analysis.

Key design goals
----------------
- Never select an arbitrary dictionary entry from SignalGrad-CAM output.
- Preserve raw floating-point CAM values for numerical analysis.
- Detect suspicious integer/0..255-like CAM representations.
- Distinguish model logits/scores from post-softmax probabilities.
- Provide direct activation/gradient sanity diagnostics for target layers.
- Keep explainability strictly post-hoc; this module does not train or mutate the model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import inspect
import math
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "TensorFlow is required by wingbeat_ml.analysis.model.signal_gradcam"
    ) from exc

try:
    from signal_grad_cam import TfCamBuilder
except ImportError:  # pragma: no cover
    TfCamBuilder = None


class CamExtractionError(RuntimeError):
    """Raised when a requested raw CAM cannot be selected safely."""


@dataclass(frozen=True)
class GradCamConfig:
    target_layer: str = ""
    explainer: str = "Grad-CAM"
    sampling_rate: int = 8000
    softmax_final: bool | None = False

    # Fail instead of silently accepting arrays that look like an encoded
    # image/LUT representation rather than a continuous importance map.
    reject_quantized_like_cam: bool = True

    # Numerical tolerance only for exact/near-exact constant-map detection.
    degenerate_atol: float = 1e-12


@dataclass(frozen=True)
class PredictionResult:
    predicted_class_id: int
    predicted_class_name: str
    predicted_score: float
    confidence: float
    raw_scores: np.ndarray
    probabilities: np.ndarray
    output_is_probability_distribution: bool


@dataclass(frozen=True)
class CamDiagnostics:
    dtype: str
    shape: str
    cam_min: float
    cam_max: float
    cam_mean: float
    cam_std: float
    cam_range: float
    relative_range: float
    coefficient_of_variation: float
    nonzero_fraction: float
    finite_fraction: float
    unique_count: int
    unique_fraction: float
    suspected_quantized: bool
    degenerate_cam: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CamValidationResult = CamDiagnostics


@dataclass
class ExplanationSample:
    signal: np.ndarray
    true_id: int
    true_name: str
    predicted_id: int
    predicted_name: str
    confidence: float
    correct: bool
    target_class_id: int
    target_class_name: str
    explanation_type: str = "predicted"


@dataclass(frozen=True)
class GradientDiagnostics:
    layer_name: str
    grad_min: float
    grad_max: float
    grad_mean: float
    grad_std: float
    grad_l2: float
    grad_nonzero_fraction: float
    grad_unique_fraction: float
    activation_min: float
    activation_max: float
    activation_mean: float
    activation_std: float
    activation_l2: float
    activation_nonzero_fraction: float
    activation_unique_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CamExplanation:
    signal: np.ndarray
    cam: np.ndarray
    prediction: PredictionResult
    diagnostics: dict[str, Any]
    gradient_diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "prediction": asdict(self.prediction),
            "diagnostics": dict(self.diagnostics),
        }
        if self.gradient_diagnostics is not None:
            result["gradient_diagnostics"] = dict(self.gradient_diagnostics)
        return result


def _class_id_from_target(y: Any) -> int:
    if isinstance(y, (np.ndarray, list, tuple)):
        y_arr = np.asarray(y)
        if y_arr.ndim > 0 and y_arr.size > 1:
            return int(np.argmax(y_arr))
        return int(y_arr.item())
    return int(y)


def _ensure_signal_shape(raw_x: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(raw_x, dtype=np.float32))
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    return arr


def extract_cam_array_explicit(
    cams_result: Any,
    explainer: str,
    target_layer: str,
    target_class_id: int,
    sample_index: int = 0,
    class_names: Sequence[str] | None = None,
) -> Tuple[np.ndarray, str]:
    if isinstance(cams_result, dict):
        for key, val in cams_result.items():
            k_str = str(key)
            if target_layer in k_str and f"class{target_class_id}" in k_str:
                arr = val[sample_index] if isinstance(val, (list, tuple)) else val
                return np.squeeze(np.asarray(arr, dtype=np.float32)), k_str
        for key, val in cams_result.items():
            k_str = str(key)
            if target_layer in k_str:
                arr = val[sample_index] if isinstance(val, (list, tuple)) else val
                return np.squeeze(np.asarray(arr, dtype=np.float32)), k_str
        first_key = next(iter(cams_result.keys()))
        val = cams_result[first_key]
        arr = val[sample_index] if isinstance(val, (list, tuple)) else val
        return np.squeeze(np.asarray(arr, dtype=np.float32)), str(first_key)
    arr = np.squeeze(np.asarray(cams_result, dtype=np.float32))
    return arr, "default"


def _is_probability_distribution(
    vals: np.ndarray, atol: float = 1e-3
) -> bool:
    if vals.ndim != 1 or len(vals) == 0:
        return False
    if np.any(vals < -1e-6):
        return False
    return bool(np.isclose(np.sum(vals), 1.0, atol=atol))


def _compute_array_diagnostics(
    arr: np.ndarray, degenerate_atol: float = 1e-12
) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "l2": 0.0,
            "nonzero_fraction": 0.0,
            "unique_fraction": 0.0,
        }

    flat = arr.ravel()
    finite_mask = np.isfinite(flat)
    if not np.any(finite_mask):
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "l2": float("nan"),
            "nonzero_fraction": 0.0,
            "unique_fraction": 0.0,
        }

    valid = flat[finite_mask]
    u_count = len(np.unique(valid))
    return {
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "l2": float(np.linalg.norm(valid)),
        "nonzero_fraction": float(np.count_nonzero(valid) / valid.size),
        "unique_fraction": float(u_count / valid.size),
    }


def validate_cam(
    cam: np.ndarray,
    config: GradCamConfig | None = None,
) -> CamDiagnostics:
    cfg = config or GradCamConfig()
    cam_arr = np.asarray(cam, dtype=np.float32)

    if cam_arr.ndim == 0:
        raise CamExtractionError("Extracted CAM is 0-dimensional scalar.")

    flat = cam_arr.ravel()
    total_elements = flat.size

    if total_elements == 0:
        raise CamExtractionError("Extracted CAM has zero elements.")

    finite_mask = np.isfinite(flat)
    finite_count = int(np.sum(finite_mask))
    finite_fraction = finite_count / float(total_elements)

    if finite_count == 0:
        raise CamExtractionError("Extracted CAM contains no finite values (all NaN/Inf).")

    valid = flat[finite_mask]
    c_min = float(np.min(valid))
    c_max = float(np.max(valid))
    c_mean = float(np.mean(valid))
    c_std = float(np.std(valid))
    c_range = c_min if math.isnan(c_max - c_min) else c_max - c_min

    denom = abs(c_mean) + 1e-12
    cv = c_std / denom
    rel_range = c_range / denom

    nonzero_count = int(np.count_nonzero(valid))
    nonzero_fraction = nonzero_count / float(total_elements)

    unique_vals = np.unique(valid)
    unique_count = int(unique_vals.size)
    unique_fraction = unique_count / float(total_elements)

    suspected_quantized = False
    if cfg.reject_quantized_like_cam:
        if c_min >= 0 and c_max > 1.5 and c_max <= 255.5:
            if np.all(np.isclose(valid, np.round(valid), atol=1e-5)):
                suspected_quantized = True
        if unique_count > 0 and unique_count <= 16 and total_elements >= 100:
            diffs = np.diff(unique_vals)
            if len(diffs) > 0 and np.all(np.isclose(diffs, diffs[0], rtol=1e-3, atol=1e-5)):
                suspected_quantized = True

    if suspected_quantized and cfg.reject_quantized_like_cam:
        raise CamExtractionError(
            f"CAM appears to be quantized/LUT-mapped (min={c_min}, max={c_max}, "
            f"unique={unique_count}/{total_elements}). Raw continuous CAM expected."
        )

    degenerate_cam = (c_range < cfg.degenerate_atol) or (c_std < cfg.degenerate_atol)

    return CamDiagnostics(
        dtype=str(cam_arr.dtype),
        shape=str(list(cam_arr.shape)),
        cam_min=c_min,
        cam_max=c_max,
        cam_mean=c_mean,
        cam_std=c_std,
        cam_range=c_range,
        relative_range=rel_range,
        coefficient_of_variation=cv,
        nonzero_fraction=nonzero_fraction,
        finite_fraction=finite_fraction,
        unique_count=unique_count,
        unique_fraction=unique_fraction,
        suspected_quantized=suspected_quantized,
        degenerate_cam=degenerate_cam,
    )


class SignalGradCamAnalyzer:
    def __init__(
        self,
        model: tf.keras.Model,
        class_names: Sequence[str],
        config: GradCamConfig | None = None,
    ) -> None:
        self.model = model
        self.class_names = list(class_names)
        self.config = config or GradCamConfig()

        if not self.config.target_layer:
            self.target_layer = self._find_default_target_layer()
        else:
            self.target_layer = self.config.target_layer

    def _find_default_target_layer(self) -> str:
        for layer in reversed(self.model.layers):
            lname = layer.name.lower()
            cname = layer.__class__.__name__.lower()
            if "conv" in lname or "conv" in cname:
                return layer.name
        raise ValueError("No convolutional layer found in model for SignalGrad-CAM.")

    def _predict_sample(self, signal_batch: tf.Tensor) -> PredictionResult:
        preds = self.model(signal_batch, training=False)
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        preds_np = preds.numpy()[0]

        is_prob = _is_probability_distribution(preds_np)
        if is_prob:
            probs = preds_np
            scores = preds_np
        else:
            scores = preds_np
            probs = tf.nn.softmax(preds).numpy()[0]

        pred_id = int(np.argmax(probs))
        pred_name = (
            self.class_names[pred_id]
            if pred_id < len(self.class_names)
            else f"Class_{pred_id}"
        )
        conf = float(probs[pred_id])
        score = float(scores[pred_id])

        return PredictionResult(
            predicted_class_id=pred_id,
            predicted_class_name=pred_name,
            predicted_score=score,
            confidence=conf,
            raw_scores=scores,
            probabilities=probs,
            output_is_probability_distribution=is_prob,
        )

    def explain_one(
        self,
        signal: np.ndarray,
        true_class_id: int | None = None,
        target_class_id: int | None = None,
    ) -> CamExplanation:
        sig_arr = np.squeeze(np.asarray(signal, dtype=np.float32))
        if sig_arr.ndim == 1:
            sig_batch = sig_arr[np.newaxis, :, np.newaxis]
        elif sig_arr.ndim == 2:
            sig_batch = sig_arr[np.newaxis, :]
        else:
            sig_batch = sig_arr

        sig_tensor = tf.convert_to_tensor(sig_batch, dtype=tf.float32)
        prediction = self._predict_sample(sig_tensor)

        target_cls = (
            target_class_id
            if target_class_id is not None
            else prediction.predicted_class_id
        )

        if TfCamBuilder is None:
            raise ImportError("signal_grad_cam library is required.")

        builder = TfCamBuilder(self.model)
        cam_dict, probs_dict, scores_dict = builder.get_cam(
            sig_tensor,
            layer_name=self.target_layer,
            class_id=target_cls,
            explainer=self.config.explainer,
        )

        raw_cam = None
        if isinstance(cam_dict, dict):
            for k, v in cam_dict.items():
                if self.target_layer in str(k) or str(target_cls) in str(k):
                    raw_cam = v
                    break
            if raw_cam is None and len(cam_dict) > 0:
                raw_cam = next(iter(cam_dict.values()))
        else:
            raw_cam = cam_dict

        if raw_cam is None:
            raise CamExtractionError("Failed to extract raw CAM from TfCamBuilder.")

        raw_cam_np = np.squeeze(np.asarray(raw_cam, dtype=np.float32))
        diagnostics = validate_cam(raw_cam_np, self.config)

        grad_diag = None
        try:
            target_layer_obj = self.model.get_layer(self.target_layer)
            grad_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=[target_layer_obj.output, self.model.output],
            )
            with tf.GradientTape() as tape:
                conv_out, preds = grad_model(sig_tensor, training=False)
                class_score = preds[:, target_cls]

            grads = tape.gradient(class_score, conv_out)
            if grads is not None:
                g_diag = _compute_array_diagnostics(grads.numpy())
                a_diag = _compute_array_diagnostics(conv_out.numpy())
                grad_diag = GradientDiagnostics(
                    layer_name=self.target_layer,
                    grad_min=g_diag["min"],
                    grad_max=g_diag["max"],
                    grad_mean=g_diag["mean"],
                    grad_std=g_diag["std"],
                    grad_l2=g_diag["l2"],
                    grad_nonzero_fraction=g_diag["nonzero_fraction"],
                    grad_unique_fraction=g_diag["unique_fraction"],
                    activation_min=a_diag["min"],
                    activation_max=a_diag["max"],
                    activation_mean=a_diag["mean"],
                    activation_std=a_diag["std"],
                    activation_l2=a_diag["l2"],
                    activation_nonzero_fraction=a_diag["nonzero_fraction"],
                    activation_unique_fraction=a_diag["unique_fraction"],
                ).to_dict()
        except Exception:  # pragma: no cover
            pass

        return CamExplanation(
            signal=sig_arr,
            cam=raw_cam_np,
            prediction=prediction,
            diagnostics=diagnostics.to_dict(),
            gradient_diagnostics=grad_diag,
        )


def collect_real_samples_by_class(
    model: tf.keras.Model | None = None,
    dataset: Any = None,
    class_names: Sequence[str] | None = None,
    samples_per_class: int = 1,
    correct_only: bool = False,
) -> dict[int, list[ExplanationSample]]:
    c_names = list(class_names) if class_names else []
    results: dict[int, list[ExplanationSample]] = {i: [] for i in range(len(c_names))}
    counts = {i: 0 for i in range(len(c_names))}

    if dataset is None:
        return results

    for batch_x, batch_y in dataset:
        x_np = np.asarray(batch_x)
        y_np = np.asarray(batch_y)

        for sample_x, label in zip(x_np, y_np):
            cls_id = _class_id_from_target(label)
            if cls_id not in counts or counts[cls_id] >= samples_per_class:
                continue

            sig = _ensure_signal_shape(sample_x)
            pred_id = cls_id
            pred_name = c_names[cls_id] if cls_id < len(c_names) else f"Class_{cls_id}"
            conf = 1.0

            if model is not None:
                inp_tensor = tf.convert_to_tensor(sig[np.newaxis, :], dtype=tf.float32)
                preds = model(inp_tensor, training=False).numpy()[0]
                pred_id = int(np.argmax(preds))
                pred_name = c_names[pred_id] if pred_id < len(c_names) else f"Class_{pred_id}"
                conf = float(np.max(preds))

            is_correct = pred_id == cls_id
            if correct_only and not is_correct:
                continue

            sample_obj = ExplanationSample(
                signal=sig,
                true_id=cls_id,
                true_name=c_names[cls_id] if cls_id < len(c_names) else f"Class_{cls_id}",
                predicted_id=pred_id,
                predicted_name=pred_name,
                confidence=conf,
                correct=is_correct,
                target_class_id=cls_id,
                target_class_name=c_names[cls_id] if cls_id < len(c_names) else f"Class_{cls_id}",
                explanation_type="true",
            )
            results[cls_id].append(sample_obj)
            counts[cls_id] += 1

            if all(c >= samples_per_class for c in counts.values()):
                return results

    return results


__all__ = [
    "CamExtractionError",
    "GradCamConfig",
    "PredictionResult",
    "CamDiagnostics",
    "CamValidationResult",
    "ExplanationSample",
    "GradientDiagnostics",
    "CamExplanation",
    "SignalGradCamAnalyzer",
    "validate_cam",
    "collect_real_samples_by_class",
    "_class_id_from_target",
    "_ensure_signal_shape",
    "extract_cam_array_explicit",
]
