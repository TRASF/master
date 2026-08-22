"""Model analysis module: standard 1D/2D Grad-CAM interpretability and heatmap calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None


@dataclass
class GradCamResult:
    """Structured result containing analytical raw CAM and normalized display CAM."""

    raw_cam: np.ndarray          # Shape: [B, T]
    display_cam: np.ndarray      # Shape: [B, T]
    raw_min: np.ndarray          # Shape: [B]
    raw_max: np.ndarray          # Shape: [B]
    raw_mean: np.ndarray         # Shape: [B]
    raw_l1: np.ndarray           # Shape: [B]
    raw_l2: np.ndarray           # Shape: [B]
    degenerate_heatmap: np.ndarray  # Shape: [B] (bool)
    class_idx: Any               # int or int array
    confidence: Any              # float or float array

    def __iter__(self):
        yield self.display_cam
        yield self.class_idx
        yield self.confidence

    def __getitem__(self, key: Any) -> Any:
        if key == 0:
            return self.display_cam
        elif key == 1:
            return self.class_idx
        elif key == 2:
            return self.confidence
        elif isinstance(key, str):
            return getattr(self, key)
        raise KeyError(key)


def find_last_conv_layer(model: Any) -> str:
    """Find the name of the last Conv1D or Conv2D layer in the Keras model."""
    for layer in reversed(getattr(model, "layers", [])):
        if "conv" in layer.name.lower():
            return layer.name
    raise ValueError("No convolutional layer found in model.")


def _extract_logits_and_conv(
    model: Any,
    inputs: Any,
    layer_name: str,
) -> Tuple[Any, Any, Any]:
    """Extract conv outputs, raw pre-softmax logits z, and softmax probabilities."""
    last_layer = model.layers[-1]
    is_softmax = False

    if hasattr(last_layer, "activation") and getattr(last_layer.activation, "__name__", "") == "softmax":
        is_softmax = True

    conv_layer = model.get_layer(layer_name)

    if is_softmax:
        if isinstance(last_layer, tf.keras.layers.Dense):
            submodel = tf.keras.Model(
                inputs=model.inputs,
                outputs=[conv_layer.output, last_layer.input],
            )
            conv_out, h = submodel(inputs)
            w = last_layer.kernel
            b = last_layer.bias if last_layer.use_bias else 0.0
            logits = tf.matmul(h, w) + b
            probs = tf.nn.softmax(logits)
            return conv_out, logits, probs
        elif isinstance(last_layer, (tf.keras.layers.Softmax, tf.keras.layers.Activation)):
            submodel = tf.keras.Model(
                inputs=model.inputs,
                outputs=[conv_layer.output, last_layer.input],
            )
            conv_out, logits = submodel(inputs)
            probs = tf.nn.softmax(logits)
            return conv_out, logits, probs
        else:
            raise ValueError(
                "Grad-CAM target score must be pre-softmax score z_k. "
                "Unable to extract pre-softmax logits from model architecture."
            )
    else:
        submodel = tf.keras.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output],
        )
        conv_out, logits = submodel(inputs)
        probs = tf.nn.softmax(logits)
        return conv_out, logits, probs


def compute_gradcam(
    model: Any,
    inputs: np.ndarray | Any,
    class_idx: Optional[int] = None,
    layer_name: Optional[str] = None,
) -> GradCamResult:
    """Compute 1D/2D Grad-CAM heatmap for single sample or batch."""
    if tf is None:
        raise RuntimeError("TensorFlow is required for Grad-CAM computation.")

    if not isinstance(inputs, tf.Tensor):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

    if len(inputs.shape) == 1:
        inputs = tf.expand_dims(tf.expand_dims(inputs, axis=0), axis=-1)
    elif len(inputs.shape) == 2:
        inputs = tf.expand_dims(inputs, axis=-1)

    batch_size = tf.shape(inputs)[0]
    target_T = tf.shape(inputs)[1]

    if layer_name is None:
        layer_name = find_last_conv_layer(model)

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        conv_outputs, logits, probs = _extract_logits_and_conv(
            model, inputs, layer_name
        )

        if class_idx is None:
            top_classes = tf.argmax(probs, axis=-1)
        else:
            top_classes = tf.fill([batch_size], tf.cast(class_idx, tf.int64))

        sample_indices = tf.range(batch_size, dtype=tf.int64)
        gather_indices = tf.stack([sample_indices, top_classes], axis=1)
        target_scores = tf.gather_nd(logits, gather_indices)

    grads = tape.gradient(target_scores, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=1)

    raw_cam = tf.reduce_sum(
        conv_outputs * pooled_grads[:, tf.newaxis, :], axis=-1
    )
    raw_cam = tf.maximum(raw_cam, 0.0)

    conv_T = tf.shape(conv_outputs)[1]

    if conv_T != target_T:
        cam_4d = tf.reshape(raw_cam, [batch_size, 1, conv_T, 1])
        upsampled_4d = tf.image.resize(
            cam_4d, [1, target_T], method="bilinear"
        )
        raw_cam_aligned = tf.reshape(upsampled_4d, [batch_size, target_T])
    else:
        raw_cam_aligned = raw_cam

    raw_cam_np = raw_cam_aligned.numpy()
    raw_min = np.min(raw_cam_np, axis=-1)
    raw_max = np.max(raw_cam_np, axis=-1)
    raw_mean = np.mean(raw_cam_np, axis=-1)
    raw_l1 = np.sum(np.abs(raw_cam_np), axis=-1)
    raw_l2 = np.sqrt(np.sum(np.square(raw_cam_np), axis=-1))

    display_cam_np = np.zeros_like(raw_cam_np)
    degenerate = np.zeros(batch_size.numpy(), dtype=bool)

    for i in range(batch_size.numpy()):
        cam_range = raw_max[i] - raw_min[i]
        if cam_range <= 1e-8:
            display_cam_np[i] = 0.0
            degenerate[i] = True
        else:
            display_cam_np[i] = (raw_cam_np[i] - raw_min[i]) / cam_range
            degenerate[i] = False

    top_classes_np = top_classes.numpy()
    probs_np = probs.numpy()
    conf_np = np.array([float(probs_np[i, top_classes_np[i]]) for i in range(batch_size.numpy())])

    res_class = int(top_classes_np[0]) if batch_size.numpy() == 1 and class_idx is not None else (
        int(top_classes_np[0]) if batch_size.numpy() == 1 else top_classes_np.tolist()
    )
    res_conf = float(conf_np[0]) if batch_size.numpy() == 1 else conf_np.tolist()

    return GradCamResult(
        raw_cam=raw_cam_np,
        display_cam=display_cam_np,
        raw_min=raw_min[0] if batch_size.numpy() == 1 else raw_min,
        raw_max=raw_max[0] if batch_size.numpy() == 1 else raw_max,
        raw_mean=raw_mean[0] if batch_size.numpy() == 1 else raw_mean,
        raw_l1=raw_l1[0] if batch_size.numpy() == 1 else raw_l1,
        raw_l2=raw_l2[0] if batch_size.numpy() == 1 else raw_l2,
        degenerate_heatmap=degenerate[0] if batch_size.numpy() == 1 else degenerate,
        class_idx=res_class,
        confidence=res_conf,
    )


def aggregate_raw_cams(
    raw_cams: Sequence[np.ndarray] | np.ndarray,
) -> Dict[str, Any]:
    """Aggregate raw analytical CAMs across multiple samples."""
    cams = np.asarray(raw_cams, dtype=np.float32)
    if cams.ndim == 3 and cams.shape[1] == 1:
        cams = cams[:, 0, :]
    elif cams.ndim == 1:
        cams = cams[np.newaxis, :]
    if cams.ndim != 2:
        raise ValueError(f"Expected 2D array of raw CAMs [N, T], got shape {cams.shape}")

    count = int(cams.shape[0])
    mean = np.mean(cams, axis=0)
    variance = np.var(cams, axis=0)
    std = np.std(cams, axis=0)
    median = np.median(cams, axis=0)
    q25 = np.percentile(cams, 25, axis=0)
    q75 = np.percentile(cams, 75, axis=0)

    return {
        "count": count,
        "mean": mean,
        "variance": variance,
        "std": std,
        "median": median,
        "quantiles": {
            0.25: q25,
            0.75: q75,
        },
    }


__all__ = [
    "GradCamResult",
    "find_last_conv_layer",
    "compute_gradcam",
    "aggregate_raw_cams",
]
