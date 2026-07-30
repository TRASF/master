"""Grad-CAM explainability module for TF/Keras audio models.

Computes class activation heatmaps showing which time-frequency regions
drive model classification decisions.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None


def find_last_conv_layer(model: "tf.keras.Model") -> str:
    """Find the name of the last Conv1D or Conv2D layer in the Keras model."""
    for layer in reversed(model.layers):
        if "conv" in layer.name.lower():
            return layer.name
    raise ValueError("No convolutional layer found in model.")


def compute_gradcam(
    model: "tf.keras.Model",
    inputs: np.ndarray | "tf.Tensor",
    class_idx: Optional[int] = None,
    layer_name: Optional[str] = None,
) -> Tuple[np.ndarray, int, float]:
    """Compute 1D/2D Grad-CAM heatmap for a single sample or batch.

    Args:
        model: Trained Keras model.
        inputs: Input audio tensor / array of shape (1, T, 1) or (1, T).
        class_idx: Target class index. If None, uses top predicted class.
        layer_name: Target conv layer name. If None, auto-detects last conv layer.

    Returns:
        Tuple of (heatmap array normalized [0, 1], predicted_class_idx, confidence)
    """
    if tf is None:
        raise RuntimeError("TensorFlow is required for Grad-CAM computation.")

    if not isinstance(inputs, tf.Tensor):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

    if len(inputs.shape) == 2:
        inputs = tf.expand_dims(inputs, axis=-1)

    if layer_name is None:
        layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        conv_outputs, predictions = grad_model(inputs)
        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]))
        confidence = float(predictions[0][class_idx])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    # ponytail: channel-wise average Pooling for weights
    pooled_grads = tf.reduce_mean(grads, axis=tuple(range(len(grads.shape) - 1)))
    conv_outputs = conv_outputs[0]

    # Weighted combination of feature maps
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0.0)

    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    return heatmap.numpy(), class_idx, confidence
