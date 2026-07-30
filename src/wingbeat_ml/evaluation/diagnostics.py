"""Comprehensive Model Diagnostics Engine (Conv + Dense Layers).

Extracts layer-by-layer activations, Grad-CAM heatmaps, Dense embeddings,
and Class Contribution breakdowns for detailed model decision analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import tensorflow as tf
    from wingbeat_ml.evaluation.gradcam import compute_gradcam, find_last_conv_layer
except ImportError:  # pragma: no cover
    tf = None
    compute_gradcam = None


@dataclass
class DiagnosticResult:
    predicted_class_id: int
    predicted_confidence: float
    logits: np.ndarray
    probabilities: np.ndarray
    gradcam_heatmap: np.ndarray
    conv_features: np.ndarray
    top_conv_filters: List[int]
    dense_embedding: np.ndarray
    embedding_l2_norm: float
    class_contributions: np.ndarray  # Shape: (256, num_classes)
    top_positive_features: List[Tuple[int, float]]  # (neuron_idx, contribution_val)
    top_negative_features: List[Tuple[int, float]]


def analyze_model_sample(
    model: "tf.keras.Model",
    audio_sample: np.ndarray | "tf.Tensor",
    target_class_idx: Optional[int] = None,
) -> DiagnosticResult:
    """Perform full Conv + Dense layer diagnostic analysis on a single audio sample.

    Args:
        model: Trained Keras MosSongPlus model (logits or softmax output).
        audio_sample: 1D or 3D audio array of shape (2400,) or (1, 2400, 1).
        target_class_idx: Optional target class index to analyze contributions.

    Returns:
        DiagnosticResult containing Conv activations, Grad-CAM, Dense embeddings,
        and Class contribution breakdowns.
    """
    if tf is None:
        raise RuntimeError("TensorFlow is required for model diagnostics.")

    if isinstance(audio_sample, np.ndarray):
        if audio_sample.ndim == 1:
            inp = audio_sample.reshape(1, -1, 1).astype(np.float32)
        elif audio_sample.ndim == 2:
            inp = audio_sample.reshape(1, audio_sample.shape[0], audio_sample.shape[1]).astype(np.float32)
        else:
            inp = audio_sample.astype(np.float32)
        inp_tensor = tf.convert_to_tensor(inp)
    else:
        inp_tensor = audio_sample

    # 1. Identify key layers
    last_conv_name = find_last_conv_layer(model)
    conv_layer = model.get_layer(last_conv_name)

    # Find Dense embedding layer (penultimate Dense layer or layer before output Dense)
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    if not dense_layers:
        raise ValueError("No Dense layers found in model for embedding extraction.")

    output_dense = dense_layers[-1]
    embedding_layer = dense_layers[-2] if len(dense_layers) >= 2 else dense_layers[-1]

    # Reconstruct multi-output diagnostic sub-model
    diag_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, embedding_layer.output, output_dense.output],
    )

    conv_out, dense_emb, raw_out = diag_model(inp_tensor)

    conv_out_np = conv_out[0].numpy()
    dense_emb_np = dense_emb[0].numpy()
    raw_out_np = raw_out[0].numpy()

    # 2. Check activation & probabilities
    is_softmax = hasattr(output_dense, "activation") and getattr(output_dense.activation, "__name__", "") == "softmax"
    if is_softmax:
        probs = raw_out_np
        logits = np.log(np.maximum(probs, 1e-10))
    else:
        logits = raw_out_np
        # Softmax computation
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

    pred_class = int(np.argmax(probs)) if target_class_idx is None else target_class_idx
    pred_conf = float(probs[pred_class])

    # 3. Grad-CAM Heatmap
    heatmap, _, _ = compute_gradcam(model, inp_tensor, class_idx=pred_class)

    # 4. Conv Layer Filter Energies
    filter_energies = np.mean(np.square(conv_out_np), axis=0)  # Average energy per filter
    top_conv_filters = list(np.argsort(filter_energies)[::-1][:5])

    # 5. Dense Embedding Analysis
    l2_norm = float(np.linalg.norm(dense_emb_np))

    # 6. Class Contribution Vector (Deconstruction of Output Logits)
    weights, biases = output_dense.get_weights()
    # weights shape: (embedding_dim, num_classes)
    # dense_emb_np shape: (embedding_dim,)
    class_contributions = weights * dense_emb_np[:, np.newaxis]  # (256, num_classes)

    target_contribs = class_contributions[:, pred_class]
    sorted_indices = np.argsort(target_contribs)

    top_pos = [(int(idx), float(target_contribs[idx])) for idx in sorted_indices[::-1][:5]]
    top_neg = [(int(idx), float(target_contribs[idx])) for idx in sorted_indices[:5]]

    return DiagnosticResult(
        predicted_class_id=pred_class,
        predicted_confidence=pred_conf,
        logits=logits,
        probabilities=probs,
        gradcam_heatmap=heatmap,
        conv_features=conv_out_np,
        top_conv_filters=top_conv_filters,
        dense_embedding=dense_emb_np,
        embedding_l2_norm=l2_norm,
        class_contributions=class_contributions,
        top_positive_features=top_pos,
        top_negative_features=top_neg,
    )
