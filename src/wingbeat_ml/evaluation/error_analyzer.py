"""In-Depth Model Misclassification & Root-Cause Error Analyzer.

Diagnoses WHY specific audio samples failed or were misclassified by examining:
1. Logit Margin Penalties & Class Probabilities
2. Dense Embedding Contradictory Neuron Differentials
3. Time-Frequency Spectral Noise & Harmonic SNR
4. Differential Grad-CAM Attention Heatmaps (False Class vs True Class)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import tensorflow as tf
    from wingbeat_ml.evaluation.gradcam import compute_gradcam, find_last_conv_layer
    from wingbeat_ml.visualizer.spectrogram import compute_spectrogram, analyze_harmonics
except ImportError:  # pragma: no cover
    tf = None
    compute_gradcam = None
    analyze_harmonics = None


@dataclass
class FailureDiagnosis:
    true_class_id: int
    predicted_class_id: int
    true_class_confidence: float
    predicted_class_confidence: float
    logit_margin: float  # z_pred - z_true
    primary_failure_reason: str  # e.g., "Harmonic Overlap", "Low SNR Noise", "Dense Neuron Bias"
    f0_hz: float
    snr_db: float
    true_class_heatmap: np.ndarray
    predicted_class_heatmap: np.ndarray
    differential_heatmap: np.ndarray  # Heatmap of regions driving false prediction
    top_contradictory_neurons: List[Tuple[int, float, float]]  # (neuron_idx, true_contrib, pred_contrib)


def diagnose_misclassification(
    model: "tf.keras.Model",
    audio_sample: np.ndarray,
    true_class_id: int,
    sample_rate: int = 8000,
) -> FailureDiagnosis:
    """Diagnose root cause of misclassification for a single audio sample.

    Args:
        model: Trained Keras MosSongPlus model.
        audio_sample: 1D float32 audio array (2400 samples).
        true_class_id: Ground truth class index.
        sample_rate: Sampling rate (default: 8000 Hz).

    Returns:
        FailureDiagnosis containing root-cause analysis and layer differentials.
    """
    if tf is None:
        raise RuntimeError("TensorFlow is required for model error analysis.")

    audio_sample = audio_sample - np.mean(audio_sample)
    peak = np.max(np.abs(audio_sample))
    if peak > 1e-6:
        audio_norm = (audio_sample / peak) * 0.95
    else:
        audio_norm = audio_sample

    inp_tensor = tf.convert_to_tensor(audio_norm.reshape(1, 2400, 1).astype(np.float32))

    # 1. Layer Extractors
    last_conv_name = find_last_conv_layer(model)
    conv_layer = model.get_layer(last_conv_name)
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    output_dense = dense_layers[-1]
    embedding_layer = dense_layers[-2] if len(dense_layers) >= 2 else dense_layers[-1]

    diag_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, embedding_layer.output, output_dense.output],
    )

    conv_out, dense_emb, raw_out = diag_model(inp_tensor)
    dense_emb_np = dense_emb[0].numpy()
    raw_out_np = raw_out[0].numpy()

    is_softmax = hasattr(output_dense, "activation") and getattr(output_dense.activation, "__name__", "") == "softmax"
    if is_softmax:
        probs = raw_out_np
        logits = np.log(np.maximum(probs, 1e-10))
    else:
        logits = raw_out_np
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

    pred_class_id = int(np.argmax(probs))
    pred_conf = float(probs[pred_class_id])
    true_conf = float(probs[true_class_id])
    logit_margin = float(logits[pred_class_id] - logits[true_class_id])

    # 2. Dual Grad-CAM Heatmaps (True vs False Prediction)
    cam_true, _, _ = compute_gradcam(model, inp_tensor, class_idx=true_class_id)
    cam_pred, _, _ = compute_gradcam(model, inp_tensor, class_idx=pred_class_id)
    diff_heatmap = np.maximum(cam_pred - cam_true, 0.0)

    # 3. Dense Neuron Contribution Differential
    weights, _ = output_dense.get_weights()  # (embedding_dim, num_classes)
    true_contribs = weights[:, true_class_id] * dense_emb_np
    pred_contribs = weights[:, pred_class_id] * dense_emb_np
    differentials = pred_contribs - true_contribs  # Positive means pushed toward wrong class

    top_contradictory_indices = np.argsort(differentials)[::-1][:5]
    top_contradictory_neurons = [
        (int(idx), float(true_contribs[idx]), float(pred_contribs[idx]))
        for idx in top_contradictory_indices
    ]

    # 4. Spectral Noise & Harmonic Analysis
    harmonics = analyze_harmonics(audio_norm, sample_rate=sample_rate)
    f0_hz = harmonics["f0_hz"]

    freqs, times, spec = compute_spectrogram(audio_norm, sample_rate=sample_rate)
    wingbeat_mask = (freqs >= 150.0) & (freqs <= 1200.0)
    noise_mask = ~wingbeat_mask

    wingbeat_power = np.mean(spec[wingbeat_mask, :]) if np.any(wingbeat_mask) else -100.0
    noise_power = np.mean(spec[noise_mask, :]) if np.any(noise_mask) else -100.0
    snr_db = float(wingbeat_power - noise_power)

    # 5. Root Cause Classification Logic
    if snr_db < 5.0:
        primary_reason = "Low SNR / High Broadband Noise Interference"
    elif f0_hz < 180.0 or f0_hz > 1000.0:
        primary_reason = "Weak or Out-of-Bound Fundamental Frequency (f0)"
    elif logit_margin < 2.0:
        primary_reason = "High Species Boundary Ambiguity (Harmonic Overlap)"
    else:
        primary_reason = "Dense Neuron Feature Bias (Over-activation)"

    return FailureDiagnosis(
        true_class_id=true_class_id,
        predicted_class_id=pred_class_id,
        true_class_confidence=true_conf,
        predicted_class_confidence=pred_conf,
        logit_margin=logit_margin,
        primary_failure_reason=primary_reason,
        f0_hz=f0_hz,
        snr_db=snr_db,
        true_class_heatmap=cam_true,
        predicted_class_heatmap=cam_pred,
        differential_heatmap=diff_heatmap,
        top_contradictory_neurons=top_contradictory_neurons,
    )
