"""Offline Sample Model Diagnostics & Visual Inspection Tool.

Generates a complete 1-page visual analysis report for Conv + Dense layers
for any given audio file or synthetic mosquito signal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure 'src' is in sys.path for direct script execution
_repo_src = Path(__file__).resolve().parent.parent / "src"
if _repo_src.exists() and str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))

try:
    import tensorflow as tf
    from wingbeat_ml.models import MosSongPlusModel
    from wingbeat_ml.evaluation.diagnostics import analyze_model_sample
    from wingbeat_ml.visualizer.spectrogram import compute_spectrogram, analyze_harmonics
    from wingbeat_ml.data.audio import load_audio
except ImportError as err:
    print(f"Warning: Failed to import wingbeat_ml components: {err}")
    tf = None
    MosSongPlusModel = None
    analyze_model_sample = None

CLASS_NAMES = [
    "Ae_aegypti_Female",
    "Ae_aegypti_Male",
    "Ae_albopictus_Female",
    "Ae_albopictus_Male",
    "An_dirus_Female",
    "An_dirus_Male",
    "An_minimus_Female",
    "An_minimus_Male",
    "Cx_quin_Female",
    "Cx_quin_Male",
    "No_Mos",
]


def load_analysis_model(weights_path: str) -> tf.keras.Model:
    import yaml
    weights_p = Path(weights_path).resolve()
    if not weights_p.exists():
        raise FileNotFoundError(
            f"Weights file not found at: '{weights_path}'\n"
            f"Absolute path checked: '{weights_p}'\n"
            f"Current directory: '{Path.cwd()}'\n"
            "Please verify the model weights file path."
        )

    cfg_path = Path("configs/models/mossong_plus.yaml")
    if not cfg_path.exists():
        cfg_path = _repo_src.parent / "configs/models/mossong_plus.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    builder = MosSongPlusModel(model_cfg)
    model = builder.build(input_shape=(2400, 1), output_units=11, output_activation=None)
    model.load_weights(str(weights_p))
    return model


def generate_diagnostic_report(
    audio: np.ndarray,
    model: tf.keras.Model,
    sample_rate: int = 8000,
    out_path: str = "output/sample_diagnostic_report.png",
) -> None:
    """Generate 1-page diagnostic visual report PNG."""
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.0], width_ratios=[2.0, 1.0])

    # Audio Preprocessing
    audio = audio - np.mean(audio)
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = (audio / peak) * 0.95

    inp_tensor = audio.reshape(1, 2400, 1).astype(np.float32)

    # 1. Run Diagnostics
    diag = analyze_model_sample(model, inp_tensor)
    harmonics = analyze_harmonics(audio, sample_rate=sample_rate)

    pred_cls_name = CLASS_NAMES[diag.predicted_class_id]
    if pred_cls_name.startswith("Ae"):
        class_color = "#00d6b4"
    elif pred_cls_name.startswith("An"):
        class_color = "#ff4d73"
    elif pred_cls_name.startswith("Cx"):
        class_color = "#f2c94c"
    else:
        class_color = "#808080"

    fig.suptitle(f"MOSQUITOSONG+ DUAL-ENGINE DIAGNOSTIC REPORT — {pred_cls_name} ({diag.predicted_confidence*100:.1f}%)", fontsize=15, fontweight="bold", color=class_color)

    # Subplot 1: STFT Spectrogram
    ax_spec = fig.add_subplot(grid[0, 0])
    freqs, times, spec = compute_spectrogram(audio, sample_rate=sample_rate)
    im_spec = ax_spec.imshow(
        spec,
        origin="lower",
        aspect="auto",
        extent=[0, len(audio) / sample_rate, 0, sample_rate / 2],
        cmap="inferno",
        vmin=-100,
        vmax=-20,
    )
    ax_spec.set_title(f"[INPUT SIGNAL] STFT Spectrogram (Dominant f0: {harmonics['f0_hz']:.1f} Hz, Power: {harmonics['peak_power_db']:.1f} dBFS)")
    ax_spec.set_ylabel("Frequency (Hz)")
    fig.colorbar(im_spec, ax=ax_spec, pad=0.01)

    # Subplot 2: Grad-CAM Attention Heatmap (PSD-weighted Time-Frequency Attention)
    ax_cam = fig.add_subplot(grid[1, 0])
    norm_spec = np.clip((spec - (-100.0)) / ((-20.0) - (-100.0)), 0.0, 1.0)
    
    # Resample 1D heatmap over STFT time columns
    if len(diag.gradcam_heatmap) != norm_spec.shape[1]:
        h_interp = np.interp(
            np.linspace(0, len(diag.gradcam_heatmap) - 1, norm_spec.shape[1]),
            np.arange(len(diag.gradcam_heatmap)),
            diag.gradcam_heatmap,
        )
    else:
        h_interp = diag.gradcam_heatmap

    cam_2d = norm_spec * h_interp[np.newaxis, :]

    im_cam = ax_cam.imshow(
        cam_2d,
        origin="lower",
        aspect="auto",
        extent=[0, len(audio) / sample_rate, 0, sample_rate / 2],
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
    )
    ax_cam.set_title(f"[HOST CONV ANALYSIS] Grad-CAM Frequency Attention Map — {pred_cls_name}", color=class_color)
    ax_cam.set_xlabel("Time (seconds)")
    ax_cam.set_ylabel("Frequency (Hz)")
    fig.colorbar(im_cam, ax=ax_cam, pad=0.01)

    # Subplot 3: Class Probabilities Bar Chart
    ax_prob = fig.add_subplot(grid[0:2, 1])
    y_pos = np.arange(len(CLASS_NAMES))
    colors = [class_color if i == diag.predicted_class_id else "#404040" for i in range(len(CLASS_NAMES))]
    ax_prob.barh(y_pos, diag.probabilities * 100.0, color=colors, align="center")
    ax_prob.set_yticks(y_pos)
    ax_prob.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax_prob.invert_yaxis()
    ax_prob.set_xlabel("Confidence (%)")
    ax_prob.set_title("[HOST CLASSIFICATION] Output Class Confidence", color=class_color)
    ax_prob.set_xlim(0, 100)

    # Subplot 4: Dense Embedding Activations
    ax_emb = fig.add_subplot(grid[2, 0])
    emb_dim = len(diag.dense_embedding)
    ax_emb.plot(diag.dense_embedding, color=class_color, linewidth=1.2)
    ax_emb.set_title(f"[HOST DENSE FEATURE LAYER] Bottleneck Embedding ({emb_dim}-dim, L2 Norm: {diag.embedding_l2_norm:.2f})", color=class_color)
    ax_emb.set_xlabel(f"Neuron Index (0-{emb_dim - 1})")
    ax_emb.set_ylabel("Activation")
    ax_emb.grid(True, alpha=0.2)

    # Subplot 5: Top Neuron Contributions to Output Class
    ax_contrib = fig.add_subplot(grid[2, 1])
    top_pos_indices = [item[0] for item in diag.top_positive_features]
    top_pos_values = [item[1] for item in diag.top_positive_features]

    ax_contrib.bar(range(len(top_pos_indices)), top_pos_values, color=class_color)
    ax_contrib.set_xticks(range(len(top_pos_indices)))
    ax_contrib.set_xticklabels([f"N#{idx}" for idx in top_pos_indices], fontsize=9)
    ax_contrib.set_title("Top Positive Neuron Contributions", color=class_color)
    ax_contrib.set_ylabel("Logit Impact")

    fig.tight_layout()
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Generated diagnostic report: {out_file.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline Model Diagnostics Inspector")
    parser.add_argument("--audio", type=str, default=None, help="Path to input audio file (.wav/.npy)")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights (.weights.h5)")
    parser.add_argument("--out", type=str, default="output/sample_diagnostic_report.png", help="Output PNG path")
    args = parser.parse_args()

    model = load_analysis_model(args.model)

    if args.audio and Path(args.audio).exists():
        audio = load_audio(args.audio, target_sample_rate=8000)
        if len(audio) < 2400:
            audio = np.pad(audio, (0, 2400 - len(audio)))
        else:
            audio = audio[:2400]
    else:
        # Synthetic mosquito wingbeat tone (450 Hz)
        t = np.linspace(0, 0.3, 2400, endpoint=False)
        audio = (0.8 * np.sin(2 * np.pi * 450.0 * t)).astype(np.float32)

    generate_diagnostic_report(audio, model, out_path=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
