"""Deep Misclassification Diagnostic Visual Tool.

Generates a detailed 1-page root-cause analysis report explaining WHY an audio sample
was misclassified by comparing True Class vs Predicted False Class features.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_repo_src = Path(__file__).resolve().parent.parent / "src"
if _repo_src.exists() and str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))

try:
    import tensorflow as tf
    from wingbeat_ml.classification.models import MosSongPlusModel
    from wingbeat_ml.classification.evaluation.error_analyzer import diagnose_misclassification
    from wingbeat_ml.visualizer.spectrogram import compute_spectrogram
    from wingbeat_ml.data.audio import load_audio
except ImportError as err:  # pragma: no cover
    print(f"Warning: Failed to import dependencies: {err}")
    tf = None

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
        raise FileNotFoundError(f"Weights file not found: {weights_p}")

    cfg_path = Path("configs/models/mossong_plus.yaml")
    if not cfg_path.exists():
        cfg_path = _repo_src.parent / "configs/models/mossong_plus.yaml"

    with open(cfg_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    builder = MosSongPlusModel(model_cfg)
    model = builder.build(input_shape=(2400, 1), output_units=11, output_activation=None)
    model.load_weights(str(weights_p))
    return model


def generate_failure_report(
    audio: np.ndarray,
    model: tf.keras.Model,
    true_class_id: int,
    sample_rate: int = 8000,
    out_path: str = "output/misclassification_diagnosis.png",
) -> None:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.2, 1.2], width_ratios=[1.8, 1.2])

    diag = diagnose_misclassification(model, audio, true_class_id=true_class_id, sample_rate=sample_rate)

    true_name = CLASS_NAMES[diag.true_class_id]
    pred_name = CLASS_NAMES[diag.predicted_class_id]

    fig.suptitle(
        f"MISCLASSIFICATION DIAGNOSTIC REPORT\n"
        f"True: {true_name} ({diag.true_class_confidence*100:.1f}%)  │  Predicted False: {pred_name} ({diag.predicted_class_confidence*100:.1f}%)\n"
        f"Root Cause: {diag.primary_failure_reason}",
        fontsize=14,
        fontweight="bold",
        color="#ff4d73" if diag.predicted_class_id != diag.true_class_id else "#00d6b4",
    )

    # Subplot 1: Input Spectrogram
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
    ax_spec.set_title(f"[INPUT STFT SPECTROGRAM] f0={diag.f0_hz:.1f} Hz | SNR={diag.snr_db:.1f} dB")
    ax_spec.set_ylabel("Frequency (Hz)")
    fig.colorbar(im_spec, ax=ax_spec, pad=0.01)

    # Subplot 2: Diagnostic Metrics Summary Card
    ax_card = fig.add_subplot(grid[0, 1])
    ax_card.axis("off")
    card_text = (
        f"═══ DIAGNOSTIC SUMMARY ═══\n\n"
        f"• Ground Truth Class  : {true_name}\n"
        f"• Model Prediction    : {pred_name}\n"
        f"• Logit Margin (Δz)   : {diag.logit_margin:.3f}\n"
        f"• True Class Conf     : {diag.true_class_confidence*100:.2f}%\n"
        f"• Pred Class Conf     : {diag.predicted_class_confidence*100:.2f}%\n"
        f"• Wingbeat SNR        : {diag.snr_db:.1f} dB\n"
        f"• Dominant f0         : {diag.f0_hz:.1f} Hz\n\n"
        f"PRIMARY FAILURE CAUSE:\n"
        f"► {diag.primary_failure_reason}"
    )
    ax_card.text(
        0.05,
        0.95,
        card_text,
        transform=ax_card.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
        family="monospace",
        bbox={"facecolor": "#151515", "edgecolor": "#ff4d73", "pad": 8},
    )

    # Subplot 3: True Class Grad-CAM Heatmap
    ax_true_cam = fig.add_subplot(grid[1, 0])
    norm_spec = np.clip((spec - (-100.0)) / ((-20.0) - (-100.0)), 0.0, 1.0)
    
    h_true = diag.true_class_heatmap
    if len(h_true) != norm_spec.shape[1]:
        h_true = np.interp(np.linspace(0, len(h_true) - 1, norm_spec.shape[1]), np.arange(len(h_true)), h_true)
    cam_true_2d = norm_spec * h_true[np.newaxis, :]

    im_true = ax_true_cam.imshow(
        cam_true_2d,
        origin="lower",
        aspect="auto",
        extent=[0, len(audio) / sample_rate, 0, sample_rate / 2],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax_true_cam.set_title(f"[TRUE CLASS FEATURE ATTENTION] Expected: {true_name}", color="#00d6b4")
    ax_true_cam.set_ylabel("Frequency (Hz)")
    fig.colorbar(im_true, ax=ax_true_cam, pad=0.01)

    # Subplot 4: False Predicted Class Grad-CAM Heatmap
    ax_pred_cam = fig.add_subplot(grid[1, 1])
    h_pred = diag.predicted_class_heatmap
    if len(h_pred) != norm_spec.shape[1]:
        h_pred = np.interp(np.linspace(0, len(h_pred) - 1, norm_spec.shape[1]), np.arange(len(h_pred)), h_pred)
    cam_pred_2d = norm_spec * h_pred[np.newaxis, :]

    im_pred = ax_pred_cam.imshow(
        cam_pred_2d,
        origin="lower",
        aspect="auto",
        extent=[0, len(audio) / sample_rate, 0, sample_rate / 2],
        cmap="plasma",
        vmin=0.0,
        vmax=1.0,
    )
    ax_pred_cam.set_title(f"[FALSE PREDICTION ATTENTION] Triggered: {pred_name}", color="#ff4d73")
    fig.colorbar(im_pred, ax=ax_pred_cam, pad=0.01)

    # Subplot 5: Differential Attention Heatmap (Tricked Region)
    ax_diff = fig.add_subplot(grid[2, 0])
    h_diff = diag.differential_heatmap
    if len(h_diff) != norm_spec.shape[1]:
        h_diff = np.interp(np.linspace(0, len(h_diff) - 1, norm_spec.shape[1]), np.arange(len(h_diff)), h_diff)
    cam_diff_2d = norm_spec * h_diff[np.newaxis, :]

    im_diff = ax_diff.imshow(
        cam_diff_2d,
        origin="lower",
        aspect="auto",
        extent=[0, len(audio) / sample_rate, 0, sample_rate / 2],
        cmap="hot",
        vmin=0.0,
        vmax=1.0,
    )
    ax_diff.set_title("[DIFFERENTIAL HEATMAP] Exact Time-Frequency Region Tricking Model", color="#f2c94c")
    ax_diff.set_xlabel("Time (seconds)")
    ax_diff.set_ylabel("Frequency (Hz)")
    fig.colorbar(im_diff, ax=ax_diff, pad=0.01)

    # Subplot 6: Contradictory Neuron Differentials
    ax_neurons = fig.add_subplot(grid[2, 1])
    n_indices = [f"N#{item[0]}" for item in diag.top_contradictory_neurons]
    true_vals = [item[1] for item in diag.top_contradictory_neurons]
    pred_vals = [item[2] for item in diag.top_contradictory_neurons]

    x = np.arange(len(n_indices))
    width = 0.35

    ax_neurons.bar(x - width/2, true_vals, width, label=f"True ({true_name[:8]})", color="#00d6b4")
    ax_neurons.bar(x + width/2, pred_vals, width, label=f"Pred ({pred_name[:8]})", color="#ff4d73")
    ax_neurons.set_xticks(x)
    ax_neurons.set_xticklabels(n_indices, fontsize=9)
    ax_neurons.set_title("Top Contradictory Neurons (Logit Contribution)", color="#ff4d73")
    ax_neurons.legend(fontsize=8)

    fig.tight_layout()
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Generated misclassification diagnostic report: {out_file.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Deep Misclassification Diagnostic Inspector")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights (.weights.h5)")
    parser.add_argument("--audio", type=str, default=None, help="Path to failed audio file (.wav/.npy)")
    parser.add_argument("--true-class-id", type=int, default=0, help="True ground-truth class index (0-10)")
    parser.add_argument("--out", type=str, default="output/misclassification_diagnosis.png", help="Output PNG path")
    args = parser.parse_args()

    model = load_analysis_model(args.model)

    if args.audio and Path(args.audio).exists():
        audio = load_audio(args.audio, target_sample_rate=8000)
        if len(audio) < 2400:
            audio = np.pad(audio, (0, 2400 - len(audio)))
        else:
            audio = audio[:2400]
    else:
        # Synthetic ambiguous signal
        t = np.linspace(0, 0.3, 2400, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 380.0 * t) + 0.3 * np.random.randn(2400)).astype(np.float32)

    generate_failure_report(audio, model, true_class_id=args.true_class_id, out_path=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
