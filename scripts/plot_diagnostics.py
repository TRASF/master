"""Diagnostic Visualization Script.

Plots audio waveform/spectrogram, ground truth event intervals, and predicted p(mosquito)
timeline to visually inspect frame and time alignment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from wingbeat_ml.data.synthetic import load_wav_pcm
from wingbeat_ml.sed.training.train import FullSEDTeacher


def generate_diagnostic_plot(
    wav_path: str | Path,
    start_gt_s: float = 0.0,
    end_gt_s: float = 2.5,
    output_png: str | Path = "MosSongPlus/output/diagnostic_plot.png",
    model_checkpoint: str | Path | None = None,
) -> None:
    """Generate diagnostic plot comparing waveform, ground truth, and predicted probabilities."""
    signal, sr = load_wav_pcm(wav_path)
    dur_s = len(signal) / float(sr)
    time_axis = np.linspace(0.0, dur_s, len(signal))

    # Run model forward pass to get p(t)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullSEDTeacher(sample_rate=sr).to(device)
    if model_checkpoint and Path(model_checkpoint).exists():
        model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.eval()

    with torch.no_grad():
        audio_tensor = torch.from_numpy(signal).float().unsqueeze(0).to(device)
        probs = model(audio_tensor).squeeze(0).squeeze(-1).cpu().numpy()

    frame_time_axis = np.linspace(0.0, dur_s, len(probs))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    # 1. Waveform
    ax1.plot(time_axis, signal, color="gray", alpha=0.7)
    ax1.axvspan(start_gt_s, end_gt_s, color="green", alpha=0.3, label="Ground Truth Event")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Audio Waveform")
    ax1.legend(loc="upper right")

    # 2. Log-Mel Spectrogram
    ax2.specgram(signal, Fs=sr, NFFT=512, noverlap=256, cmap="magma")
    ax2.axvspan(start_gt_s, end_gt_s, color="green", alpha=0.3)
    ax2.set_ylabel("Freq (Hz)")
    ax2.set_title("Spectrogram")

    # 3. Predicted Probability p(mosquito)
    ax3.plot(frame_time_axis, probs, color="crimson", linewidth=2, label="Predicted p(mosquito)")
    ax3.axhline(0.8, color="black", linestyle="--", alpha=0.5, label="High Threshold (0.8)")
    ax3.axhline(0.4, color="black", linestyle=":", alpha=0.5, label="Low Threshold (0.4)")
    ax3.axvspan(start_gt_s, end_gt_s, color="green", alpha=0.3)
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Probability")
    ax3.set_title("Predicted Probability p(mosquito)")
    ax3.legend(loc="upper right")

    plt.tight_layout()
    out_p = Path(output_png)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=150)
    plt.close()
    print(f"Saved diagnostic plot to {out_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", default="MosSongPlus/dataset/synthetic/audio/synth_000000.wav")
    parser.add_argument("--start", type=float, default=0.18)
    parser.add_argument("--end", type=float, default=2.68)
    parser.add_argument("--output", default="MosSongPlus/output/diagnostic_plot.png")
    args = parser.parse_args()

    generate_diagnostic_plot(args.wav, args.start, args.end, args.output)
