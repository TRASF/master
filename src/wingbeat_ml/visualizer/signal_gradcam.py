"""Visualization helpers for validated SignalGrad-CAM outputs."""

from __future__ import annotations

from typing import Optional, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np

from wingbeat_ml.analysis.model.signal_gradcam import CamDiagnostics


def _normalize_cam_for_display(cam: np.ndarray) -> np.ndarray:
    """Normalize only for plotting; never use this result for diagnostics."""
    x = np.asarray(cam, dtype=np.float64)
    lo = float(np.min(x))
    hi = float(np.max(x))

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("CAM contains non-finite values.")

    if np.isclose(lo, hi, rtol=0.0, atol=1e-12):
        return np.zeros_like(x, dtype=np.float64)

    return (x - lo) / (hi - lo)


def plot_signal_gradcam(
    signal: np.ndarray,
    cam_heatmap: np.ndarray,
    sr: int = 8000,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    diagnostics: Optional[CamDiagnostics] = None,
    n_fft: int = 512,
    hop_length: int = 32,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot waveform, temporal CAM, and ORIGINAL linear-frequency STFT.

    Important
    ---------
    The third panel is the original waveform STFT aligned in time with a
    temporal 1D CAM.  It is NOT a frequency-specific Grad-CAM.
    """
    sig = np.asarray(signal, dtype=np.float32).squeeze()
    cam = np.asarray(cam_heatmap, dtype=np.float32).squeeze()

    if sig.ndim != 1:
        raise ValueError(f"Expected mono signal, received {sig.shape}.")
    if cam.ndim != 1:
        raise ValueError(f"Expected 1D temporal CAM, received {cam.shape}.")
    if sig.size == 0 or cam.size == 0:
        raise ValueError("Signal and CAM must be non-empty.")
    if not np.all(np.isfinite(sig)):
        raise ValueError("Signal contains NaN or Inf.")
    if not np.all(np.isfinite(cam)):
        raise ValueError("CAM contains NaN or Inf.")

    duration_s = sig.size / float(sr)
    waveform_time = np.arange(sig.size, dtype=np.float64) / float(sr)
    cam_time = np.linspace(
        0.0,
        duration_s,
        num=cam.size,
        endpoint=False,
        dtype=np.float64,
    )
    cam_display = _normalize_cam_for_display(cam)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 7),
        sharex=True,
        constrained_layout=True,
    )

    # 1) Original waveform
    axes[0].plot(waveform_time, sig, linewidth=0.8)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(title or "SignalGrad-CAM explanation")
    axes[0].grid(True, alpha=0.25)

    # 2) Temporal CAM
    axes[1].plot(cam_time, cam_display, linewidth=1.2)
    axes[1].fill_between(
        cam_time,
        0.0,
        cam_display,
        alpha=0.25,
    )
    axes[1].set_ylabel("CAM importance\n(display-normalized)")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.25)

    if diagnostics is not None:
        axes[1].set_title(
            "Raw CAM: "
            f"min={diagnostics.cam_min:.3e}, "
            f"max={diagnostics.cam_max:.3e}, "
            f"std={diagnostics.cam_std:.3e}, "
            f"unique={diagnostics.unique_count}, "
            f"degenerate={diagnostics.degenerate_cam}"
        )

    # 3) Original linear-frequency STFT
    n_fft = int(n_fft)
    hop_length = int(hop_length)
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft and hop_length must be > 0.")

    # Use a shorter window only if a future input is shorter than 512 samples.
    win_length = min(n_fft, sig.size)
    fft_length = max(n_fft, win_length)

    stft = librosa.stft(
        sig,
        n_fft=fft_length,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=False,
    )
    magnitude_db = librosa.amplitude_to_db(
        np.abs(stft),
        ref=np.max,
    )
    freqs = librosa.fft_frequencies(
        sr=sr,
        n_fft=fft_length,
    )

    if magnitude_db.shape[1] > 1:
        stft_time = (
            np.arange(magnitude_db.shape[1], dtype=np.float64)
            * hop_length
            + win_length / 2.0
        ) / float(sr)
        t0 = max(0.0, stft_time[0] - hop_length / (2.0 * sr))
        t1 = min(
            duration_s,
            stft_time[-1] + hop_length / (2.0 * sr),
        )
    else:
        t0, t1 = 0.0, duration_s

    im = axes[2].imshow(
        magnitude_db,
        origin="lower",
        aspect="auto",
        extent=[
            t0,
            t1,
            float(freqs[0]),
            float(freqs[-1]),
        ],
    )
    axes[2].set_ylim(0.0, sr / 2.0)
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Original STFT aligned with temporal CAM")
    fig.colorbar(im, ax=axes[2], label="Magnitude (dB)")

    for ax in axes:
        ax.set_xlim(0.0, duration_s)

    if save_path:
        fig.savefig(
            save_path,
            dpi=250,
            bbox_inches="tight",
        )

    return fig, axes


__all__ = ["plot_signal_gradcam"]
