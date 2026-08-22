"""Signal analysis module: spectrum, STFT, PSD, harmonics, and waveform statistics."""

from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np
from scipy import signal as scipy_signal


def compute_psd(
    signal: np.ndarray,
    sample_rate: int = 8000,
    n_fft: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Power Spectral Density (PSD) using FFT."""
    sig = np.squeeze(np.asarray(signal, dtype=np.float32))
    fft_vals = np.fft.rfft(sig, n=n_fft)
    psd = np.abs(fft_vals) ** 2 / max(len(sig), 1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return freqs, psd


def compute_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 8000,
    n_fft: int = 512,
    hop_length: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Short-Time Fourier Transform (STFT) magnitude spectrogram.

    Args:
        audio: 1D PCM audio array (float32 [-1, 1] or int16).
        sample_rate: Audio sampling rate in Hz (default: 8000 Hz).
        n_fft: FFT window size (default: 512).
        hop_length: Hop stride between STFT frames (default: 128).

    Returns:
        Tuple of (frequencies, times, magnitude_db_spectrogram)
    """
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    audio = audio - np.mean(audio)  # DC removal

    frequencies, times, stft = scipy_signal.stft(
        audio,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary=None,
    )
    magnitude = np.abs(stft)
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-5))

    return frequencies, times, magnitude_db


def analyze_harmonics(
    audio: np.ndarray,
    sample_rate: int = 8000,
    freq_range: Tuple[float, float] = (150.0, 1000.0),
) -> Dict[str, Any]:
    """Extract dominant fundamental frequency (f0) and PSD peak ratios.

    Args:
        audio: 1D PCM audio signal.
        sample_rate: Sampling rate in Hz.
        freq_range: Min and max expected mosquito wingbeat frequency bounds in Hz.

    Returns:
        Dict with 'f0_hz', 'peak_power_db', 'psd_freqs', 'psd_db'
    """
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    freqs, psd = scipy_signal.welch(audio, fs=sample_rate, nperseg=min(len(audio), 1024))
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-10))

    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if np.any(mask):
        valid_freqs = freqs[mask]
        valid_psd = psd_db[mask]
        max_idx = np.argmax(valid_psd)
        f0_hz = float(valid_freqs[max_idx])
        peak_power = float(valid_psd[max_idx])
    else:
        f0_hz = 0.0
        peak_power = -100.0

    return {
        "f0_hz": f0_hz,
        "peak_power_db": peak_power,
        "psd_freqs": freqs,
        "psd_db": psd_db,
    }


def analyze_waveform_stats(signal: np.ndarray) -> dict[str, float]:
    """Calculate basic waveform statistics."""
    sig = np.squeeze(np.asarray(signal, dtype=np.float32))
    rms = float(np.sqrt(np.mean(sig ** 2)))
    peak = float(np.max(np.abs(sig))) if sig.size > 0 else 0.0
    return {
        "mean": float(np.mean(sig)),
        "std": float(np.std(sig)),
        "rms": rms,
        "peak": peak,
        "crest_factor": float(peak / (rms + 1e-12)),
    }


__all__ = [
    "compute_psd",
    "compute_spectrogram",
    "analyze_harmonics",
    "analyze_waveform_stats",
]
