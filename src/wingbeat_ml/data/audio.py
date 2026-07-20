"""Stateless audio loading and preprocessing operations.

All functions operate on plain numpy arrays and do not import TensorFlow.
Operation order is preserved exactly from the legacy FileLoader and
AudioAugmentor.apply_post_processing pipeline.

Audio pipeline order (per legacy code):
  1. sf.read() → float32, always_2d=False
  2. to_mono()  (mean over channels if ndim > 1)
  3. librosa.resample() with res_type="kaiser_best" if sr != target_sr
  4. .astype(np.float32)

Post-processing order (after segmentation, inside apply_post_processing):
  1. [Optional] High-pass filter (FIR, prob-gated, always applied in eval)
  2. [Optional] Pre-emphasis  (prob-gated, always applied in eval)
  3. [Training only] Pitch shift  (prob-gated)
  4. [Training only] Time shift   (prob-gated)
  5. [Training only] Time masking (prob-gated)
  6. [Training only] Random gain  (prob-gated)
  7. [Training only] Gaussian noise (prob-gated)
  8. [Training only] Noise overlay (prob-gated, requires noise bank)
  9. DC removal: audio -= mean(audio)   [controlled by preprocess.dc_removal, default True]
 10. RMS normalization: clip gain to [min_gain, max_gain] so RMS → target_rms
 11. Clip to [-1.0, 1.0]

KNOWN ISSUES (do not fix here):
  - RMS normalization always runs unconditionally, even when p=0 for all
    augmentations and no signal changes have occurred.  This means validation
    and test waveforms are also RMS-normalised, which may affect absolute
    amplitude reproduction.  Documented for later review.
  - DC removal occurs AFTER augmentation transforms, so the order differs from
    a classic preprocessing approach.  Preserved here intentionally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import soundfile as sf
import librosa


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_raw(path: str | Path) -> tuple[np.ndarray, int, str]:
    """Read an audio file and return (data, sample_rate, subtype).

    Uses soundfile with dtype='float32' and always_2d=False to match the
    legacy FileLoader behaviour.  Returns the raw (possibly multi-channel)
    float32 array.

    Raises FileNotFoundError if path does not exist.
    """
    path = str(path)
    if not Path(path).is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    info = sf.info(path)
    return data, sr, info.subtype


def to_mono(data: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging channels.

    Matches legacy to_mono(): mean over axis=1 when ndim > 1.
    Always returns float32.
    """
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32)


def resample_audio(
    data: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    """Resample audio using librosa with res_type='kaiser_best'.

    Returns data unchanged (as float32) when source_rate == target_rate.
    Matches legacy resample_audio() exactly.
    """
    if source_rate == target_rate:
        return data.astype(np.float32)
    data = librosa.resample(
        data.astype(np.float32),
        orig_sr=source_rate,
        target_sr=target_rate,
        res_type="kaiser_best",
    )
    return data.astype(np.float32)


def load_audio(path: str | Path, target_sample_rate: int = 8000) -> np.ndarray:
    """Load an audio file as a mono float32 waveform at *target_sample_rate*.

    This is the primary audio loading entry point.  Mirrors FileLoader.load():
      1. sf.read() with dtype='float32'
      2. to_mono()
      3. librosa resample to target_sample_rate
      4. cast to float32

    For .npy files the array is loaded directly (no resampling step).

    Args:
        path: Path to the audio file (.wav or .npy).
        target_sample_rate: Desired output sample rate (default 8000).

    Returns:
        1-D float32 numpy array at target_sample_rate.
    """
    path = str(path)
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32, copy=False)
    data, sr, _ = load_raw(path)
    data = to_mono(data)
    data = resample_audio(data, sr, target_sample_rate)
    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# Audio preprocessing (numpy, no TF, applied before segmentation)
# ---------------------------------------------------------------------------

def preprocess_audio(
    waveform: np.ndarray,
    source_rate: int,
    config: Mapping[str, object] | None = None,
) -> np.ndarray:
    """Apply preprocessing steps to a waveform before segmentation.

    Currently the legacy pipeline does NOT apply preprocessing steps here
    (DC removal / RMS normalisation happen *after* segmentation and augment-
    ation inside apply_post_processing).  This function exists as the correct
    future home for pre-segmentation operations.

    For now it only resamples to the configured sample_rate if necessary.
    All downstream conditioning (DC removal, RMS norm, clip) happens inside
    the TF augmentation pipeline.

    Args:
        waveform: 1-D float32 numpy array.
        source_rate: Sample rate of the input waveform.
        config: Optional mapping; reads 'sample_rate' (default 8000).

    Returns:
        float32 numpy array at config sample_rate.
    """
    cfg = config or {}
    target_rate = int(cfg.get("sample_rate", 8000))
    waveform = resample_audio(waveform, source_rate, target_rate)
    return waveform.astype(np.float32)


# ---------------------------------------------------------------------------
# Segmentation (numpy)
# ---------------------------------------------------------------------------

def segment_audio(
    waveform: np.ndarray,
    segment_length: int,
    overlap: float = 0.0,
    max_segments: int | None = None,
) -> np.ndarray:
    """Divide a waveform into overlapping segments.

    Segments are extracted with a stride of segment_length * (1 - overlap).
    If the waveform is shorter than segment_length it is zero-padded.
    Segments are zero-padded if the tail is shorter than segment_length.

    This is the numpy-side equivalent of the windowed extraction in
    AudioAugmentor.create_segments().  The TF version handles random overlap
    during training; this deterministic version is used in manifest building
    and test fixtures.

    Args:
        waveform: 1-D float32 numpy array.
        segment_length: Number of samples per segment.
        overlap: Overlap ratio in [0, 1).  Default 0.0 (no overlap).
        max_segments: Maximum number of segments to return.

    Returns:
        2-D float32 array of shape (n_segments, segment_length).
    """
    if waveform.size == 0:
        return np.zeros((0, segment_length), dtype=np.float32)

    # Pad if shorter than one segment
    if len(waveform) < segment_length:
        waveform = np.pad(waveform, (0, segment_length - len(waveform)))

    step = max(1, int(segment_length * (1.0 - overlap)))
    starts = list(range(0, len(waveform), step))

    segments = []
    for s in starts:
        chunk = waveform[s: s + segment_length]
        if len(chunk) < segment_length:
            chunk = np.pad(chunk, (0, segment_length - len(chunk)))
        segments.append(chunk.astype(np.float32))
        if max_segments is not None and len(segments) >= max_segments:
            break

    return np.stack(segments, axis=0) if segments else np.zeros(
        (0, segment_length), dtype=np.float32
    )


# Compatibility API retained while callers migrate to load_audio().
load_file = load_raw


class FileLoader:
    """Compatibility loader delegating to the canonical audio functions."""

    def __init__(self, path: str | Path, sample_rate: int = 8000):
        self.path = path
        self.sample_rate = sample_rate

    def load(self) -> np.ndarray:
        return load_audio(
            self.path,
            target_sample_rate=self.sample_rate,
        )


__all__ = [
    "FileLoader",
    "load_audio",
    "load_file",
    "load_raw",
    "preprocess_audio",
    "resample_audio",
    "segment_audio",
    "to_mono",
]
