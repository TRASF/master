"""Misclassification and anomaly audio/diagnostic frame exporter.

Saves PCM audio buffers and diagnostic metadata for offline error analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Optional, Dict, Any

import numpy as np
try:
    import soundfile as sf
except ImportError:
    sf = None


def export_anomaly_frame(
    audio: np.ndarray,
    sample_rate: int,
    metadata: Dict[str, Any],
    output_dir: str | Path = "output/misclassifications",
    heatmap: Optional[np.ndarray] = None,
) -> Path:
    """Save audio wave buffer and diagnostic JSON to output directory.

    Args:
        audio: 1D PCM audio array (float32 [-1, 1] or int16).
        sample_rate: Sampling rate (e.g. 8000).
        metadata: Dict containing MCU prediction, host prediction, confidence, timestamp.
        output_dir: Directory path for exporting anomalies.
        heatmap: Optional 1D/2D Grad-CAM heatmap array.

    Returns:
        Path of the saved directory containing wav and json.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    seq = metadata.get("seq", 0)
    frame_dir = out_path / f"anomaly_{timestamp_str}_seq{seq:05d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Save audio WAV
    if audio.dtype == np.int16:
        audio_float = audio.astype(np.float32) / 32768.0
    else:
        audio_float = audio

    if sf is not None:
        wav_file = frame_dir / "audio.wav"
        sf.write(str(wav_file), audio_float, sample_rate)

    # Save metadata JSON
    meta_file = frame_dir / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if heatmap is not None:
        heatmap_file = frame_dir / "heatmap.npy"
        np.save(str(heatmap_file), heatmap)

    return frame_dir
