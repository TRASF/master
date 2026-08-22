"""Synthetic Strongly-Labeled Soundscape Generator.

Mixes verified train-partition mosquito clips into train-partition background recordings
at specified SNRs and random positions with full provenance tracking.
Raises LeakageError if any non-train clip or background is used.
"""

from __future__ import annotations

import hashlib
import json
import random
import wave
import numpy as np
from pathlib import Path
import pandas as pd
import torch
import torchaudio.transforms as T


class LeakageError(Exception):
    """Raised when non-train audio is passed to synthetic generator."""
    pass


def decode_pcm_bytes(data: bytes, sample_width: int) -> np.ndarray:
    """Decode 16-bit, 24-bit, or 32-bit PCM audio bytes to float32 normalized signal."""
    if sample_width == 2:
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 3:
        a = np.frombuffer(data, dtype=np.uint8)
        n_samples = len(a) // 3
        a = a[: n_samples * 3].reshape(-1, 3)
        padded = np.zeros((n_samples, 4), dtype=np.uint8)
        padded[:, 1:] = a
        int32_samples = padded.view(dtype=np.int32).reshape(-1)
        return (int32_samples >> 8).astype(np.float32) / 8388608.0
    elif sample_width == 4:
        return np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def load_wav_pcm(
    path: str | Path,
    channel_index: int | None = None,
    target_sample_rate: int = 8_000,
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> tuple[np.ndarray, int]:
    """Load one physical microphone channel as canonical 8 kHz mono audio."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        wf.setpos(min(int(start_s * sr), wf.getnframes()))
        n_frames = wf.getnframes() - wf.tell() if duration_s is None else int(duration_s * sr)
        data = wf.readframes(n_frames)
        sample_width = wf.getsampwidth()
        
        signal = decode_pcm_bytes(data, sample_width)

        if n_ch > 1:
            signal = signal[: len(signal) - (len(signal) % n_ch)].reshape(-1, n_ch)
            if channel_index is not None:
                if channel_index < 1 or channel_index > n_ch:
                    raise ValueError(f"Channel {channel_index} does not exist in {path}")
                signal = signal[:, channel_index - 1]
            else:
                signal = signal[:, int(np.argmax(np.sqrt(np.mean(signal**2, axis=0))))]

        signal_tensor = torch.from_numpy(signal.copy()).float()
        if sr != target_sample_rate:
            signal_tensor = T.Resample(
                sr,
                target_sample_rate,
                resampling_method="sinc_interp_hann",
            )(signal_tensor)
        return signal_tensor.numpy(), target_sample_rate


def save_wav_pcm(path: str | Path, signal: np.ndarray, sample_rate: int) -> None:
    """Save float32 array as 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    int_signal = np.clip(signal * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_signal.tobytes())


def generate_synthetic_soundscapes(
    metadata_dir: str | Path,
    output_dir: str | Path,
    num_scenes: int = 100,
    scene_duration_s: float = 4.0,
    snr_db_range: tuple[float, float] = (0.0, 12.0),
    seed: int = 42,
    sample_rate_hz: int = 16_000,
    event_count_weights: tuple[float, float, float, float] = (0.10, 0.55, 0.30, 0.05),
    fade_ms: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate SyntheticV2 scenes from real crops with exact strong labels."""
    rng = random.Random(seed)
    meta_p = Path(metadata_dir)
    rec_df = pd.read_csv(meta_p / "recordings.csv")
    event_df = pd.read_csv(meta_p / "events.csv")

    train_bgs = rec_df[
        (rec_df["split"] == "train")
        & (rec_df["supervision_type"] == "negative")
        & (rec_df["duration_s"] >= scene_duration_s)
    ]
    if train_bgs.empty:
        raise ValueError("SyntheticV2 requires train negative backgrounds at least as long as the scene")

    foregrounds = event_df.merge(
        rec_df[["file_id", "path", "channel_index", "split", "dataset"]],
        left_on="recording_file_id",
        right_on="file_id",
        suffixes=("", "_recording"),
    )
    foregrounds = foregrounds[
        (foregrounds["split_recording"] == "train")
        & (foregrounds["dataset"] != "Synthetic")
        & foregrounds["provenance"].astype(str).str.endswith("ground_truth")
        & (foregrounds["end_s"] > foregrounds["start_s"])
        & ((foregrounds["end_s"] - foregrounds["start_s"]) < scene_duration_s)
    ]
    if foregrounds.empty and any(event_count_weights[1:]):
        raise ValueError("SyntheticV2 requires real train strong events as foreground assets")

    out_p = Path(output_dir)
    audio_dir = out_p / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    scenes_records = []
    events_records = []
    fade_samples = int(sample_rate_hz * fade_ms / 1000.0)

    for i in range(num_scenes):
        scene_id = f"synth_{i:06d}"
        bg_row = train_bgs.iloc[rng.randrange(len(train_bgs))]
        if bg_row["split"] != "train":
            raise LeakageError(f"Background {bg_row['path']} is not in train split")
        bg_offset = rng.uniform(0.0, float(bg_row["duration_s"]) - scene_duration_s)
        bg_channel = int(bg_row.get("channel_index", 0) or 0) or None
        mixed, _ = load_wav_pcm(
            bg_row["path"], bg_channel, sample_rate_hz, bg_offset, scene_duration_s
        )
        target_len = int(scene_duration_s * sample_rate_hz)
        if len(mixed) < target_len:
            raise ValueError(f"Background crop shorter than requested scene: {bg_row['path']}")
        mixed = mixed[:target_len].copy()
        background = mixed.copy()

        event_count = rng.choices((0, 1, 2, 3), weights=event_count_weights, k=1)[0]
        scene_sources = []
        scene_snrs = []
        for event_number in range(event_count):
            source = foregrounds.iloc[rng.randrange(len(foregrounds))]
            if source["split_recording"] != "train":
                raise LeakageError(f"Foreground {source['path_recording']} is not in train split")
            duration = float(source["end_s"] - source["start_s"])
            channel = int(source.get("channel_index", 0) or 0) or None
            foreground, _ = load_wav_pcm(
                source["path_recording"], channel, sample_rate_hz,
                float(source["start_s"]), duration,
            )
            if not len(foreground) or len(foreground) >= target_len:
                continue
            if fade_samples:
                fade = min(fade_samples, len(foreground) // 2)
                ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
                foreground[:fade] *= ramp
                foreground[-fade:] *= ramp[::-1]

            start_sample = rng.randint(0, target_len - len(foreground))
            end_sample = start_sample + len(foreground)
            local_bg = background[start_sample:end_sample]
            bg_rms = float(np.sqrt(np.mean(local_bg**2) + 1e-8))
            foreground_rms = float(np.sqrt(np.mean(foreground**2) + 1e-8))
            snr_db = rng.uniform(*snr_db_range)
            scaled = foreground * (bg_rms * 10 ** (snr_db / 20.0) / foreground_rms)
            mixed[start_sample:end_sample] += scaled

            scene_sources.append(str(source["path_recording"]))
            scene_snrs.append(snr_db)
            start_s = start_sample / sample_rate_hz
            end_s = end_sample / sample_rate_hz
            events_records.append({
                "event_id": f"evt_{scene_id}_{event_number:02d}",
                "recording_file_id": scene_id,
                "path": str(audio_dir / f"{scene_id}.wav"),
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
                "label": "mosquito",
                "confidence": 1.0,
                "provenance": "synthetic_ground_truth",
                "split": "train",
            })

        peak = float(np.max(np.abs(mixed), initial=0.0))
        if peak > 0.95:
            mixed *= 0.95 / peak
        out_wav_path = audio_dir / f"{scene_id}.wav"
        save_wav_pcm(out_wav_path, mixed, sample_rate_hz)
        scenes_records.append({
            "scene_id": scene_id,
            "path": str(out_wav_path),
            "duration_s": scene_duration_s,
            "background_source": bg_row["path"],
            "background_crop_start_s": bg_offset,
            "mosquito_sources": json.dumps(scene_sources),
            "snr_db": json.dumps(scene_snrs),
            "event_count": len(scene_sources),
            "random_seed": seed,
            "split": "train",
        })

    scenes_df = pd.DataFrame(scenes_records)
    events_df = pd.DataFrame(
        events_records,
        columns=[
            "event_id", "recording_file_id", "path", "start_s", "end_s",
            "duration_s", "label", "confidence", "provenance", "split",
        ],
    )

    scenes_df.to_csv(out_p / "generation_manifest.csv", index=False)
    events_df.to_csv(out_p / "events.csv", index=False)

    return scenes_df, events_df


def add_synthetic_to_metadata(
    scenes_df: pd.DataFrame,
    events_df: pd.DataFrame,
    metadata_dir: str | Path,
    sample_rate_hz: int = 8_000,
) -> None:
    """Add generated training scenes to main SED manifests idempotently."""
    metadata_path = Path(metadata_dir)
    recordings_path = metadata_path / "recordings.csv"
    events_path = metadata_path / "events.csv"
    recordings = pd.read_csv(recordings_path)
    events = pd.read_csv(events_path)

    recordings = recordings[recordings["dataset"] != "Synthetic"]
    if "provenance" in events.columns:
        events = events[events["provenance"] != "synthetic_ground_truth"]

    synthetic_recordings = []
    for _, scene in scenes_df.iterrows():
        path = Path(scene["path"])
        synthetic_recordings.append({
            "file_id": scene["scene_id"],
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
            "source_id": scene["scene_id"],
            "parent_recording_id": scene["scene_id"],
            "dataset": "Synthetic",
            "supervision_type": "strong",
            "sample_rate_hz": sample_rate_hz,
            "channels": 1,
            "duration_s": scene["duration_s"],
            "device": "synthetic_mixer",
            "session": scene["scene_id"],
            "environment": "synthetic",
            "location": "synthetic",
            "species": "mosquito",
            "sex": "unknown",
            "individual_id": "mixed",
            "annotation_source": "synthetic_ground_truth",
            "split_group": scene["scene_id"],
            "channel_index": 0,
            "split": "train",
        })

    synthetic_events = events_df.copy()
    synthetic_events["split_group"] = synthetic_events["recording_file_id"]
    synthetic_events["split"] = "train"
    pd.concat([recordings, pd.DataFrame(synthetic_recordings)], ignore_index=True).to_csv(recordings_path, index=False)
    pd.concat([events, synthetic_events], ignore_index=True).to_csv(events_path, index=False)
    print(f"Added {len(synthetic_recordings)} synthetic scenes to training metadata.")


if __name__ == "__main__":
    generate_synthetic_soundscapes(
        metadata_dir="/home/miru4090s/clones/Master Thesises/MosSongPlus/metadata",
        output_dir="/home/miru4090s/clones/Master Thesises/MosSongPlus/dataset/synthetic",
        num_scenes=50,
    )
