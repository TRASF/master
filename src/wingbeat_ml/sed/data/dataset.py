"""Manifest adapter for canonical 16 kHz ATST-SED training windows with bounded gain & bandwidth augmentation."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio.functional as AF
import torchaudio.transforms as T
from torch.utils.data import Dataset

from wingbeat_ml.data.synthetic import decode_pcm_bytes


def compute_recording_bounded_gain(
    signal: np.ndarray,
    target_rms: float = 0.05,
    min_gain: float = 0.1,
    max_gain: float = 10.0,
    eps: float = 1e-6,
) -> float:
    """Calculate bounded RMS gain for a full recording/channel."""
    rms = np.sqrt(np.mean(signal**2) + eps)
    raw_gain = target_rms / max(rms, eps)
    return float(np.clip(raw_gain, min_gain, max_gain))


def apply_bandwidth_augmentation(
    audio: torch.Tensor,
    sample_rate: int = 16000,
    cutoff_freq: float = 3800.0,
) -> torch.Tensor:
    """Apply random low-pass filter to simulate 8 kHz acquisition cutoff."""
    return AF.lowpass_biquad(audio, sample_rate, cutoff_freq)


class ATSTMosquitoDataset(Dataset):
    """Leakage-safe, strongly labelled 4-second SED windows with bounded RMS gain."""

    def __init__(
        self,
        recordings_csv: str | Path,
        events_csv: str | Path,
        teacher_sample_rate: int = 8000,
        frame_rate_hz: float = 25.0,
        segment_len_s: float = 10.0,
        split: str = "train",
        crop_seed: int = 42,
        bandwidth_aug_prob: float = 0.5,
        target_rms: float = 0.05,
        synthetic_ratio_max: float = 0.3,
    ):
        self.teacher_sample_rate = teacher_sample_rate
        self.frame_rate_hz = frame_rate_hz
        self.segment_len_s = segment_len_s
        self.num_frames = int(segment_len_s * frame_rate_hz)
        self.split = split
        self.bandwidth_aug_prob = bandwidth_aug_prob if split == "train" else 0.0
        self.target_rms = target_rms
        rng = np.random.default_rng(crop_seed + {"train": 0, "validation": 1, "test": 2}.get(split, 3))

        recordings = pd.read_csv(recordings_csv)
        recordings = recordings[recordings["split"] == split].drop_duplicates("file_id")
        if Path(events_csv).exists() and Path(events_csv).stat().st_size > 0:
            self.events = pd.read_csv(events_csv)
        else:
            self.events = pd.DataFrame()

        real_samples = []
        synth_samples = []
        for _, row in recordings.iterrows():
            supervision = str(row["supervision_type"])
            dataset_name = str(row.get("dataset", ""))

            # Exclude unlabeled or unverified positive clips from Stage 1 BCE
            if supervision in {"unlabeled", "positive_clip"}:
                continue

            record = row.to_dict()
            is_synth = dataset_name == "Synthetic"

            if supervision == "strong":
                file_events = self.events[self.events["recording_file_id"] == str(row["file_id"])]
                starts = set()
                duration = float(row["duration_s"])
                max_start = max(0.0, duration - segment_len_s)
                for _, event in file_events.iterrows():
                    event_start = float(event["start_s"])
                    event_end = float(event["end_s"])
                    lower = max(0.0, event_start - segment_len_s)
                    upper = min(max_start, event_end)
                    start = round(float(rng.uniform(lower, upper)) if upper > lower else lower, 3)
                    if start not in starts:
                        starts.add(start)
                        sample_dict = {**record, "segment_start_s": start}
                        if is_synth:
                            synth_samples.append(sample_dict)
                        else:
                            real_samples.append(sample_dict)
            elif supervision == "clean_negative":
                sample_dict = {**record, "segment_start_s": 0.0}
                if is_synth:
                    synth_samples.append(sample_dict)
                else:
                    real_samples.append(sample_dict)

        # Cap synthetic exposure to auxiliary ratio (max 30% of total)
        if synth_samples and real_samples:
            max_synth = int(len(real_samples) * synthetic_ratio_max / (1.0 - synthetic_ratio_max))
            if len(synth_samples) > max_synth:
                synth_samples = list(rng.choice(synth_samples, size=max_synth, replace=False))

        samples = real_samples + synth_samples
        self.recs = pd.DataFrame(samples).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.recs)

    def _load_audio(self, path: str, channel_index: int = 0, start_s: float = 0.0) -> torch.Tensor:
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            wf.setpos(min(int(start_s * sr), wf.getnframes()))
            data = wf.readframes(int(self.segment_len_s * sr))
            signal = decode_pcm_bytes(data, wf.getsampwidth())

        if n_ch > 1:
            signal = signal[: len(signal) - (len(signal) % n_ch)].reshape(-1, n_ch)
            if channel_index:
                if channel_index > n_ch:
                    raise ValueError(f"Channel {channel_index} does not exist in {path}")
                signal = signal[:, channel_index - 1]
            else:
                signal = signal[:, int(np.argmax(np.sqrt(np.mean(signal**2, axis=0))))]

        gain = compute_recording_bounded_gain(signal, target_rms=self.target_rms)
        signal = signal * gain

        audio = torch.from_numpy(signal.copy()).float()
        if sr != self.teacher_sample_rate:
            audio = T.Resample(
                sr,
                self.teacher_sample_rate,
                resampling_method="sinc_interp_hann",
            )(audio)

        if self.split == "train" and sr > 8000 and np.random.rand() < self.bandwidth_aug_prob:
            cutoff = float(np.random.uniform(3400.0, 4000.0))
            audio = apply_bandwidth_augmentation(audio, sample_rate=self.teacher_sample_rate, cutoff_freq=cutoff)

        target_samples = int(self.segment_len_s * self.teacher_sample_rate)
        if len(audio) < target_samples:
            audio = torch.nn.functional.pad(audio, (0, target_samples - len(audio)))
        else:
            audio = audio[:target_samples]
        return audio

        target_samples = int(self.segment_len_s * self.teacher_sample_rate)
        if len(audio) < target_samples:
            audio = torch.nn.functional.pad(audio, (0, target_samples - len(audio)))
        return audio[:target_samples]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | float]:
        row = self.recs.iloc[idx]
        file_path = str(row["path"])
        file_id = str(row["file_id"])
        segment_start = float(row["segment_start_s"])
        channel_index = int(row.get("channel_index", 0) or 0)
        audio = self._load_audio(file_path, channel_index, segment_start)

        target = torch.zeros((self.num_frames, 1), dtype=torch.float32)
        supervision = str(row["supervision_type"])
        if supervision == "strong" and not self.events.empty:
            file_events = self.events[self.events["recording_file_id"] == file_id]
            segment_end = segment_start + self.segment_len_s
            for _, event in file_events.iterrows():
                start_s = max(float(event["start_s"]), segment_start)
                end_s = min(float(event["end_s"]), segment_end)
                if end_s <= start_s:
                    continue
                start_frame = max(0, int((start_s - segment_start) * self.frame_rate_hz))
                end_frame = min(self.num_frames, int(np.ceil((end_s - segment_start) * self.frame_rate_hz - 1e-9)))
                target[start_frame:end_frame, 0] = 1.0

        return {
            "audio": audio,
            "target": target,
            "file_id": file_id,
            "path": file_path,
            "segment_start_s": segment_start,
        }
