"""Head-only fine-tuning from human-verified archive regions."""

from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset

from wingbeat_ml.data.synthetic import decode_pcm_bytes
from wingbeat_ml.sed.inference.long_recording import load_sed_teacher_model, parse_pcm_filename_meta
from wingbeat_ml.sed.training.cache import CachedATSTDataset, cache_atst_features
from wingbeat_ml.sed.training.train import align_target_to_logits, evaluate_loss


class VerifiedReviewDataset(Dataset):
    """Ten-second positive and localized hard-negative crops."""

    def __init__(
        self,
        dataset_dir: str | Path,
        recording_ids: set[str],
        sample_rate: int = 8_000,
        frame_rate_hz: float = 25.0,
        segment_length_s: float = 10.0,
    ) -> None:
        root = Path(dataset_dir)
        recordings = pd.read_csv(root / "recordings.csv", dtype={"recording_id": str})
        recordings["recording_id"] = recordings["recording_id"].astype(str)
        self.recordings = recordings.set_index("recording_id")
        events = pd.read_csv(root / "events.csv", dtype={"recording_id": str})
        negatives = pd.read_csv(root / "hard_negatives.csv", dtype={"recording_id": str})
        events["recording_id"] = events["recording_id"].astype(str)
        negatives["recording_id"] = negatives["recording_id"].astype(str)
        events = events[events.recording_id.isin(recording_ids)].assign(positive=True)
        negatives = negatives[negatives.recording_id.isin(recording_ids)].assign(positive=False)
        self.samples = pd.concat([events, negatives], ignore_index=True)
        self.sample_rate = sample_rate
        self.frame_rate_hz = frame_rate_hz
        self.segment_length_s = segment_length_s
        self.segment_samples = int(sample_rate * segment_length_s)
        self.target_frames = int(frame_rate_hz * segment_length_s)
        if self.samples.empty:
            raise ValueError("Verified review split contains no samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples.iloc[index]
        source = Path(self.recordings.loc[str(sample.recording_id), "audio_path"])
        midpoint = (float(sample.start_s) + float(sample.end_s)) / 2
        if source.suffix.lower() == ".pcm":
            source_rate, sample_width, channels = parse_pcm_filename_meta(source.name)
            frame_count = source.stat().st_size // (sample_width * channels)
            duration_s = frame_count / source_rate
            crop_start = min(max(0.0, midpoint - self.segment_length_s / 2), max(0.0, duration_s - self.segment_length_s))
            with source.open("rb") as audio_file:
                audio_file.seek(int(crop_start * source_rate) * sample_width * channels)
                signal = decode_pcm_bytes(
                    audio_file.read(int(self.segment_length_s * source_rate) * sample_width * channels),
                    sample_width,
                )
        else:
            with wave.open(str(source), "rb") as wav_file:
                source_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                duration_s = wav_file.getnframes() / source_rate
                crop_start = min(max(0.0, midpoint - self.segment_length_s / 2), max(0.0, duration_s - self.segment_length_s))
                wav_file.setpos(min(int(crop_start * source_rate), wav_file.getnframes()))
                signal = decode_pcm_bytes(
                    wav_file.readframes(int(self.segment_length_s * source_rate)),
                    wav_file.getsampwidth(),
                )

        if channels > 1:
            signal = signal[: len(signal) - len(signal) % channels].reshape(-1, channels)
            signal = signal[:, int(np.argmax(np.sqrt(np.mean(signal**2, axis=0))))]
        peak = np.max(np.abs(signal), initial=0.0)
        if peak > 1e-6:
            signal = signal / peak * 0.5
        audio = torch.from_numpy(signal.copy()).float()
        if source_rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, source_rate, self.sample_rate)
        audio = torch.nn.functional.pad(audio, (0, max(0, self.segment_samples - len(audio))))[: self.segment_samples]

        target = torch.zeros((self.target_frames, 1), dtype=torch.float32)
        if bool(sample.positive):
            start = max(float(sample.start_s), crop_start)
            end = min(float(sample.end_s), crop_start + self.segment_length_s)
            if end > start:
                start_frame = max(0, int((start - crop_start) * self.frame_rate_hz))
                end_frame = min(self.target_frames, int(np.ceil((end - crop_start) * self.frame_rate_hz)))
                target[start_frame:end_frame] = 1
        return {"audio": audio, "target": target}


def _fingerprint(paths: list[Path], values: dict) -> str:
    digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode())
    for path in paths:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()[:16]


def fine_tune_verified(
    dataset_dir: str | Path,
    base_checkpoint: str | Path = "artifacts/teacher_v0/teacher_v0_best.pt",
    output_dir: str | Path = "artifacts/teacher_v1",
    epochs: int = 20,
    learning_rate: float = 3e-5,
    batch_size: int = 64,
    patience: int = 5,
    seed: int = 42,
) -> Path:
    """Fine-tune only temporal head and save separate V1 checkpoint."""
    dataset_root = Path(dataset_dir)
    recordings = pd.read_csv(dataset_root / "recordings.csv", dtype={"recording_id": str})
    recording_ids = sorted(recordings.recording_id.unique())
    if len(recording_ids) < 2:
        raise ValueError("Fine-tuning requires at least two reviewed recordings")
    rng = np.random.default_rng(seed)
    rng.shuffle(recording_ids)
    validation_count = max(1, round(len(recording_ids) * 0.2))
    validation_ids = set(recording_ids[:validation_count])
    training_ids = set(recording_ids[validation_count:])

    train_raw = VerifiedReviewDataset(dataset_root, training_ids)
    validation_raw = VerifiedReviewDataset(dataset_root, validation_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_sed_teacher_model(base_checkpoint, device=str(device))
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(
        [dataset_root / name for name in ("recordings.csv", "events.csv", "hard_negatives.csv")]
        + [Path(base_checkpoint), Path(__file__)],
        {"training_ids": sorted(training_ids), "validation_ids": sorted(validation_ids)},
    )
    cache_root = output / "atst_cache" / fingerprint
    cache_atst_features(model.encoder, train_raw, cache_root / "train", device=device)
    cache_atst_features(model.encoder, validation_raw, cache_root / "validation", device=device)
    train_dataset = CachedATSTDataset(cache_root / "train")
    validation_dataset = CachedATSTDataset(cache_root / "validation")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    positive_frames = sum(float(train_dataset[index]["target"].sum()) for index in range(len(train_dataset)))
    total_frames = sum(train_dataset[index]["target"].numel() for index in range(len(train_dataset)))
    if positive_frames == 0:
        raise ValueError("Fine-tuning split contains no positive frames")
    pos_weight = (total_frames - positive_frames) / positive_frames
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss = float("inf")
    stale_epochs = 0
    best_path = output / "teacher_v1_best.pt"

    print(f"Reviewed split: {len(training_ids)} train recordings, {len(validation_ids)} validation recordings")
    print(f"Samples: {len(train_dataset)} train, {len(validation_dataset)} validation")
    print(f"Positive-frame weight: {pos_weight:.3f}")
    for epoch in range(1, epochs + 1):
        model.head.train()
        losses = []
        for batch in train_loader:
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model.head(features)
            loss = criterion(logits.float(), align_target_to_logits(target, logits))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.head.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))

        validation_loss = evaluate_loss(model.head, validation_loader, criterion, device, is_cached=True)
        print(f"Epoch {epoch:03d}/{epochs} | train={np.mean(losses):.5f} | val={validation_loss:.5f}")
        if validation_loss < best_loss - 0.001:
            best_loss = validation_loss
            stale_epochs = 0
            torch.save({
                "format_version": 1,
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "n_atst_blocks": model.encoder.n_blocks,
                    "conv_dim": model.head.local_conv[0].out_channels,
                    "hidden_dim": model.head.gru.hidden_size,
                    "gru_layers": model.head.gru.num_layers,
                    "dropout": model.head.local_conv[3].p,
                },
                "preprocessing": {"sample_rate": 8000, "frame_rate_hz": 25.0, "segment_length_s": 10.0},
                "fine_tune_dataset": str(dataset_root),
                "base_checkpoint": str(base_checkpoint),
                "epoch": epoch,
                "val_loss": validation_loss,
            }, best_path)
            print(f"  saved: {best_path}")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {patience} epochs without validation improvement.")
                break
    return best_path
