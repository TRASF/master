"""ATST-Frame mosquito SED teacher V0.

16 kHz waveform -> official pretrained ATST-Frame -> Conv1D + BiGRU -> frame logits.

Supports both:
1. Frozen pretrained ATST encoder (cached ATST feature embeddings [T, 768] + AMP head training).
2. End-to-end fine-tuning when ATST encoder is unfrozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import yaml
from torch.utils.data import DataLoader

from wingbeat_ml.sed.training.cache import CachedATSTDataset, cache_atst_features
from wingbeat_ml.sed.data.dataset import ATSTMosquitoDataset

try:
    from audiossl.methods.atstframe.embedding import load_model as load_atst_frame
except ImportError as exc:
    raise ImportError(
        "The official Audio-WestlakeU/audiossl package is required. "
        "Install it from https://github.com/Audio-WestlakeU/audiossl"
    ) from exc


class ATSTFrameEncoder(nn.Module):
    """Official pretrained ATST-Frame encoder."""

    INPUT_SAMPLE_RATE = 16_000
    EXPECTED_SAMPLE_RATE = 16_000
    MAX_MEL_FRAMES = 1001

    def __init__(
        self,
        checkpoint_path: str | Path,
        n_blocks: int = 1,
        freeze: bool = True,
        input_sample_rate: int = 16_000,
    ) -> None:
        super().__init__()

        checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        if not checkpoint_path or not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Official pretrained ATST-Frame checkpoint not found: {checkpoint_path}. "
                "Refusing to train or infer with a random frozen encoder."
            )
        previous_weights_setting = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        try:
            self.encoder = load_atst_frame(str(checkpoint_path))
        finally:
            if previous_weights_setting is None:
                os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous_weights_setting
        self.input_sample_rate = input_sample_rate
        self.input_resampler = T.Resample(
            self.input_sample_rate,
            self.EXPECTED_SAMPLE_RATE,
            resampling_method="sinc_interp_hann",
        ) if self.input_sample_rate != self.EXPECTED_SAMPLE_RATE else nn.Identity()
        self.n_blocks = int(n_blocks)
        self.freeze = bool(freeze)
        self.output_dim = int(self.encoder.embed_dim) * self.n_blocks

        if self.freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True) -> "ATSTFrameEncoder":
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    def _extract(self, audio: torch.Tensor) -> torch.Tensor:
        audio = self.input_resampler(audio)

        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        elif audio.ndim != 3 or audio.shape[1] != 1:
            raise ValueError(
                "Expected mono [B,S] or [B,1,S], got " f"{tuple(audio.shape)}"
            )

        self.encoder.transform.transforms[0].to(audio.device)
        mel = self.encoder.transform(audio)

        if mel.shape[-1] > self.MAX_MEL_FRAMES:
            raise ValueError(
                "V0 training chunks must be <= about 10 s for the official "
                f"ATST positional embedding; got {mel.shape[-1]} mel frames."
            )

        lengths = torch.full(
            (mel.shape[0],),
            mel.shape[-1],
            dtype=torch.long,
            device=audio.device,
        )

        return self.encoder.get_intermediate_layers(
            mel,
            lengths,
            n=self.n_blocks,
            scene=False,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                return self._extract(audio)
        return self._extract(audio)


class FocalLoss(nn.Module):
    """Binary Focal Loss for frame-imbalanced sound event detection."""

    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = target * probs + (1.0 - target) * (1.0 - probs)
        focal_weight = (1.0 - p_t) ** self.gamma
        if self.pos_weight != 1.0:
            weight = target * self.pos_weight + (1.0 - target)
            loss = focal_weight * weight * bce_loss
        else:
            loss = focal_weight * bce_loss
        return loss.mean()


class SEDTemporalHead(nn.Module):
    """Small mosquito-specific temporal detector supporting BiGRU or Transformer heads."""

    def __init__(
        self,
        in_dim: int,
        conv_dim: int = 256,
        hidden_dim: int = 256,
        gru_layers: int = 2,
        dropout: float = 0.2,
        head_type: str = "gru",
        transformer_heads: int = 4,
        transformer_layers: int = 2,
    ) -> None:
        super().__init__()
        self.head_type = head_type.lower()
        self.input_norm = nn.LayerNorm(in_dim)
        self.local_conv = nn.Sequential(
            nn.Conv1d(in_dim, conv_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(conv_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if self.head_type == "gru":
            self.temporal = nn.GRU(
                input_size=conv_dim,
                hidden_size=hidden_dim,
                num_layers=gru_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if gru_layers > 1 else 0.0,
            )
            classifier_in = hidden_dim * 2
            self.proj = None
        elif self.head_type in ("transformer", "conformer"):
            self.proj = nn.Linear(conv_dim, hidden_dim) if conv_dim != hidden_dim else nn.Identity()
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=transformer_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
            classifier_in = hidden_dim
        else:
            raise ValueError(f"Unsupported head_type: {head_type}. Choose 'gru' or 'transformer'.")

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_in),
            nn.Dropout(dropout),
            nn.Linear(classifier_in, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(features)                    # [B,T,D]
        x = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        if self.head_type == "gru":
            x, _ = self.temporal(x)
        else:
            x = self.proj(x)
            x = self.temporal(x)
        return self.classifier(x)                       # logits [B,T,1]


class FullSEDTeacher(nn.Module):
    """Official ATST-Frame backbone plus task-specific SED head."""

    def __init__(
        self,
        atst_checkpoint: str | Path,
        n_atst_blocks: int = 1,
        conv_dim: int = 256,
        hidden_dim: int = 256,
        gru_layers: int = 2,
        dropout: float = 0.2,
        freeze_encoder: bool = True,
        head_type: str = "gru",
        transformer_heads: int = 4,
        transformer_layers: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = ATSTFrameEncoder(
            atst_checkpoint,
            n_blocks=n_atst_blocks,
            freeze=freeze_encoder,
        )
        self.head = SEDTemporalHead(
            in_dim=self.encoder.output_dim,
            conv_dim=conv_dim,
            hidden_dim=hidden_dim,
            gru_layers=gru_layers,
            dropout=dropout,
            head_type=head_type,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(audio))


def align_target_to_logits(target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Nearest-align strong targets if framing differs by a small amount."""
    if target.ndim == 2:
        target = target.unsqueeze(-1)
    if target.ndim != 3 or target.shape[-1] != 1:
        raise ValueError(f"Expected target [B,T,1] or [B,T], got {target.shape}")
    if target.shape[1] == logits.shape[1]:
        return target.float()
    return F.interpolate(
        target.transpose(1, 2).float(),
        size=logits.shape[1],
        mode="nearest",
    ).transpose(1, 2)


@torch.no_grad()
def evaluate_loss(
    model_or_head: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_cached: bool = True,
    amp_dtype: torch.dtype = torch.float16,
) -> float:
    model_or_head.eval()
    total = 0.0
    count = 0
    use_amp = device.type == "cuda"
    for batch in loader:
        if is_cached:
            inp = batch["features"].to(device, non_blocking=True)
        else:
            inp = batch["audio"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model_or_head(inp)
        target = align_target_to_logits(target, logits)
        loss = criterion(logits.float(), target.float())
        total += float(loss.item())
        count += 1
    return total / max(1, count)


def train_teacher_v0(
    config_path: str | Path,
    metadata_dir: str | Path,
    output_dir: str | Path,
    epochs_override: int | None = None,
) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    dcfg = cfg["dataset"]
    mcfg = cfg["model"]
    tcfg = cfg["training"]

    sample_rate = int(dcfg["teacher_sample_rate"])
    frame_rate = float(dcfg["frame_rate_hz"])
    segment_len = float(dcfg["segment_length_s"])

    if sample_rate != 16_000:
        raise ValueError(f"V2 requires 16 kHz input, got {sample_rate}")
    if abs(frame_rate - 25.0) > 1e-6:
        raise ValueError("Use frame_rate_hz=25 for ~40 ms ATST-Frame tokens")
    if abs(segment_len - 4.0) > 1e-4:
        raise ValueError(f"V2 requires 4.0 s segments, got {segment_len}")
    configured_pos_weight = tcfg.get("pos_weight", 1.0)
    if configured_pos_weight == "auto":
        raise ValueError("V2 precision-first training must not use pos_weight=auto")
    if int(mcfg.get("n_atst_blocks", 1)) != 1 or int(mcfg.get("gru_layers", 2)) != 2:
        raise ValueError("V2 requires n_atst_blocks=1 and gru_layers=2")

    print("=== Stage 1 V2 Training ===")
    print(f"Config: {config_path}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Segment: {segment_len} s")
    print(f"Frame rate: {frame_rate} Hz")
    print(f"ATST blocks: {mcfg.get('n_atst_blocks', 1)}")
    print(f"GRU layers: {mcfg.get('gru_layers', 2)}")
    print(f"pos_weight: {configured_pos_weight}")
    print("Normalization: bounded RMS")
    print("Bandwidth augmentation: enabled")

    meta = Path(metadata_dir)
    recordings_csv = meta / "recordings.csv"
    events_csv = meta / "events.csv"

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cache_hash = hashlib.sha256()
    for fingerprint_path in (
        Path(mcfg["atst_checkpoint"]), recordings_csv, events_csv,
        Path(__file__).parents[1] / "data" / "dataset.py", Path(__file__),
    ):
        if not fingerprint_path.is_file():
            raise FileNotFoundError(f"Required training input not found: {fingerprint_path}")
        with fingerprint_path.open("rb") as fingerprint_file:
            for chunk in iter(lambda: fingerprint_file.read(1024 * 1024), b""):
                cache_hash.update(chunk)
    cache_hash.update(json.dumps({
        "teacher_sample_rate": sample_rate,
        "frame_rate_hz": frame_rate,
        "segment_length_s": segment_len,
        "n_atst_blocks": int(mcfg.get("n_atst_blocks", 1)),
        "bandwidth_aug_prob": float(tcfg.get("bandwidth_aug_prob", 0.5)),
        "bandwidth_cutoff_min": 3400.0,
        "bandwidth_cutoff_max": 4000.0,
        "target_rms": float(tcfg.get("target_rms", 0.05)),
        "min_gain": 0.1,
        "max_gain": 10.0,
        "synthetic_ratio_max": float(tcfg.get("synthetic_ratio_max", 0.3)),
    }, sort_keys=True).encode())
    cache_dir = output / "atst_cache" / cache_hash.hexdigest()[:16]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head_type = str(mcfg.get("head_type", "gru"))
    transformer_heads = int(mcfg.get("transformer_heads", 4))
    transformer_layers = int(mcfg.get("transformer_layers", 2))

    model = FullSEDTeacher(
        atst_checkpoint=mcfg["atst_checkpoint"],
        n_atst_blocks=int(mcfg.get("n_atst_blocks", 1)),
        conv_dim=int(mcfg.get("conv_dim", 256)),
        hidden_dim=int(mcfg.get("hidden_dim", 256)),
        gru_layers=int(mcfg.get("gru_layers", 2)),
        dropout=float(mcfg.get("dropout", 0.2)),
        freeze_encoder=bool(mcfg.get("freeze_encoder", True)),
        head_type=head_type,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
    )

    train_raw_ds = ATSTMosquitoDataset(
        recordings_csv=recordings_csv,
        events_csv=events_csv,
        teacher_sample_rate=sample_rate,
        frame_rate_hz=frame_rate,
        segment_len_s=segment_len,
        split="train",
    )
    val_raw_ds = ATSTMosquitoDataset(
        recordings_csv=recordings_csv,
        events_csv=events_csv,
        teacher_sample_rate=sample_rate,
        frame_rate_hz=frame_rate,
        segment_len_s=segment_len,
        split="validation",
    )

    is_frozen = model.encoder.freeze

    workers = min(8, int(tcfg.get("num_workers", 4)))
    batch_size = int(tcfg.get("batch_size", 64))
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )

    if is_frozen:
        cache_atst_features(model.encoder, train_raw_ds, cache_dir / "train", device=device)
        cache_atst_features(model.encoder, val_raw_ds, cache_dir / "val", device=device)

        train_ds = CachedATSTDataset(cache_dir / "train")
        val_ds = CachedATSTDataset(cache_dir / "val")
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    else:
        train_loader = DataLoader(train_raw_ds, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_raw_ds, shuffle=False, **loader_kwargs)

    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg["weight_decay"]),
    )
    configured_pos_weight = tcfg.get("pos_weight", 1.0)
    if configured_pos_weight == "auto":
        positive_frames = 0.0
        total_frames = 0
        for batch in train_loader:
            target = batch["target"]
            positive_frames += float(target.sum().item())
            total_frames += target.numel()
        if positive_frames == 0:
            raise ValueError("Training split contains no positive frames")
        pos_weight = (total_frames - positive_frames) / positive_frames
        print(f"Automatic positive-frame weight: {pos_weight:.3f}")
    else:
        pos_weight = float(configured_pos_weight)
    loss_type = str(tcfg.get("loss_type", "bce")).lower()
    focal_gamma = float(tcfg.get("focal_gamma", 2.0))
    if loss_type == "focal":
        criterion = FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )

    max_epochs = epochs_override if epochs_override is not None else int(tcfg["max_epochs"])
    best_path = output / "proposer_best.pt"
    legacy_best_path = output / "teacher_v0_best.pt"
    best_val = float("inf")
    patience = int(tcfg.get("early_stopping_patience", 0))
    min_delta = float(tcfg.get("early_stopping_min_delta", 0.0))
    epochs_without_improvement = 0

    amp_dtype_str = str(tcfg.get("amp_dtype", "float16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_str in ("bfloat16", "bf16") else torch.float16

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Device: {device}")
    print(
        f"Model: ATST-Frame (frozen={is_frozen}) -> Conv1D -> {head_type.upper()} -> Linear (AMP {amp_dtype_str})"
    )

    for epoch in range(1, max_epochs + 1):
        if is_frozen:
            model.head.train()
        else:
            model.train()

        total = 0.0
        count = 0

        for batch in train_loader:
            if is_frozen:
                inp = batch["features"].to(device, non_blocking=True)
            else:
                inp = batch["audio"].to(device, non_blocking=True)

            target = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if is_frozen:
                    logits = model.head(inp)
                else:
                    logits = model(inp)

            target_aligned = align_target_to_logits(target, logits)
            loss = criterion(logits.float(), target_aligned.float())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            clip = float(tcfg.get("grad_clip_norm", 5.0))
            if clip > 0:
                nn.utils.clip_grad_norm_(trainable, clip)

            scaler.step(optimizer)
            scaler.update()

            total += float(loss.item())
            count += 1

        train_loss = total / max(1, count)
        val_loss = evaluate_loss(
            model.head if is_frozen else model,
            val_loader,
            criterion,
            device,
            is_cached=is_frozen,
            amp_dtype=amp_dtype,
        )
        print(
            f"Epoch {epoch:03d}/{max_epochs} "
            f"| train={train_loss:.5f} | val={val_loss:.5f}"
        )

        if val_loss < best_val - min_delta:
            best_val = val_loss
            epochs_without_improvement = 0
            ckpt_dict = {
                "format_version": 2,
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "n_atst_blocks": int(mcfg.get("n_atst_blocks", 1)),
                    "conv_dim": int(mcfg.get("conv_dim", 256)),
                    "hidden_dim": int(mcfg.get("hidden_dim", 256)),
                    "gru_layers": int(mcfg.get("gru_layers", 2)),
                    "dropout": float(mcfg.get("dropout", 0.2)),
                    "head_type": head_type,
                    "transformer_heads": transformer_heads,
                    "transformer_layers": transformer_layers,
                },
                "preprocessing": {
                    "sample_rate": sample_rate,
                    "frame_rate_hz": frame_rate,
                    "segment_length_s": segment_len,
                },
                "epoch": epoch,
                "val_loss": val_loss,
            }
            torch.save(ckpt_dict, best_path)
            torch.save(ckpt_dict, legacy_best_path)
            print(f"  saved: {best_path}")
        else:
            epochs_without_improvement += 1
            if patience and epochs_without_improvement >= patience:
                print(f"Early stopping after {patience} epochs without validation improvement.")
                break

    print(f"Best validation loss: {best_val:.5f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="MosSongPlus/src/wingbeat_ml/sed/configs/mosquito_v0.yaml",
    )
    parser.add_argument("--metadata", default="MosSongPlus/metadata")
    parser.add_argument("--output", default="MosSongPlus/artifacts/teacher_v0")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    train_teacher_v0(
        config_path=args.config,
        metadata_dir=args.metadata,
        output_dir=args.output,
        epochs_override=args.epochs,
    )
