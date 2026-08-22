"""ATST Feature Caching for Frozen Encoder SED Training.

Pre-extracts frozen ATST embeddings [T, 768] once to disk
to eliminate repeated 12-block Transformer forward passes during head training.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class CachedATSTDataset(Dataset):
    """Dataset serving pre-extracted ATST feature tensors [T, D] and target labels [T, 1]."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.files = sorted(list(self.cache_dir.glob("sample_*.pt")))
        if not self.files:
            raise FileNotFoundError(f"No cached ATST samples found in {self.cache_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        return {
            "features": data["features"].float(),
            "target": data["target"].float(),
        }


def cache_atst_features(
    encoder: nn.Module,
    dataset: Dataset,
    cache_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    device: torch.device | None = None,
    force_recompute: bool = False,
) -> Path:
    """Pre-extract ATST features in float32 and save clean tensors to cache_dir."""
    cache_path = Path(cache_dir)
    if force_recompute and cache_path.exists():
        shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    existing = list(cache_path.glob("sample_*.pt"))
    if not force_recompute and len(existing) == len(dataset) and len(dataset) > 0:
        print(f"Using {len(existing)} existing cached ATST features in {cache_path}")
        return cache_path

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder.to(device)
    encoder.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Caching ATST features for {len(dataset)} samples to {cache_path}...")
    sample_count = 0

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            features = encoder(audio)  # [B, T, D]
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            for b in range(features.shape[0]):
                out_file = cache_path / f"sample_{sample_count:06d}.pt"
                torch.save(
                    {
                        "features": features[b].cpu(),
                        "target": target[b].cpu(),
                    },
                    out_file,
                )
                sample_count += 1

    print(f"Successfully cached {sample_count} samples to {cache_path}")
    return cache_path
