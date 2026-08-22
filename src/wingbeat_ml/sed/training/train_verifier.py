"""Training workflow for Stage 2 Verifier (event classifier with hard-negative support)."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from wingbeat_ml.sed.models.verifier import Stage2Verifier
from wingbeat_ml.sed.data.dataset import compute_recording_bounded_gain
import wave
from wingbeat_ml.data.synthetic import decode_pcm_bytes
import torchaudio.transforms as T


class VerifierDataset(Dataset):
    """Dataset for Stage 2 clip-level verification (positives vs environmental & hard negatives)."""

    def __init__(
        self,
        samples_df: pd.DataFrame,
        sample_rate: int = 16000,
        crop_duration_s: float = 2.0,
        target_rms: float = 0.05,
    ):
        self.samples = samples_df.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.crop_duration_s = crop_duration_s
        self.crop_samples = int(sample_rate * crop_duration_s)
        self.target_rms = target_rms

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.samples.iloc[idx]
        path = str(row["audio_path"])
        start_s = float(row.get("start_s", 0.0))
        label = float(row["label"])  # 1.0 = positive, 0.0 = negative

        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            wf.setpos(min(int(start_s * sr), wf.getnframes()))
            data = wf.readframes(int(self.crop_duration_s * sr))
            signal = decode_pcm_bytes(data, wf.getsampwidth())

        if n_ch > 1:
            signal = signal[: len(signal) - (len(signal) % n_ch)].reshape(-1, n_ch)
            signal = signal[:, 0]

        gain = compute_recording_bounded_gain(signal, target_rms=self.target_rms)
        signal = signal * gain

        audio = torch.from_numpy(signal.copy()).float()
        if sr != self.sample_rate:
            audio = T.Resample(sr, self.sample_rate)(audio)

        if len(audio) < self.crop_samples:
            audio = torch.nn.functional.pad(audio, (0, self.crop_samples - len(audio)))
        else:
            audio = audio[: self.crop_samples]

        return {"audio": audio, "label": torch.tensor([label], dtype=torch.float32)}


def train_verifier(
    atst_checkpoint: str | Path,
    samples_csv: str | Path,
    output_dir: str | Path,
    epochs: int = 15,
    lr: float = 1e-4,
    batch_size: int = 64,
) -> Path:
    """Train Stage 2 Verifier model with explicit dataset metrics & validation tracking."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    import numpy as np

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_path / "verifier_best.pt"

    df = pd.read_csv(samples_csv)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]

    t_pos = int((train_df["label"] == 1.0).sum())
    t_neg = int((train_df["label"] == 0.0).sum())
    v_pos = int((val_df["label"] == 1.0).sum())
    v_neg = int((val_df["label"] == 0.0).sum())

    print("Verifier dataset:")
    print(f"  train positive: {t_pos} | train negative: {t_neg}")
    print(f"  val positive:   {v_pos} | val negative:   {v_neg}")

    train_ds = VerifierDataset(train_df)
    val_ds = VerifierDataset(val_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Stage2Verifier(atst_checkpoint=atst_checkpoint, input_sample_rate=16000).to(device)

    optimizer = torch.optim.AdamW(model.mlp.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            audio = batch["audio"].to(device)
            label = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(audio)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device)
                label = batch["label"].to(device)
                logits = model(audio)
                val_loss += criterion(logits, label).item()
                all_logits.extend(logits.squeeze(-1).cpu().numpy())
                all_labels.extend(label.squeeze(-1).cpu().numpy())

        val_loss /= max(1, len(val_loader))
        all_probs = 1.0 / (1.0 + np.exp(-np.array(all_logits)))
        all_targets = np.array(all_labels)

        auroc = float(roc_auc_score(all_targets, all_probs)) if len(np.unique(all_targets)) > 1 else 0.5
        auprc = float(average_precision_score(all_targets, all_probs)) if len(np.unique(all_targets)) > 1 else 0.5
        preds_50 = (all_probs >= 0.5).astype(int)
        tp = int(((preds_50 == 1) & (all_targets == 1.0)).sum())
        fp = int(((preds_50 == 1) & (all_targets == 0.0)).sum())
        fn = int(((preds_50 == 0) & (all_targets == 1.0)).sum())
        prec_50 = tp / max(1, tp + fp)
        rec_50 = tp / max(1, tp + fn)

        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"| AUROC={auroc:.4f} | AUPRC={auprc:.4f} | Prec@0.5={prec_50:.4f} | Rec@0.5={rec_50:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "format_version": 2,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_auroc": auroc,
                    "val_auprc": auprc,
                    "val_precision_50": prec_50,
                    "val_recall_50": rec_50,
                    "epoch": epoch,
                },
                best_ckpt,
            )
            print(f"  saved: {best_ckpt}")

    return best_ckpt
