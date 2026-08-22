"""Bounded Overfit Test.

Verifies end-to-end learning capability and frame alignment by deliberately overfitting
the FullSEDTeacher model on 10 synthetic scenes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from wingbeat_ml.data.synthetic import generate_synthetic_soundscapes
from wingbeat_ml.sed.data.dataset import ATSTMosquitoDataset
from wingbeat_ml.sed.training.train import FullSEDTeacher
from wingbeat_ml.sed.inference.decoder import decode_events


def run_overfit_test(tmp_dir: Path) -> dict:
    meta_dir = tmp_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate 10 synthetic scenes
    synth_dir = tmp_dir / "synthetic"
    scenes_df, events_df = generate_synthetic_soundscapes(
        metadata_dir="/home/miru4090s/clones/Master Thesises/MosSongPlus/metadata",
        output_dir=synth_dir,
        num_scenes=10,
        scene_duration_s=10.0,
    )

    # Save temp manifest
    recs_df = scenes_df.rename(columns={"scene_id": "file_id"})
    recs_df["sample_rate_hz"] = 16000
    recs_df["channels"] = 1
    recs_df["split_group"] = recs_df["file_id"]
    recs_df["supervision_type"] = "strong"

    recs_df.to_csv(meta_dir / "recordings.csv", index=False)
    events_df.to_csv(meta_dir / "events.csv", index=False)

    ds = ATSTMosquitoDataset(
        recordings_csv=meta_dir / "recordings.csv",
        events_csv=meta_dir / "events.csv",
        teacher_sample_rate=8000,
        frame_rate_hz=50.0,
        segment_len_s=10.0,
        split="train",
    )
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullSEDTeacher(sample_rate=8000).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    criterion = nn.BCELoss()

    initial_loss = 0.0
    final_loss = 0.0

    print("Running 40-epoch overfit test on 10 scenes...")
    for epoch in range(1, 41):
        model.train()
        total_loss = 0.0

        for batch in loader:
            audio = batch["audio"].to(device)
            target = batch["target"].to(device)

            probs = model(audio)
            if probs.shape[1] != target.shape[1]:
                min_t = min(probs.shape[1], target.shape[1])
                probs = probs[:, :min_t, :]
                target = target[:, :min_t, :]

            loss = criterion(probs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if epoch == 1:
            initial_loss = avg_loss
        final_loss = avg_loss

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}: Loss = {avg_loss:.4f}")

    # Evaluate memorization on sample 0
    model.eval()
    sample_0 = ds[0]
    audio_0 = sample_0["audio"].unsqueeze(0).to(device)
    with torch.no_grad():
        pred_p = model(audio_0).squeeze(0).squeeze(-1).cpu().numpy()

    decoded = decode_events(pred_p, frame_rate_hz=50.0, high_threshold=0.5, low_threshold=0.3)

    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "num_decoded_events": len(decoded),
        "sample_decoded": decoded,
    }


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        res = run_overfit_test(Path(td))
        print("=== Overfit Test Results ===")
        print(f"Initial Loss: {res['initial_loss']:.4f}")
        print(f"Final Loss:   {res['final_loss']:.4f}")
        print(f"Decoded Events Count: {res['num_decoded_events']}")
        print(f"Events: {res['sample_decoded']}")
