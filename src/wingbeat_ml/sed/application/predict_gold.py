"""Generate same-file event predictions for every gold recording."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from wingbeat_ml.sed.inference.long_recording import infer_long_recording, load_sed_teacher_model


PREDICTION_COLUMNS = [
    "file_name",
    "start_s",
    "end_s",
    "confidence",
    "mean_score",
    "p90_score",
    "top25_mean_score",
]


def predict_gold(
    gold_dir: str | Path,
    model_checkpoint: str | Path,
    output_csv: str | Path,
    chunk_len_s: float = 10.0,
    hop_len_s: float = 5.0,
    frame_rate_hz: float = 25.0,
    batch_size: int = 32,
    high_threshold: float = 0.8,
    low_threshold: float = 0.4,
    min_duration_s: float = 0.1,
    max_merge_gap_s: float = 0.2,
) -> pd.DataFrame:
    gold = Path(gold_dir)
    recordings = pd.read_csv(gold / "recordings.csv")
    audio_dir = gold / "audio"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sed_teacher_model(model_checkpoint=model_checkpoint, device=device)

    rows = []
    for recording in recordings.itertuples():
        audio_path = audio_dir / recording.file_name
        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing gold audio: {audio_path}")
        events = infer_long_recording(
            wav_path=audio_path,
            model=model,
            chunk_len_s=chunk_len_s,
            hop_len_s=hop_len_s,
            frame_rate_hz=frame_rate_hz,
            batch_size=batch_size,
            device=device,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            min_duration_s=min_duration_s,
            max_merge_gap_s=max_merge_gap_s,
        )
        rows.extend(
            {
                "file_name": recording.file_name,
                "start_s": event.start_s,
                "end_s": event.end_s,
                "confidence": event.confidence,
                "mean_score": event.mean_score,
                "p90_score": event.p90_score,
                "top25_mean_score": event.top25_mean_score,
            }
            for event in events
        )

    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    predictions.to_csv(temporary, index=False)
    temporary.replace(destination)
    print(f"Saved {len(predictions)} gold predictions to {destination}")
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    predict_gold(arguments.gold_dir, arguments.model, arguments.output)
