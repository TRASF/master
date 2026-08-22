"""Gold benchmark selection and recording/event table creation."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd
import soundfile as sf


RECORDING_COLUMNS = [
    "file_name",
    "duration_s",
    "sha256",
    "split",
    "is_exhaustively_annotated",
    "verified_clean_negative",
    "annotation_version",
]
EVENT_COLUMNS = ["file_name", "event_id", "start_s", "end_s", "label"]


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def build_gold_benchmarks(
    metadata_dir: str | Path,
    gold_dir: str | Path,
    max_val_files: int = 5,
    max_test_files: int = 5,
    min_recording_duration_s: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select validation/test recordings without fabricating event labels."""
    recordings_path = Path(metadata_dir) / "recordings.csv"
    if not recordings_path.exists():
        raise FileNotFoundError(f"Missing recordings.csv at {recordings_path}")

    source = pd.read_csv(recordings_path)
    selected = {
        "validation": source[
            (source["split"] == "validation") & (source["duration_s"] > min_recording_duration_s)
        ].head(max_val_files),
        "test": source[
            (source["split"] == "test") & (source["duration_s"] > min_recording_duration_s)
        ].head(max_test_files),
    }

    outputs: list[pd.DataFrame] = []
    for split, rows in selected.items():
        target = Path(gold_dir) / f"gold_{split}"
        audio_dir = target / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        existing_audio = [path for path in audio_dir.iterdir() if path.is_file()]
        if not existing_audio:
            for _, row in rows.iterrows():
                source_audio = Path(row["path"])
                destination = audio_dir / source_audio.name
                if source_audio.exists():
                    shutil.copy2(source_audio, destination)
            existing_audio = [path for path in audio_dir.iterdir() if path.is_file()]

        recording_csv = target / "recordings.csv"
        if recording_csv.exists():
            recording_table = pd.read_csv(recording_csv)
            print(f"Preserving existing recording verification: {recording_csv}")
        else:
            duration_by_name = {
                Path(row.path).name: float(row.duration_s) for row in source.itertuples()
            }
            recording_table = pd.DataFrame(
                {
                    "file_name": [path.name for path in existing_audio],
                    "duration_s": [
                        duration_by_name.get(path.name, float(sf.info(path).duration))
                        for path in existing_audio
                    ],
                    "split": split,
                    "is_exhaustively_annotated": False,
                    "verified_clean_negative": False,
                    "annotation_version": "v1",
                },
                columns=RECORDING_COLUMNS,
            )

        audio_by_name = {path.name: path for path in existing_audio}
        recording_table["sha256"] = recording_table["file_name"].map(
            lambda name: _sha256(audio_by_name[name])
        )
        recording_table = recording_table[RECORDING_COLUMNS]
        recording_table.to_csv(recording_csv, index=False)

        event_csv = target / "events.csv"
        if not event_csv.exists():
            legacy_csv = target / "annotations.csv"
            if legacy_csv.exists():
                legacy = pd.read_csv(legacy_csv)
                verified = (
                    "verified_by_human" in legacy.columns
                    and not legacy.empty
                    and legacy["verified_by_human"].eq(True).all()
                )
                if verified:
                    events = legacy[["file_name", "start_s", "end_s", "label"]].copy()
                    events.insert(1, "event_id", [f"e{i:06d}" for i in range(1, len(events) + 1)])
                else:
                    events = pd.DataFrame(columns=EVENT_COLUMNS)
            else:
                events = pd.DataFrame(columns=EVENT_COLUMNS)
            events.to_csv(event_csv, index=False)

        outputs.append(recording_table)
        print(f"Prepared {len(recording_table)} gold {split} recordings in {target}")

    return outputs[0], outputs[1]


if __name__ == "__main__":
    build_gold_benchmarks(
        metadata_dir="/home/miru4090s/clones/Master Thesises/MosSongPlus/metadata",
        gold_dir="/home/miru4090s/clones/Master Thesises/MosSongPlus/dataset",
    )
