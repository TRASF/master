"""Ocenaudio-compatible SRT & CSV exporter for SED mosquito detections."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def seconds_to_srt_time(seconds: float) -> str:
    """Format seconds float into SRT time string HH:MM:SS,mmm."""
    milliseconds = int(round(seconds * 1000))

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    secs = milliseconds // 1000
    milliseconds %= 1000

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def export_ocenaudio_regions(
    events: list[dict],
    audio_path: str | Path,
    output_dir: str | Path,
    high_threshold: float = 0.8,
) -> tuple[Path, Path]:
    """Export event list to ocenaudio SRT region file and analysis CSV file.

    events: list of dicts with keys 'start_s', 'end_s', 'confidence'
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = audio_path.stem

    srt_path = output_dir / f"{stem}.mosquito.srt"
    csv_path = output_dir / f"{stem}.mosquito.csv"

    lines = []
    for index, event in enumerate(events, start=1):
        start_s = float(event["start_s"])
        end_s = float(event["end_s"])
        confidence = float(event.get("confidence", 1.0))
        status = str(event.get("status", "REVIEW"))
        label = f"MOS | conf={confidence:.4f} | {status}"

        lines.extend([
            str(index),
            f"{seconds_to_srt_time(start_s)} --> {seconds_to_srt_time(end_s)}",
            label,
            "",
        ])

    srt_path.write_text("\n".join(lines), encoding="utf-8")

    df_data = []
    for event in events:
        start_s = float(event["start_s"])
        end_s = float(event["end_s"])
        confidence = float(event.get("confidence", 1.0))
        status = str(event.get("status", "REVIEW"))
        df_data.append({
            "audio_file": audio_path.name,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": end_s - start_s,
            "confidence": confidence,
            "status": status,
        })

    dataframe = pd.DataFrame(df_data)
    if dataframe.empty:
        dataframe = pd.DataFrame(columns=["audio_file", "start_s", "end_s", "duration_s", "confidence", "status"])
    dataframe.to_csv(csv_path, index=False)

    return srt_path, csv_path
