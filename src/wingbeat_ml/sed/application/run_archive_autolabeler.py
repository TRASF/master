"""Archive Auto-Labeler Execution Script.

Scans long PCM/WAV recordings in target archive:
'/media/miru4090s/New Volume2/Mosquitoes Dataset/Philip/Lab/'
Runs sliding-window SED teacher inference, continuous probability aggregation, event decoding,
and exports high-confidence auto-label candidates, human-review queues, and ocenaudio SRT/CSV region files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import torch
from wingbeat_ml.sed.annotation.ocenaudio_exporter import export_ocenaudio_regions
from wingbeat_ml.sed.annotation.review_queue import export_review_queues
from wingbeat_ml.sed.annotation.review_wav_exporter import export_review_wav
from wingbeat_ml.sed.inference.long_recording import infer_long_recording, load_sed_teacher_model


def run_autolabeler_archive(
    target_dir: str | Path = "/media/miru4090s/New Volume2/Mosquitoes Dataset/Philip/Lab",
    model_checkpoint: str | Path | None = None,
    output_dir: str | Path = "MosSongPlus/output/archive_detections",
    high_threshold: float = 0.80,
    low_threshold: float = 0.40,
    max_files: int | None = None,
    chunk_len_s: float = 10.0,
    hop_len_s: float = 5.0,
    frame_rate_hz: float = 25.0,
    batch_size: int = 32,
    decoder_high_threshold: float = 0.80,
    decoder_low_threshold: float = 0.40,
    min_duration_s: float = 0.1,
    max_merge_gap_s: float = 0.2,
    audit_ratio: float = 0.05,
    audit_seed: int = 42,
    export_embedded_wavs: bool = False,
    review_wav_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Scan target recording archive and export detections, review queues, and ocenaudio SRT regions."""
    archive_p = Path(target_dir)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    regions_dir = out_p / "regions"
    regions_dir.mkdir(parents=True, exist_ok=True)

    # Find long PCM and WAV recordings
    audio_files = sorted(list(archive_p.rglob("*.pcm")) + list(archive_p.rglob("*.wav")))
    if max_files is not None:
        audio_files = audio_files[:max_files]

    print(f"Found {len(audio_files)} long recording files in target archive.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_model = load_sed_teacher_model(
        model_checkpoint=model_checkpoint,
        device=device,
    )
    checkpoint_path = Path(model_checkpoint) if model_checkpoint is not None else None
    model_hash = ""
    if checkpoint_path and checkpoint_path.is_file():
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as checkpoint_file:
            for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(block)
        model_hash = digest.hexdigest()
    decoder_policy_hash = hashlib.sha256(
        json.dumps(
            {
                "frame_rate_hz": frame_rate_hz,
                "high_threshold": decoder_high_threshold,
                "low_threshold": decoder_low_threshold,
                "min_duration_s": min_duration_s,
                "max_merge_gap_s": max_merge_gap_s,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    all_detections = []
    for idx, audio_p in enumerate(audio_files, 1):
        try:
            events = infer_long_recording(
                wav_path=audio_p,
                model=teacher_model,
                chunk_len_s=chunk_len_s,
                hop_len_s=hop_len_s,
                frame_rate_hz=frame_rate_hz,
                batch_size=batch_size,
                device=device,
                high_threshold=decoder_high_threshold,
                low_threshold=decoder_low_threshold,
                min_duration_s=min_duration_s,
                max_merge_gap_s=max_merge_gap_s,
            )
            file_events = []
            for e in events:
                all_detections.append({
                    "file_path": str(audio_p),
                    "file_name": audio_p.name,
                    "parent_folder": audio_p.parent.name,
                    "start_s": e.start_s,
                    "end_s": e.end_s,
                    "duration_s": e.end_s - e.start_s,
                    "confidence": e.confidence,
                    "mean_score": e.mean_score,
                    "p90_score": e.p90_score,
                    "top25_mean_score": e.top25_mean_score,
                    "event_score_method": "max",
                    "status": "CANDIDATE",
                    "model_hash": model_hash,
                    "decoder_policy_hash": decoder_policy_hash,
                    "gold_calibration_version": "",
                    "label": "mosquito",
                })
                file_events.append({
                    "start_s": e.start_s,
                    "end_s": e.end_s,
                    "confidence": e.confidence,
                })

            export_ocenaudio_regions(
                events=file_events,
                audio_path=audio_p,
                output_dir=regions_dir,
                high_threshold=high_threshold,
            )
            if export_embedded_wavs and file_events:
                relative_path = audio_p.relative_to(archive_p).with_suffix(".mosquito.wav")
                export_review_wav(
                    audio_p,
                    Path(review_wav_dir or out_p / "review_wavs") / relative_path,
                    [{**event, "label": "mosquito"} for event in file_events],
                )

            if idx % 10 == 0 or idx == len(audio_files):
                print(f"Processed {idx}/{len(audio_files)} files...")
        except Exception as ex:
            print(f"Skipping {audio_p.name} due to error: {ex}")

    det_df = pd.DataFrame(all_detections)
    all_cand_path = out_p / "all_candidates.csv"
    det_df.to_csv(all_cand_path, index=False)
    print(f"Saved {len(det_df)} total candidates to {all_cand_path}")

    if not det_df.empty:
        export_review_queues(
            detections_csv=all_cand_path,
            output_dir=out_p,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            sample_audit_ratio=audit_ratio,
            seed=audit_seed,
        )

    return det_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="/media/miru4090s/New Volume2/Mosquitoes Dataset/Philip/Lab")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default="MosSongPlus/output/archive_detections")
    parser.add_argument("--max_files", type=int, default=5)
    args = parser.parse_args()

    run_autolabeler_archive(
        target_dir=args.archive,
        model_checkpoint=args.model,
        output_dir=args.output,
        max_files=args.max_files,
    )
