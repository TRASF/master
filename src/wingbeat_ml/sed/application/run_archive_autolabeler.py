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
import wave
import pandas as pd
import torch
from wingbeat_ml.sed.annotation.ocenaudio_exporter import export_ocenaudio_regions
from wingbeat_ml.sed.annotation.review_queue import export_review_queues
from wingbeat_ml.sed.annotation.review_wav_exporter import export_review_wav
from wingbeat_ml.sed.inference.long_recording import infer_long_recording, load_sed_teacher_model


def run_autolabeler_archive(
    target_dir: str | Path = "/media/miru4090s/New Volume2/Mosquitoes Dataset/Philip/Lab",
    model_checkpoint: str | Path | None = None,
    verifier_checkpoint: str | Path | None = None,
    output_dir: str | Path = "MosSongPlus/output/archive_detections",
    high_threshold: float = 0.80,
    low_threshold: float = 0.40,
    max_files: int | None = None,
    chunk_len_s: float = 4.0,
    hop_len_s: float = 1.0,
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

    verifier_hash = ""
    if verifier_checkpoint and Path(verifier_checkpoint).is_file():
        v_digest = hashlib.sha256()
        with Path(verifier_checkpoint).open("rb") as vf:
            for block in iter(lambda: vf.read(1024 * 1024), b""):
                v_digest.update(block)
        verifier_hash = v_digest.hexdigest()[:16]
        print(f"Loaded verifier checkpoint: {verifier_checkpoint} (SHA256: {verifier_hash})")

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
    total_audio_duration_s = 0.0
    total_stage1_proposals = 0
    total_retained_candidates = 0

    for idx, audio_p in enumerate(audio_files, 1):
        try:
            # Measure audio file duration
            with wave.open(str(audio_p), "rb") if audio_p.suffix.lower() == ".wav" else open(str(audio_p), "rb") as af:
                if audio_p.suffix.lower() == ".wav":
                    dur_s = af.getnframes() / float(af.getframerate())
                else:
                    dur_s = audio_p.stat().st_size / (2 * 44100)  # estimate 16-bit 44.1k
            total_audio_duration_s += dur_s

            events = infer_long_recording(
                wav_path=audio_p,
                model=teacher_model,
                verifier_checkpoint=verifier_checkpoint,
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
            total_stage1_proposals += len(events)
            file_events = []
            for e in events:
                s1_score = float(e.mean_score) if e.mean_score is not None else float(e.confidence)
                v_score = float(e.confidence)
                retained = v_score >= 0.5
                if retained:
                    total_retained_candidates += 1

                all_detections.append({
                    "file_path": str(audio_p),
                    "file_name": audio_p.name,
                    "parent_folder": audio_p.parent.name,
                    "start_s": e.start_s,
                    "end_s": e.end_s,
                    "duration_s": e.end_s - e.start_s,
                    "confidence": v_score,
                    "stage1_score": s1_score,
                    "verifier_score": v_score,
                    "status": "REVIEW" if retained else "DISCARD",
                    "model_hash": model_hash,
                    "decoder_policy_hash": decoder_policy_hash,
                    "gold_calibration_version": "v2",
                    "label": "mosquito",
                })
                if retained:
                    file_events.append({
                        "start_s": e.start_s,
                        "end_s": e.end_s,
                        "confidence": v_score,
                        "status": "REVIEW",
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

    total_hours = total_audio_duration_s / 3600.0
    stage2_rejected = max(0, total_stage1_proposals - total_retained_candidates)
    rejection_rate = (stage2_rejected / max(1, total_stage1_proposals)) * 100.0

    scores = [d["verifier_score"] for d in all_detections] if all_detections else []

    print("\n=== Archive Autolabeler Execution Summary ===")
    print(f"Audio processed:             {total_hours:.2f} h")
    print(f"Stage 1 proposals:          {total_stage1_proposals}")
    print(f"Stage 1 proposals/hour:     {total_stage1_proposals / max(1e-4, total_hours):.1f}")
    print(f"Stage 2 rejected:           {stage2_rejected}")
    print(f"Stage 2 retained:           {total_retained_candidates}")
    print(f"Stage 2 rejection rate:      {rejection_rate:.1f}%")
    print(f"Final candidates/hour:       {total_retained_candidates / max(1e-4, total_hours):.1f}")
    print("AUTO_ACCEPT:               DISABLED")

    if scores:
        import numpy as np
        p_arr = np.array(scores)
        print("\nVerifier Score Distribution:")
        print(f"  min:    {np.min(p_arr):.4f}")
        print(f"  p01:    {np.percentile(p_arr, 1):.4f}")
        print(f"  p05:    {np.percentile(p_arr, 5):.4f}")
        print(f"  p10:    {np.percentile(p_arr, 10):.4f}")
        print(f"  p25:    {np.percentile(p_arr, 25):.4f}")
        print(f"  median: {np.median(p_arr):.4f}")
        print(f"  p75:    {np.percentile(p_arr, 75):.4f}")
        print(f"  p90:    {np.percentile(p_arr, 90):.4f}")
        print(f"  p95:    {np.percentile(p_arr, 95):.4f}")
        print(f"  p99:    {np.percentile(p_arr, 99):.4f}")
        print(f"  max:    {np.max(p_arr):.4f}\n")

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
