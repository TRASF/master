"""Active Learning Closed-Loop Command: Ingest completed reviews, retrain Stage 2 verifier, evaluate, and prepare next review batch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd
import torch

from wingbeat_ml.sed.annotation.review_wav_exporter import export_review_wav
from wingbeat_ml.sed.application.evaluate_detector import run_evaluation
from wingbeat_ml.sed.application.ingest_reviewed import read_riff_regions
from wingbeat_ml.sed.data.hard_negatives import ingest_reviewed_candidates_to_verifier
from wingbeat_ml.sed.inference.long_recording import infer_long_recording, load_sed_teacher_model
from wingbeat_ml.sed.training.train_verifier import train_verifier
import wave


def generate_candidate_id(audio_path: str, start_s: float, end_s: float) -> str:
    """Generate deterministic, idempotent SHA256 ID for candidate clip."""
    digest = hashlib.sha256()
    digest.update(f"{Path(audio_path).name}:{start_s:.3f}:{end_s:.3f}".encode("utf-8"))
    return digest.hexdigest()[:16]


def run_active_learning_loop(
    config_path: str | Path = "configs/sed/pipeline.yaml",
    target_dir: str | Path | None = None,
    review_batch_size: int = 20,
) -> None:
    """Execute complete active-learning iteration: completed review ingestion -> Stage 2 retraining -> evaluation -> next review batch."""
    root = Path(__file__).resolve().parents[4]
    cfg_p = Path(config_path)
    if not cfg_p.is_absolute():
        cfg_p = root / cfg_p

    with cfg_p.open("r", encoding="utf-8") as f:
        import yaml
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    inference = cfg["inference"]
    decoder = cfg["decoder"]

    base_review_dir = Path(paths.get("review_wav_dir", "output/review_wavs"))
    pending_dir = base_review_dir / "pending"
    completed_dir = base_review_dir / "completed"
    pending_dir.mkdir(parents=True, exist_ok=True)
    completed_dir.mkdir(parents=True, exist_ok=True)

    metadata_dir = Path(paths["metadata_dir"])
    verifier_csv = metadata_dir / "verifier_samples.csv"
    cand_csv = Path(paths["archive_output"]) / "all_candidates.csv"

    # Step 1: Auto-discover and ingest completed reviews from completed/
    completed_files = list(completed_dir.rglob("*.mosquito.wav"))
    ingested_positives = 0
    ingested_negatives = 0

    if completed_files and cand_csv.is_file():
        cand_df = pd.read_csv(cand_csv)
        new_verifier_rows = []

        for wav_p in completed_files:
            try:
                regions = read_riff_regions(wav_p)
                kept_starts = {round(r["start_s"], 2) for r in regions}
            except Exception:
                continue

            stem = wav_p.name.replace(".mosquito.wav", "")
            matching_cands = cand_df[cand_df["file_name"].astype(str).str.contains(stem)]

            for _, row in matching_cands.iterrows():
                s_start = float(row["start_s"])
                s_end = float(row["end_s"])
                a_path = str(row.get("file_path", row.get("audio_path", "")))
                is_kept = any(abs(s_start - k) < 0.3 for k in kept_starts)
                label_val = 1.0 if is_kept else 0.0
                cand_id = generate_candidate_id(a_path, s_start, s_end)

                if is_kept:
                    ingested_positives += 1
                else:
                    ingested_negatives += 1

                new_verifier_rows.append({
                    "candidate_id": cand_id,
                    "audio_path": a_path,
                    "start_s": s_start,
                    "end_s": s_end,
                    "label": label_val,
                    "split": "train",
                    "review_source": wav_p.name,
                })

        if new_verifier_rows:
            new_df = pd.DataFrame(new_verifier_rows)
            if verifier_csv.is_file():
                existing_df = pd.read_csv(verifier_csv)
                merged = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(
                    subset=["audio_path", "start_s", "end_s"]
                )
            else:
                merged = new_df
            merged.to_csv(verifier_csv, index=False)

    print("=== Active Learning Ingestion Summary ===")
    print(f"Completed review files:     {len(completed_files)}")
    print(f"Ingested target positives:  {ingested_positives}")
    print(f"Ingested hard negatives:    {ingested_negatives}")

    # Step 2: Retrain Stage 2 Verifier ONLY
    print("\n=== Retraining Stage 2 Verifier Only ===")
    atst_ckpt = cfg.get("model", {}).get("atst_checkpoint", "checkpoints/atstframe_base.ckpt")
    output_dir = Path(paths["teacher_output"])

    if verifier_csv.is_file():
        verifier_ckpt = train_verifier(
            atst_checkpoint=atst_ckpt,
            samples_csv=verifier_csv,
            output_dir=output_dir,
            epochs=15,
        )
        print(f"Updated verifier checkpoint: {verifier_ckpt}")

    # Step 3: Evaluate on Untouched Target Files
    print("\n=== Active Learning Target Evaluation ===")
    archive_dir = Path(target_dir or paths["archive_dir"])
    archive_files = sorted(list(archive_dir.rglob("*.pcm")) + list(archive_dir.rglob("*.wav")))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proposer_ckpt = output_dir / "proposer_best.pt"
    verifier_ckpt = output_dir / "verifier_best.pt"

    teacher_model = load_sed_teacher_model(model_checkpoint=proposer_ckpt, device=device)
    eval_files = archive_files[:min(10, len(archive_files))]

    total_proposals = 0
    total_retained = 0
    total_duration_s = 0.0

    for a_file in eval_files:
        try:
            with wave.open(str(a_file), "rb") if a_file.suffix.lower() == ".wav" else open(str(a_file), "rb") as af:
                dur = a_file.stat().st_size / (2 * 44100) if a_file.suffix.lower() != ".wav" else af.getnframes() / float(af.getframerate())
            total_duration_s += dur

            props = infer_long_recording(
                wav_path=a_file,
                model=teacher_model,
                verifier_checkpoint=verifier_ckpt if verifier_ckpt.is_file() else None,
                chunk_len_s=float(inference["chunk_length_s"]),
                hop_len_s=float(inference["hop_length_s"]),
                frame_rate_hz=float(inference["frame_rate_hz"]),
                batch_size=int(inference.get("batch_size", 32)),
                device=device,
            )
            total_proposals += len(props)
            total_retained += sum(1 for p in props if p.confidence >= 0.5)
        except Exception:
            continue

    total_hours = total_duration_s / 3600.0
    rejected = max(0, total_proposals - total_retained)
    rejection_rate = (rejected / max(1, total_proposals)) * 100.0

    print(f"Audio processed:             {total_hours:.2f} h")
    print(f"Stage 1 proposals:          {total_proposals}")
    print(f"Stage 1 proposals/hour:     {total_proposals / max(1e-4, total_hours):.1f}")
    print(f"Stage 2 rejected:           {rejected}")
    print(f"Stage 2 retained:           {total_retained}")
    print(f"Stage 2 rejection rate:      {rejection_rate:.1f}%")
    print(f"Final candidates/hour:       {total_retained / max(1e-4, total_hours):.1f}")
    print("AUTO_ACCEPT:               DISABLED")

    # Step 4: Generate Next Review Batch into pending/
    print("\n=== Generating Next Review Batch ===")
    next_batch_files = archive_files[min(10, len(archive_files)) : min(10 + review_batch_size, len(archive_files))]
    for n_file in next_batch_files:
        try:
            events = infer_long_recording(
                wav_path=n_file,
                model=teacher_model,
                verifier_checkpoint=verifier_ckpt if verifier_ckpt.is_file() else None,
                chunk_len_s=float(inference["chunk_length_s"]),
                hop_len_s=float(inference["hop_length_s"]),
                frame_rate_hz=float(inference["frame_rate_hz"]),
                batch_size=int(inference.get("batch_size", 32)),
                device=device,
            )
            if events:
                event_dicts = [{"start_s": e.start_s, "end_s": e.end_s, "confidence": e.confidence, "label": "mosquito"} for e in events]
                out_wav = pending_dir / f"{n_file.stem}.mosquito.wav"
                export_review_wav(n_file, out_wav, event_dicts)
        except Exception:
            continue

    print(f"Generated {len(next_batch_files)} review files in {pending_dir}")
    print("\nWorkflow Instructions:")
    print("  1. Review .mosquito.wav files in review_wavs/pending/")
    print("  2. Move completed files to review_wavs/completed/")
    print("  3. Run: ./sed.sh learn")
