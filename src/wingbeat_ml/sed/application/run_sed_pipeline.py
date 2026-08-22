"""YAML-configured mosquito SED workflow stages."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import pandas as pd
import torch
import yaml

from wingbeat_ml.sed.application.evaluate_detector import run_evaluation
from wingbeat_ml.sed.application.predict_gold import predict_gold
from wingbeat_ml.sed.application.run_archive_autolabeler import run_autolabeler_archive
from wingbeat_ml.data.gold_validation import build_gold_benchmarks
from wingbeat_ml.sed.data.manifest import generate_sed_manifests
from wingbeat_ml.sed.data.split import split_sed_manifest
from wingbeat_ml.data.synthetic import add_synthetic_to_metadata, generate_synthetic_soundscapes
from wingbeat_ml.sed.training.train import train_teacher_v0


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return config


def run_pipeline(
    config_path: str | Path,
    epochs: int | None = None,
    max_files: int | None = None,
    stages: tuple[str, ...] = ("train", "evaluate", "label"),
    target_dir: str | Path | None = None,
) -> None:
    """Run only requested independent workflow stages."""
    root = Path(__file__).resolve().parents[4]
    os.chdir(root)
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = load_yaml(config_path)

    unknown = set(stages) - {"train", "evaluate", "label"}
    if unknown:
        raise ValueError(f"Unknown SED stages: {sorted(unknown)}")

    paths = cfg["paths"]
    inference = cfg["inference"]
    decoder = cfg["decoder"]
    review = cfg["review"]
    evaluation = cfg["evaluation"]
    if "evaluate" in stages and len(stages) > 1 and not evaluation.get("enabled", False):
        print("Evaluation skipped by evaluation.enabled=false")
        stages = tuple(stage for stage in stages if stage != "evaluate")
    atst_checkpoint_path = cfg.get("model", {}).get("atst_checkpoint", "checkpoints/atstframe_base.ckpt")
    atst_checkpoint = Path(atst_checkpoint_path)
    if not atst_checkpoint.is_file():
        raise FileNotFoundError(
            f"Official pretrained ATST-Frame checkpoint missing: {atst_checkpoint.resolve()}"
        )

    if "train" in stages:
        manifest = cfg["manifest"]
        split = cfg["split"]
        gold = cfg["gold"]
        synthetic = cfg["synthetic"]

        print("=== Prepare: manifests, splits, and gold benchmark ===")
        generate_sed_manifests(
            msb_dir=paths["msb_dir"],
            philip_dir=paths["philip_dir"],
            recordings_dir=paths["recordings_dir"],
            output_dir=paths["metadata_dir"],
            trang_dir=paths["trang_dir"],
            active_channel_rms_threshold=float(manifest["active_channel_rms_threshold"]),
        )
        split_sed_manifest(
            paths["metadata_dir"],
            train_ratio=float(split["train_ratio"]),
            val_ratio=float(split["val_ratio"]),
            test_ratio=float(split["test_ratio"]),
            seed=int(split["seed"]),
        )
        build_gold_benchmarks(
            metadata_dir=paths["metadata_dir"],
            gold_dir=paths["gold_dir"],
            max_val_files=int(gold["max_validation_files"]),
            max_test_files=int(gold["max_test_files"]),
            min_recording_duration_s=float(gold["min_recording_duration_s"]),
        )

        print("=== Prepare: SyntheticV2 ===")
        synthetic_scenes, synthetic_events = generate_synthetic_soundscapes(
            metadata_dir=paths["metadata_dir"],
            output_dir=paths["synthetic_dir"],
            num_scenes=int(synthetic["num_scenes"]),
            scene_duration_s=float(synthetic["scene_duration_s"]),
            snr_db_range=(float(synthetic["snr_db_min"]), float(synthetic["snr_db_max"])),
            seed=int(synthetic["seed"]),
            sample_rate_hz=int(synthetic["sample_rate_hz"]),
            event_count_weights=tuple(float(value) for value in synthetic["event_count_weights"]),
            fade_ms=float(synthetic["fade_ms"]),
        )
        add_synthetic_to_metadata(
            synthetic_scenes,
            synthetic_events,
            paths["metadata_dir"],
            sample_rate_hz=int(synthetic["sample_rate_hz"]),
        )

        print("=== Train Stage 1 Proposer ===")
        configured_epochs = cfg.get("training", {}).get("epochs_override")
        train_teacher_v0(
            config_path=config_path,
            metadata_dir=paths["metadata_dir"],
            output_dir=paths["teacher_output"],
            epochs_override=epochs if epochs is not None else configured_epochs,
        )
        proposer_ckpt = Path(paths["teacher_output"]) / "proposer_best.pt"
        print(f"Best proposer: {proposer_ckpt}")

        print("=== Train Stage 2 Verifier ===")
        from wingbeat_ml.sed.training.train_verifier import train_verifier
        verifier_csv = Path(paths["metadata_dir"]) / "verifier_samples.csv"
        atst_ckpt = cfg.get("model", {}).get("atst_checkpoint", "checkpoints/atstframe_base.ckpt")

        # Build verifier training manifest if needed
        recs_df = pd.read_csv(Path(paths["metadata_dir"]) / "recordings.csv")
        evts_df = pd.read_csv(Path(paths["metadata_dir"]) / "events.csv")
        v_rows = []
        for _, r in recs_df.iterrows():
            sup = str(r["supervision_type"])
            split_name = str(r["split"])
            path_str = str(r["path"])
            dur = float(r.get("duration_s", 0.0))

            if sup == "strong":
                r_evts = evts_df[evts_df["recording_file_id"] == str(r["file_id"])].sort_values("start_s")
                last_end = 0.0
                for _, e in r_evts.iterrows():
                    e_start = float(e["start_s"])
                    e_end = float(e["end_s"])
                    mid_s = (e_start + e_end) / 2.0
                    crop_start = max(0.0, mid_s - 1.0)
                    v_rows.append({
                        "audio_path": path_str,
                        "start_s": crop_start,
                        "end_s": crop_start + 2.0,
                        "label": 1.0,
                        "split": split_name,
                    })
                    if e_start - last_end >= 2.0:
                        v_rows.append({
                            "audio_path": path_str,
                            "start_s": last_end,
                            "end_s": last_end + 2.0,
                            "label": 0.0,
                            "split": split_name,
                        })
                    last_end = max(last_end, e_end)
                if dur - last_end >= 2.0:
                    v_rows.append({
                        "audio_path": path_str,
                        "start_s": last_end,
                        "end_s": last_end + 2.0,
                        "label": 0.0,
                        "split": split_name,
                    })
            elif sup == "positive_clip":
                v_rows.append({
                    "audio_path": path_str,
                    "start_s": 0.0,
                    "end_s": min(2.0, dur) if dur > 0 else 2.0,
                    "label": 1.0,
                    "split": split_name,
                })
            elif sup == "negative":
                curr = 0.0
                max_dur = dur if dur > 0 else 2.0
                while curr + 2.0 <= max_dur:
                    v_rows.append({
                        "audio_path": path_str,
                        "start_s": curr,
                        "end_s": curr + 2.0,
                        "label": 0.0,
                        "split": split_name,
                    })
                    curr += 2.0

        if v_rows:
            pd.DataFrame(v_rows).to_csv(verifier_csv, index=False)

        if verifier_csv.is_file():
            verifier_ckpt = train_verifier(
                atst_checkpoint=atst_ckpt,
                samples_csv=verifier_csv,
                output_dir=paths["teacher_output"],
                epochs=15,
            )
            print(f"Best verifier: {verifier_ckpt}")
        else:
            print("No verifier samples found; skipping Stage 2 training.")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_checkpoint = Path(paths["teacher_output"]) / "teacher_v0_best.pt"

    if "evaluate" in stages:
        print("=== Evaluate: gold prediction and metrics ===")
        predictions_csv = evaluation["predictions_csv"]
        predict_gold(
            gold_dir=evaluation["gold_dir"],
            model_checkpoint=model_checkpoint,
            output_csv=predictions_csv,
            chunk_len_s=float(inference["chunk_length_s"]),
            hop_len_s=float(inference["hop_length_s"]),
            frame_rate_hz=float(inference["frame_rate_hz"]),
            batch_size=int(inference.get("batch_size", 32)),
            high_threshold=float(decoder["high_threshold"]),
            low_threshold=float(decoder["low_threshold"]),
            min_duration_s=float(decoder["min_duration_s"]),
            max_merge_gap_s=float(decoder["max_merge_gap_s"]),
        )
        run_evaluation(
            evaluation["gold_dir"],
            predictions_csv,
            iou_threshold=float(evaluation["iou_threshold"]),
            high_conf_threshold=float(evaluation["high_confidence_threshold"]),
            score_column=evaluation.get("event_score_method", "confidence"),
        )

    if "label" in stages:
        print("=== Label: archive candidate generation ===")
        run_autolabeler_archive(
            target_dir=target_dir or paths["archive_dir"],
            model_checkpoint=model_checkpoint,
            output_dir=paths["archive_output"],
            high_threshold=float(review["high_threshold"]),
            low_threshold=float(review["low_threshold"]),
            max_files=max_files if max_files is not None else inference.get("max_files"),
            chunk_len_s=float(inference["chunk_length_s"]),
            hop_len_s=float(inference["hop_length_s"]),
            frame_rate_hz=float(inference["frame_rate_hz"]),
            batch_size=int(inference.get("batch_size", 32)),
            decoder_high_threshold=float(decoder["high_threshold"]),
            decoder_low_threshold=float(decoder["low_threshold"]),
            min_duration_s=float(decoder["min_duration_s"]),
            max_merge_gap_s=float(decoder["max_merge_gap_s"]),
            audit_ratio=float(review["audit_ratio"]),
            audit_seed=int(review["seed"]),
            export_embedded_wavs=bool(review.get("export_embedded_wavs", False)),
            review_wav_dir=paths.get("review_wav_dir"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", nargs="?", type=int, default=None)
    parser.add_argument("max_files", nargs="?", type=int, default=None)
    parser.add_argument("--config", default="configs/sed/pipeline.yaml")
    args = parser.parse_args()
    run_pipeline(args.config, args.epochs, args.max_files)
