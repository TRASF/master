"""Small user-facing CLI for independent SED workflow stages."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from wingbeat_ml.sed.annotation.review_wav_exporter import export_review_wav
from wingbeat_ml.sed.application.ingest_reviewed import ingest_reviewed_wavs
from wingbeat_ml.sed.application.run_sed_pipeline import load_yaml, run_pipeline
from wingbeat_ml.sed.inference.long_recording import infer_long_recording, load_sed_teacher_model
from wingbeat_ml.sed.training.fine_tune import fine_tune_verified


def detect_file(audio_path: str | Path, config_path: str | Path) -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[4]
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_yaml(config_path)
    paths = config["paths"]
    inference = config["inference"]
    decoder = config["decoder"]
    source = Path(audio_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Audio file not found: {source}")

    checkpoint = root / paths["teacher_output"] / "teacher_v0_best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sed_teacher_model(checkpoint, device=device)
    events = infer_long_recording(
        source,
        model=model,
        chunk_len_s=float(inference["chunk_length_s"]),
        hop_len_s=float(inference["hop_length_s"]),
        frame_rate_hz=float(inference["frame_rate_hz"]),
        batch_size=int(inference.get("batch_size", 32)),
        device=device,
        high_threshold=float(decoder["high_threshold"]),
        low_threshold=float(decoder["low_threshold"]),
        min_duration_s=float(decoder["min_duration_s"]),
        max_merge_gap_s=float(decoder["max_merge_gap_s"]),
    )
    rows = [
        {
            "start_s": event.start_s,
            "end_s": event.end_s,
            "confidence": event.confidence,
            "mean_score": event.mean_score,
            "p90_score": event.p90_score,
            "top25_mean_score": event.top25_mean_score,
            "label": "mosquito",
        }
        for event in events
    ]
    csv_path = source.with_name(f"{source.stem}.mosquito.csv")
    wav_path = source.with_name(f"{source.stem}.mosquito.wav")
    pd.DataFrame(rows, columns=[
        "start_s", "end_s", "confidence", "mean_score", "p90_score",
        "top25_mean_score", "label",
    ]).to_csv(csv_path, index=False)
    export_review_wav(source, wav_path, rows)
    print(f"Detected {len(events)} events")
    print(csv_path)
    print(wav_path)
    return csv_path, wav_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sed.sh")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/sed/pipeline.yaml")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", parents=[common])
    train.add_argument("--epochs", type=int)
    learn = commands.add_parser("learn", parents=[common])
    learn.add_argument("--target", nargs="?")
    learn.add_argument("--batch-size", type=int, default=20)
    commands.add_parser("evaluate", parents=[common])
    label = commands.add_parser("label", parents=[common])
    label.add_argument("target", nargs="?")
    label.add_argument("--max-files", type=int)
    detect = commands.add_parser("detect", parents=[common])
    detect.add_argument("audio")
    ingest = commands.add_parser("ingest-reviewed", parents=[common])
    ingest.add_argument("reviewed_dir", nargs="?")
    ingest.add_argument("--output", default="dataset/verified_sed/v1")
    ingest.add_argument("--exhaustive", action="store_true")
    fine_tune = commands.add_parser("fine-tune")
    fine_tune.add_argument("--dataset", default="dataset/verified_sed/v1")
    fine_tune.add_argument("--base-checkpoint", default="artifacts/teacher_v0/teacher_v0_best.pt")
    fine_tune.add_argument("--output", default="artifacts/teacher_v1")
    fine_tune.add_argument("--epochs", type=int, default=20)
    fine_tune.add_argument("--learning-rate", type=float, default=3e-5)
    fine_tune.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args(argv)
    if args.command == "train":
        run_pipeline(args.config, epochs=args.epochs, stages=("train",))
    elif args.command == "learn":
        from wingbeat_ml.sed.application.run_active_learning import run_active_learning_loop
        run_active_learning_loop(
            config_path=args.config,
            target_dir=args.target,
            review_batch_size=args.batch_size,
        )
    elif args.command == "evaluate":
        run_pipeline(args.config, stages=("evaluate",))
    elif args.command == "label":
        run_pipeline(
            args.config,
            max_files=args.max_files,
            stages=("label",),
            target_dir=args.target,
        )
    elif args.command == "detect":
        detect_file(args.audio, args.config)
    elif args.command == "fine-tune":
        fine_tune_verified(
            dataset_dir=args.dataset,
            base_checkpoint=args.base_checkpoint,
            output_dir=args.output,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )
    else:
        root = Path(__file__).resolve().parents[4]
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = root / config_path
        config = load_yaml(config_path)
        paths = config["paths"]
        generated_review_dir = Path(paths["review_wav_dir"])
        reviewed_dir = args.reviewed_dir or generated_review_dir.with_name("reviewed_wavs")
        ingest_reviewed_wavs(
            reviewed_dir=reviewed_dir,
            candidates_csv=Path(paths["archive_output"]) / "all_candidates.csv",
            output_dir=args.output,
            exhaustive=args.exhaustive,
        )


if __name__ == "__main__":
    main()
