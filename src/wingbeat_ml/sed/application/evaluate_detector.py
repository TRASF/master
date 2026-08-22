"""Fail-closed, file-aware event evaluation for exhaustive gold benchmarks."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from wingbeat_ml.sed.inference.decoder import DecodedEvent


RECORDING_COLUMNS = {
    "file_name",
    "duration_s",
    "sha256",
    "split",    "is_exhaustively_annotated",
    "verified_clean_negative",
    "annotation_version",
}
EVENT_COLUMNS = {"file_name", "event_id", "start_s", "end_s", "label"}
PREDICTION_COLUMNS = {"file_name", "start_s", "end_s", "confidence"}


@dataclass
class EvaluationMetrics:
    event_precision: float
    event_recall: float
    event_f1: float
    fp_per_hour: float
    fn_per_hour: float
    median_onset_error_s: float
    median_offset_error_s: float
    precision_at_high_conf: float
    recall_at_high_conf: float
    auto_label_yield: float
    true_positives: int
    false_positives: int
    false_negatives: int
    total_audio_duration_s: float

    @property
    def coverage_at_high_conf(self) -> float:
        """Backward-compatible name for auto-label yield."""
        return self.auto_label_yield


def compute_event_iou(e1: DecodedEvent, e2_start: float, e2_end: float) -> float:
    intersection = max(0.0, min(e1.end_s, e2_end) - max(e1.start_s, e2_start))
    union = (e1.end_s - e1.start_s) + (e2_end - e2_start) - intersection
    return intersection / union if union > 0 else 0.0


def _require_columns(table: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _true_series(series: pd.Series, name: str) -> pd.Series:
    values = series.map(lambda value: str(value).strip().lower() if not isinstance(value, bool) else value)
    valid = values.isin([True, False, "true", "false"])
    if not valid.all():
        raise ValueError(f"{name} must contain only true/false values")
    return values.isin([True, "true"])


def validate_gold_tables(
    recordings: pd.DataFrame,
    events: pd.DataFrame,
    audio_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate recording-level completeness and event boundaries; fail closed."""
    _require_columns(recordings, RECORDING_COLUMNS, "Gold recordings table")
    _require_columns(events, EVENT_COLUMNS, "Gold events table")
    if recordings.empty:
        raise ValueError("Gold recordings table is empty")
    if recordings["file_name"].duplicated().any():
        raise ValueError("Gold recordings table contains duplicate file_name values")

    recordings = recordings.copy()
    events = events.copy()
    recordings["duration_s"] = pd.to_numeric(recordings["duration_s"], errors="coerce")
    if recordings["duration_s"].isna().any() or (recordings["duration_s"] <= 0).any():
        raise ValueError("Gold recording durations must be finite and greater than zero")
    if recordings["annotation_version"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Every gold recording requires annotation_version")
    hashes = recordings["sha256"].fillna("").astype(str).str.lower()
    if not hashes.str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Every gold recording requires a valid SHA-256 hash")
    recordings["sha256"] = hashes

    exhaustive = _true_series(recordings["is_exhaustively_annotated"], "is_exhaustively_annotated")
    clean = _true_series(recordings["verified_clean_negative"], "verified_clean_negative")
    recordings["is_exhaustively_annotated"] = exhaustive
    recordings["verified_clean_negative"] = clean
    if not exhaustive.all():
        count = int(exhaustive.sum())
        raise ValueError(
            f"Gold validation incomplete: {count}/{len(recordings)} recordings are exhaustively annotated; "
            "calibration aborted."
        )

    known_files = set(recordings["file_name"])
    unknown_events = set(events["file_name"]) - known_files
    if unknown_events:
        raise ValueError(f"Gold events reference unknown recordings: {sorted(unknown_events)}")
    if not events.empty:
        if events.duplicated(["file_name", "event_id"]).any():
            raise ValueError("Gold event IDs must be unique within each recording")
        events["start_s"] = pd.to_numeric(events["start_s"], errors="coerce")
        events["end_s"] = pd.to_numeric(events["end_s"], errors="coerce")
        if events[["start_s", "end_s"]].isna().any().any():
            raise ValueError("Gold event boundaries must be numeric")
        durations = recordings.set_index("file_name")["duration_s"]
        invalid = (
            (events["start_s"] < 0)
            | (events["end_s"] <= events["start_s"])
            | (events["end_s"] > events["file_name"].map(durations) + 1e-6)
        )
        if invalid.any():
            raise ValueError("Gold events contain invalid or out-of-recording boundaries")

    files_with_events = set(events["file_name"])
    expected_clean = ~recordings["file_name"].isin(files_with_events)
    if not clean.equals(expected_clean):
        raise ValueError("verified_clean_negative must be true exactly for verified recordings with zero events")

    if audio_dir is not None:
        audio_files = {path.name for path in Path(audio_dir).iterdir() if path.is_file()}
        missing_audio = known_files - audio_files
        extra_audio = audio_files - known_files
        if missing_audio or extra_audio:
            raise ValueError(
                f"Gold recording/audio mismatch; missing={sorted(missing_audio)}, extra={sorted(extra_audio)}"
            )
        expected_hashes = recordings.set_index("file_name")["sha256"]
        for path in Path(audio_dir).iterdir():
            if not path.is_file():
                continue
            with path.open("rb") as file:
                actual_hash = hashlib.file_digest(file, "sha256").hexdigest()
            if actual_hash != expected_hashes[path.name]:
                raise ValueError(f"Gold recording hash mismatch: {path.name}")

    return recordings, events


def _iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = end_a - start_a + end_b - start_b - intersection
    return intersection / union if union > 0 else 0.0


def _match_tables(
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    iou_threshold: float,
) -> list[tuple[int, int]]:
    """Maximum-IoU one-to-one assignment, independently within each recording."""
    matches: list[tuple[int, int]] = []
    for file_name in sorted(set(predictions["file_name"]) | set(events["file_name"])):
        pred = predictions[predictions["file_name"] == file_name]
        truth = events[events["file_name"] == file_name]
        if pred.empty or truth.empty:
            continue
        ious = np.array(
            [
                [_iou(p.start_s, p.end_s, g.start_s, g.end_s) for g in truth.itertuples()]
                for p in pred.itertuples()
            ],
            dtype=float,
        )
        pred_positions, truth_positions = linear_sum_assignment(1.0 - ious)
        for pred_position, truth_position in zip(pred_positions, truth_positions):
            if ious[pred_position, truth_position] >= iou_threshold:
                matches.append((pred.index[pred_position], truth.index[truth_position]))
    return matches


def evaluate_tables(
    predictions: pd.DataFrame,
    recordings: pd.DataFrame,
    events: pd.DataFrame,
    iou_threshold: float = 0.2,
    high_conf_threshold: float = 0.9,
    score_column: str = "confidence",
) -> EvaluationMetrics:
    _require_columns(predictions, PREDICTION_COLUMNS, "Predictions table")
    if score_column not in predictions:
        raise ValueError(f"Predictions table missing event score column: {score_column}")
    predictions = predictions.copy()
    if not predictions.empty:
        unknown = set(predictions["file_name"]) - set(recordings["file_name"])
        if unknown:
            raise ValueError(f"Predictions reference unknown gold recordings: {sorted(unknown)}")
        predictions[["start_s", "end_s", score_column]] = predictions[
            ["start_s", "end_s", score_column]
        ].apply(pd.to_numeric, errors="coerce")
        invalid = (
            predictions[["start_s", "end_s", score_column]].isna().any(axis=1)
            | (predictions["start_s"] < 0)
            | (predictions["end_s"] <= predictions["start_s"])
        )
        if invalid.any():
            raise ValueError("Predictions contain invalid event boundaries or scores")

    matches = _match_tables(predictions, events, iou_threshold)
    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(events) - tp
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    duration = float(recordings["duration_s"].sum())
    hours = duration / 3600.0

    onset_errors = [abs(predictions.loc[p, "start_s"] - events.loc[g, "start_s"]) for p, g in matches]
    offset_errors = [abs(predictions.loc[p, "end_s"] - events.loc[g, "end_s"]) for p, g in matches]

    accepted = predictions[predictions[score_column] >= high_conf_threshold]
    accepted_matches = _match_tables(accepted, events, iou_threshold)
    accepted_tp = len(accepted_matches)
    accepted_precision = accepted_tp / len(accepted) if len(accepted) else 1.0
    accepted_recall = accepted_tp / len(events) if len(events) else 1.0

    return EvaluationMetrics(
        event_precision=precision,
        event_recall=recall,
        event_f1=f1,
        fp_per_hour=fp / hours,
        fn_per_hour=fn / hours,
        median_onset_error_s=float(np.median(onset_errors)) if onset_errors else 0.0,
        median_offset_error_s=float(np.median(offset_errors)) if offset_errors else 0.0,
        precision_at_high_conf=accepted_precision,
        recall_at_high_conf=accepted_recall,
        auto_label_yield=len(accepted) / len(predictions) if len(predictions) else 0.0,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        total_audio_duration_s=duration,
    )


def evaluate_detections(
    predicted_events: list[DecodedEvent],
    ground_truth_df: pd.DataFrame,
    total_audio_duration_s: float,
    iou_threshold: float = 0.2,
    high_conf_threshold: float = 0.9,
) -> EvaluationMetrics:
    """Backward-compatible single-recording evaluator."""
    predictions = pd.DataFrame(
        [
            {"file_name": "recording", "start_s": event.start_s, "end_s": event.end_s, "confidence": event.confidence}
            for event in predicted_events
        ],
        columns=sorted(PREDICTION_COLUMNS),
    )
    events = ground_truth_df.copy()
    events["file_name"] = "recording"
    if "event_id" not in events:
        events["event_id"] = [f"e{i}" for i in range(len(events))]
    if "label" not in events:
        events["label"] = "mosquito"
    recordings = pd.DataFrame([{"file_name": "recording", "duration_s": total_audio_duration_s}])
    return evaluate_tables(predictions, recordings, events, iou_threshold, high_conf_threshold)


def calibration_table(
    predictions: pd.DataFrame,
    recordings: pd.DataFrame,
    events: pd.DataFrame,
    iou_threshold: float,
    score_columns: list[str],
    thresholds: list[float],
) -> pd.DataFrame:
    rows = []
    for score_column in score_columns:
        if score_column not in predictions:
            continue
        for threshold in thresholds:
            metrics = evaluate_tables(
                predictions, recordings, events, iou_threshold, threshold, score_column
            )
            rows.append(
                {
                    "event_score_method": score_column,
                    "auto_threshold": threshold,
                    "precision": metrics.precision_at_high_conf,
                    "recall": metrics.recall_at_high_conf,
                    "auto_label_yield": metrics.auto_label_yield,
                    "fp_per_hour": (
                        len(predictions[predictions[score_column] >= threshold])
                        * (1.0 - metrics.precision_at_high_conf)
                        / (metrics.total_audio_duration_s / 3600.0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_evaluation(
    gold_dir: str | Path,
    predictions_csv: str | Path | None = None,
    iou_threshold: float = 0.2,
    high_conf_threshold: float = 0.9,
    score_column: str = "confidence",
) -> EvaluationMetrics:
    gold = Path(gold_dir)
    recording_csv = gold / "recordings.csv"
    event_csv = gold / "events.csv"
    if not recording_csv.is_file() or not event_csv.is_file():
        raise FileNotFoundError("Gold evaluation requires recordings.csv and events.csv")
    if predictions_csv is None or not Path(predictions_csv).is_file():
        raise FileNotFoundError(f"Missing gold predictions CSV: {predictions_csv}")

    recordings, events = validate_gold_tables(
        pd.read_csv(recording_csv), pd.read_csv(event_csv), gold / "audio"
    )
    predictions = pd.read_csv(predictions_csv)
    metrics = evaluate_tables(
        predictions, recordings, events, iou_threshold, high_conf_threshold, score_column
    )

    print("=== Gold completeness ===")
    print(f"{len(recordings)} recordings; {len(events)} events; {int(recordings['verified_clean_negative'].sum())} clean negatives: PASS")
    print("=== Detector evaluation ===")
    for name, value in asdict(metrics).items():
        print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_dir", default="dataset/gold_validation")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.2)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.9)
    parser.add_argument("--score-column", default="confidence")
    arguments = parser.parse_args()
    run_evaluation(
        arguments.gold_dir,
        arguments.predictions,
        arguments.iou_threshold,
        arguments.high_confidence_threshold,
        arguments.score_column,
    )
