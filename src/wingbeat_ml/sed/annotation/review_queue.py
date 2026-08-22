"""Partition uncalibrated detections into candidate review queues."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def export_review_queues(
    detections_csv: str | Path,
    output_dir: str | Path,
    high_threshold: float = 0.95,
    low_threshold: float = 0.60,
    sample_audit_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition candidates without granting AUTO_ACCEPT status."""
    det_p = Path(detections_csv)
    if not det_p.exists():
        raise FileNotFoundError(f"Detections file not found: {det_p}")

    df = pd.read_csv(det_p)
    if "confidence" not in df.columns:
        raise KeyError("Detections CSV must contain a 'confidence' column.")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    high_confidence_df = df[df["confidence"] >= high_threshold].copy()
    review_queue_df = df[(df["confidence"] >= low_threshold) & (df["confidence"] < high_threshold)].copy()
    high_confidence_df["status"] = "CANDIDATE"
    review_queue_df["status"] = "REVIEW"
    audit_sample_df = high_confidence_df.sample(frac=sample_audit_ratio, random_state=seed) if not high_confidence_df.empty else pd.DataFrame()

    high_confidence_df.to_csv(out_p / "high_confidence_candidates.csv", index=False)
    review_queue_df.to_csv(out_p / "human_review_queue.csv", index=False)
    audit_sample_df.to_csv(out_p / "high_conf_precision_audit.csv", index=False)

    print(f"Exported {len(high_confidence_df)} high-confidence candidates (not auto-accepted).")
    print(f"Exported {len(review_queue_df)} candidates to human review queue.")
    print(f"Exported {len(audit_sample_df)} high-confidence audit samples.")

    return high_confidence_df, review_queue_df, audit_sample_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--output", default="MosSongPlus/output/review_queues")
    args = parser.parse_args()

    export_review_queues(args.detections, args.output)
