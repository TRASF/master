"""Quality validation pipeline for model metrics."""

import json
from pathlib import Path

from wingbeat_ml.quality import (
    evaluate_quality_gates,
    write_quality_report,
)


def load_metrics(path):
    """Load a flat or nested JSON metric mapping."""
    with Path(path).open(encoding="utf-8") as stream:
        payload = json.load(stream)

    if not isinstance(payload, dict):
        raise ValueError("metrics JSON must contain an object")

    metrics = payload.get("metrics", payload)
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a JSON object")

    return metrics


def parse_minimums(expressions):
    """Parse repeated metric=value expressions."""
    minimums = {}

    for expression in expressions or ():
        if "=" not in expression:
            raise ValueError(
                f"invalid minimum {expression!r}; expected metric=value"
            )

        metric, raw_value = expression.split("=", 1)
        metric = metric.strip()

        if not metric:
            raise ValueError(
                "quality-gate metric name cannot be empty"
            )

        try:
            minimums[metric] = float(raw_value)
        except ValueError as error:
            raise ValueError(
                f"minimum for {metric!r} must be numeric"
            ) from error

    if not minimums:
        raise ValueError("at least one --minimum is required")

    return minimums


def validate_metrics(metrics, minimums, *, output_path=None):
    """Run quality gates and optionally persist their report."""
    report = evaluate_quality_gates(metrics, minimums)

    if output_path:
        write_quality_report(report, output_path)

    return report


__all__ = [
    "load_metrics",
    "parse_minimums",
    "validate_metrics",
]
