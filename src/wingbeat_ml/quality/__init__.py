"""Model quality-gate components."""

from wingbeat_ml.quality.gates import evaluate_quality_gates
from wingbeat_ml.quality.report import write_quality_report

__all__ = [
    "evaluate_quality_gates",
    "write_quality_report",
]
