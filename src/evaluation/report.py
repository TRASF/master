"""Compatibility wrapper for canonical evaluation reporting."""

from wingbeat_ml.evaluation.report import (
    log_class_support_tables,
    log_confusion_matrices,
    log_prediction_table,
    log_test_report_metrics,
    report_results,
)

__all__ = [
    "log_class_support_tables",
    "log_confusion_matrices",
    "log_prediction_table",
    "log_test_report_metrics",
    "report_results",
]
