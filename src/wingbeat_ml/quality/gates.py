"""Deterministic metric quality gates."""

from numbers import Real


def evaluate_quality_gates(metrics, minimums):
    """Compare numeric metrics with required minimum values."""
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a mapping")
    if not isinstance(minimums, dict) or not minimums:
        raise ValueError(
            "at least one quality-gate minimum is required"
        )

    checks = []

    for metric in sorted(minimums):
        minimum = minimums[metric]
        actual = metrics.get(metric)

        if isinstance(minimum, bool) or not isinstance(minimum, Real):
            raise TypeError(
                f"minimum for {metric!r} must be numeric"
            )

        if isinstance(actual, bool) or not isinstance(actual, Real):
            actual_value = None
            passed = False
            reason = "missing or non-numeric metric"
        else:
            actual_value = float(actual)
            passed = actual_value >= float(minimum)
            reason = None if passed else "below minimum"

        checks.append({
            "metric": metric,
            "actual": actual_value,
            "minimum": float(minimum),
            "passed": passed,
            "reason": reason,
        })

    failed = [
        check["metric"]
        for check in checks
        if not check["passed"]
    ]

    return {
        "passed": not failed,
        "failed": failed,
        "checks": checks,
    }


__all__ = ["evaluate_quality_gates"]
