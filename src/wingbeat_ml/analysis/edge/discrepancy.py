"""Edge analysis module: Host-vs-MCU numerical discrepancy and accuracy analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import numpy as np


@dataclass(frozen=True)
class DiscrepancyResult:
    argmax_agreement: bool
    host_class_id: int
    mcu_class_id: int
    max_absolute_error: float
    mean_absolute_error: float
    confidence_difference: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "argmax_agreement": self.argmax_agreement,
            "host_class_id": self.host_class_id,
            "mcu_class_id": self.mcu_class_id,
            "max_absolute_error": self.max_absolute_error,
            "mean_absolute_error": self.mean_absolute_error,
            "confidence_difference": self.confidence_difference,
        }


def calculate_prediction_discrepancy(
    host_probs: np.ndarray,
    mcu_probs: np.ndarray,
) -> DiscrepancyResult:
    """Calculate numerical discrepancy between host and MCU model predictions."""
    h_p = np.squeeze(np.asarray(host_probs, dtype=np.float32))
    m_p = np.squeeze(np.asarray(mcu_probs, dtype=np.float32))

    host_cls = int(np.argmax(h_p))
    mcu_cls = int(np.argmax(m_p))
    agreement = bool(host_cls == mcu_cls)

    abs_err = np.abs(h_p - m_p)
    max_err = float(np.max(abs_err)) if abs_err.size > 0 else 0.0
    mean_err = float(np.mean(abs_err)) if abs_err.size > 0 else 0.0
    conf_diff = float(h_p[host_cls] - m_p[mcu_cls])

    return DiscrepancyResult(
        argmax_agreement=agreement,
        host_class_id=host_cls,
        mcu_class_id=mcu_cls,
        max_absolute_error=max_err,
        mean_absolute_error=mean_err,
        confidence_difference=conf_diff,
    )


__all__ = [
    "DiscrepancyResult",
    "calculate_prediction_discrepancy",
]
