"""Small integration helpers for the existing real-data explain pipeline.

This module intentionally does NOT load datasets.  Keep using the same real
evaluation/test sample collector that already produced explanation_manifest.csv.

Replace the old `_extract_cam_array(...)` path with `explain_sample(...)`.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from wingbeat_ml.analysis.model.signal_gradcam import (
    CamExplanation,
    GradCamConfig,
    SignalGradCamAnalyzer,
)
from wingbeat_ml.visualizer.signal_gradcam import plot_signal_gradcam


def explain_sample(
    *,
    model: Any,
    class_names: Sequence[str],
    signal: np.ndarray,
    true_class_id: int,
    target_layer: str,
    explainer: str,
    output_plot: str,
    sampling_rate: int = 8000,
    target_class_id: int | None = None,
) -> tuple[CamExplanation, dict[str, Any], dict[str, Any]]:
    """Explain one REAL sample and return manifest + diagnostic rows."""
    analyzer = SignalGradCamAnalyzer(
        model=model,
        class_names=class_names,
        config=GradCamConfig(
            target_layer=target_layer,
            explainer=explainer,
            sampling_rate=sampling_rate,
        ),
    )

    explanation = analyzer.explain_one(
        signal,
        true_class_id=int(true_class_id),
        target_class_id=target_class_id,
    )

    pred = explanation.prediction

    # Save plot only after raw CAM validation succeeded.
    Path(output_plot).parent.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_signal_gradcam(
        signal=signal,
        cam_heatmap=explanation.cam,
        sr=sampling_rate,
        title=(
            f"{explainer} | {target_layer} | "
            f"true={class_names[int(true_class_id)]} | "
            f"pred={pred.predicted_class_name} "
            f"(p={pred.confidence:.3f})"
        ),
        save_path=output_plot,
        diagnostics=explanation.diagnostics,
    )

    # Avoid retaining figures in large batch jobs.
    import matplotlib.pyplot as plt
    plt.close(fig)

    manifest_row = {
        "class_name": class_names[int(true_class_id)],
        "true_class_id": int(true_class_id),
        "predicted_class_name": pred.predicted_class_name,
        "predicted_class_id": pred.predicted_class_id,
        "correct": pred.predicted_class_id == int(true_class_id),

        # IMPORTANT: score/logit and probability are separate fields.
        "predicted_score": pred.predicted_score,
        "confidence": pred.confidence,
        "model_output_is_probability_distribution": (
            pred.output_is_probability_distribution
        ),

        "target_class_name": class_names[
            pred.predicted_class_id
            if target_class_id is None
            else int(target_class_id)
        ],
        "target_class_id": (
            pred.predicted_class_id
            if target_class_id is None
            else int(target_class_id)
        ),
        "target_layer": target_layer,
        "explainer": explainer,
        "output_plot": output_plot,
        "selected_cam_key": repr(explanation.selected_key),
    }

    diagnostic_row = {
        "target_layer": target_layer,
        "explainer": explainer,
        **explanation.diagnostics.to_dict(),
        **{
            f"gradient_{k}": v
            for k, v in explanation.gradient_diagnostics.to_dict().items()
            if k != "layer_name"
        },
    }

    return explanation, manifest_row, diagnostic_row


__all__ = ["explain_sample"]
