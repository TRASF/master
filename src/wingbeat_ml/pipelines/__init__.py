"""High-level Wingbeat ML pipelines."""

from wingbeat_ml.pipelines.train import (
    build_training_components,
    configure_trainable_layers,
    run_training,
)


def get_training_entrypoint(mode: str):
    """Return the canonical entrypoint for a selectable training mode."""
    normalized = mode.strip().casefold().replace("-", "_")
    normalized = {
        "finetune": "fine_tune",
        "linearprobe": "linear_probe",
    }.get(normalized, normalized)

    if normalized == "pretrain":
        from wingbeat_ml.pipelines.pretrain import train_supervised
        return train_supervised
    if normalized == "linear_probe":
        from wingbeat_ml.pipelines.linear_probe import train_linear_probe
        return train_linear_probe
    if normalized == "fine_tune":
        from wingbeat_ml.pipelines.fine_tune import train_finetune
        return train_finetune

    raise ValueError(
        f"Unsupported training mode {mode!r}; expected pretrain, "
        "linear_probe, or fine_tune"
    )


__all__ = [
    "build_training_components",
    "configure_trainable_layers",
    "get_training_entrypoint",
    "run_training",
]
