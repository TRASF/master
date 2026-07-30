"""High-level Wingbeat ML pipelines."""

_TRAIN_EXPORTS = {
    "build_training_components",
    "configure_trainable_layers",
    "run_training",
}

_SSL_PYTORCH_EXPORTS = {
    "run_ssl_pipeline",
    "train_fixmatch",
    "train_flexmatch",
}

_SSL_TF_EXPORTS = {
    "run_tf_ssl_pipeline",
}


def __getattr__(name: str):
    if name in _TRAIN_EXPORTS:
        from wingbeat_ml.pipelines import train
        return getattr(train, name)
    if name in _SSL_PYTORCH_EXPORTS:
        from wingbeat_ml.pipelines import ssl
        return getattr(ssl, name)
    if name in _SSL_TF_EXPORTS:
        from wingbeat_ml.pipelines import ssl_tf
        return getattr(ssl_tf, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def get_training_entrypoint(mode: str):
    """Return canonical entrypoint for selectable training mode."""
    normalized = mode.strip().casefold().replace("-", "_")
    normalized = {
        "finetune": "fine_tune",
        "linearprobe": "linear_probe",
        "ssl_tf": "tf_ssl",
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
    if normalized in ("tf_ssl", "ssl_tf"):
        from wingbeat_ml.pipelines.ssl_tf import run_tf_ssl_pipeline
        return run_tf_ssl_pipeline
    if normalized == "ssl":
        from wingbeat_ml.pipelines.ssl import run_ssl_pipeline
        return run_ssl_pipeline

    raise ValueError(
        f"Unsupported training mode {mode!r}; expected pretrain, "
        "linear_probe, fine_tune, ssl_tf, or ssl"
    )


__all__ = [
    "build_training_components",
    "configure_trainable_layers",
    "get_training_entrypoint",
    "run_training",
    "run_ssl_pipeline",
    "train_fixmatch",
    "train_flexmatch",
    "run_tf_ssl_pipeline",
]
