"""Explicit component registry for selectable Wingbeat ML models."""

from wingbeat_ml.models import MosSongPlusModel

_MODEL_BUILDERS = {
    "mossong_plus": MosSongPlusModel,
}


def available_models() -> tuple[str, ...]:
    """Return canonical model identifiers."""
    return tuple(sorted(_MODEL_BUILDERS))


def get_model_builder(model_id: str):
    """Return the builder class registered for *model_id*."""
    normalized = model_id.strip().casefold().replace("-", "_")
    if normalized == "mossongplus":
        normalized = "mossong_plus"

    try:
        return _MODEL_BUILDERS[normalized]
    except KeyError as error:
        raise ValueError(
            f"Unknown model {model_id!r}; available models: "
            f"{', '.join(available_models())}"
        ) from error


def build_model(config, architecture_config, **build_overrides):
    """Build the model selected by the resolved configuration."""
    model_config = config.get("model", {})
    builder_class = get_model_builder(
        str(model_config.get("id", "mossong_plus"))
    )
    builder = builder_class(
        architecture_config,
        model_overrides=model_config,
    )
    build_options = {
        "input_shape": (config["audio"]["segment_length"], 1),
        "output_units": config["num_classes"],
        "output_activation": model_config.get("output_activation"),
    }
    build_options.update(build_overrides)
    return builder.build(**build_options)


__all__ = ["available_models", "build_model", "get_model_builder"]
