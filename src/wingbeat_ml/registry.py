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


__all__ = ["available_models", "get_model_builder"]
