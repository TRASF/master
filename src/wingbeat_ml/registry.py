"""Explicit component registry for selectable Wingbeat ML models and components."""

from typing import Generic, TypeVar, Dict, Optional, Tuple, Any

T = TypeVar("T")


class Registry(Generic[T]):
    """Generic reusable component registry."""

    def __init__(self, name: str):
        self.name = name
        self._entries: Dict[str, T] = {}

    def register(self, name: str, item: Optional[T] = None):
        def decorator(obj: T) -> T:
            key = name.lower().replace("-", "_")
            self._entries[key] = obj
            return obj

        if item is not None:
            decorator(item)
            return item
        return decorator

    def get(self, name: str) -> T:
        key = name.lower().replace("-", "_")
        if key not in self._entries:
            raise KeyError(
                f"Item {name!r} not found in registry {self.name!r}. "
                f"Available: {sorted(self._entries.keys())}"
            )
        return self._entries[key]

    def contains(self, name: str) -> bool:
        return name.lower().replace("-", "_") in self._entries

    def available(self) -> Tuple[str, ...]:
        return tuple(sorted(self._entries.keys()))


MODEL_BUILDERS: Registry[Any] = Registry("model_builder")
_MODEL_BUILDERS = MODEL_BUILDERS._entries


def available_models() -> tuple[str, ...]:
    """Return canonical model identifiers."""
    return MODEL_BUILDERS.available()


def get_model_builder(model_id: str):
    """Return the builder class registered for *model_id*."""
    normalized = model_id.strip().casefold().replace("-", "_")
    if normalized == "mossongplus":
        normalized = "mossong_plus"

    if not MODEL_BUILDERS.available():
        import wingbeat_ml.classification.models  # noqa: F401

    try:
        return MODEL_BUILDERS.get(normalized)
    except KeyError as error:
        raise ValueError(
            f"Unknown model {model_id!r}; available models: "
            f"{', '.join(available_models())}"
        ) from error


def build_model(config, architecture_config, **build_overrides):
    """Build the model selected by the resolved configuration."""
    from wingbeat_ml.config.schema import validate_config

    app_cfg = validate_config(config)
    builder_class = get_model_builder(app_cfg.model.id)
    builder = builder_class(
        architecture_config,
        model_overrides=app_cfg.model.model_dump(),
    )
    build_options = {
        "input_shape": (app_cfg.audio.segment_length, 1),
        "output_units": app_cfg.num_classes,
        "output_activation": app_cfg.model.output_activation,
    }
    build_options.update(build_overrides)
    return builder.build(**build_options)


__all__ = ["Registry", "available_models", "build_model", "get_model_builder"]

