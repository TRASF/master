"""Explicit strategy registry. Add entries here when implementing new methods."""

from wingbeat_ml.training.strategies.supervised import SupervisedStrategy

STRATEGIES = {
    "supervised": SupervisedStrategy,
}


def build_strategy(name, **kwargs):
    try:
        strategy_class = STRATEGIES[name]
    except KeyError as error:
        available = ", ".join(sorted(STRATEGIES))
        raise ValueError(
            f"Unknown training strategy {name!r}. "
            f"Available: {available}"
        ) from error
    return strategy_class(**kwargs)


__all__ = ["STRATEGIES", "build_strategy"]
