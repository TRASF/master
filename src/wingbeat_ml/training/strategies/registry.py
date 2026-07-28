"""Explicit strategy registry. Add entries here when implementing new strategies."""

from wingbeat_ml.training.strategies.supervised import SupervisedStrategy
from wingbeat_ml.training.strategies.ssl import FixMatchStrategy, FlexMatchStrategy

STRATEGIES = {
    "supervised": SupervisedStrategy,
    "fixmatch": FixMatchStrategy,
    "flexmatch": FlexMatchStrategy,
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
