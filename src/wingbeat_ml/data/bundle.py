"""Dataset bundle passed between the pipeline and training strategies."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetBundle:
    train: object
    validation: object
    test: object

    train_steps: int = 0
    validation_steps: int = 0
    test_steps: int = 0

    # ponytail: dict keeps algorithm names out of this type; strategies use
    # extra["target"] etc. rather than named fields like adabn_target.
    extra: dict = field(default_factory=dict)


__all__ = ["DatasetBundle"]
