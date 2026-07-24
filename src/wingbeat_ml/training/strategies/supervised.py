"""Supervised training strategy: GradientTape loop adapted to the strategy contract."""

from __future__ import annotations

from wingbeat_ml.training.strategies.base import TrainingStrategy


class SupervisedStrategy(TrainingStrategy):
    """Wraps the existing Trainer in the TrainingStrategy interface.

    Two construction paths:
      - Registry path: pass model/optimizer/loss_fn/config (+ optional kwargs);
        the strategy builds a Trainer internally.
      - Internal pipeline path: pass a pre-built Trainer via ``_trainer``; all
        other positional args are ignored.
    """

    required_datasets = {"train", "validation"}

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        config,
        *,
        train_dataset=None,
        class_weights=None,
        evaluate_fn=None,
        profiler_logdir=None,
        _trainer=None,
    ):
        if _trainer is not None:
            self._trainer = _trainer
        else:
            from wingbeat_ml.training.trainer import Trainer

            performance = (config or {}).get("performance", {})
            self._trainer = Trainer(
                model,
                optimizer,
                loss_fn,
                train_dataset,
                class_weights=class_weights,
                steps_per_call=int(performance.get("steps_per_call", 20)),
                jit_compile=bool(performance.get("jit_compile", False)),
                profiler=performance.get("profiler", {}),
                profiler_logdir=profiler_logdir,
            )
        self._evaluate_fn = evaluate_fn

    def train_epoch(self, datasets=None, *, epoch):
        return self._trainer.train_epoch()

    def validate_epoch(self, dataset=None, *, epoch):
        if self._evaluate_fn is not None:
            return self._evaluate_fn()
        return {}

    def checkpoint_objects(self):
        return {}


__all__ = ["SupervisedStrategy"]
