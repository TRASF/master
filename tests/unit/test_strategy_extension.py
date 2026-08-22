"""Proves the extension boundary: a new strategy needs one file + one registry entry."""

import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from wingbeat_ml.classification.training.strategies.base import TrainingStrategy
from wingbeat_ml.classification.training.strategies import registry as _registry_module
from wingbeat_ml.classification.training.strategies.registry import build_strategy, STRATEGIES


class DummyStrategy(TrainingStrategy):
    required_datasets = {"train"}

    def __init__(self, model, optimizer, loss_fn, config, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self._step_counter = tf.Variable(0, dtype=tf.int32, trainable=False)

    def train_epoch(self, datasets=None, *, epoch):
        return {"loss": 0.1, "custom_metric": 42.0}

    def validate_epoch(self, dataset=None, *, epoch):
        return {"loss": 0.2}

    def checkpoint_objects(self):
        return {"step_counter": self._step_counter}

    def finalize(self, context=None, datasets=None):
        self._step_counter.assign(999)


def _make_model():
    inputs = tf.keras.layers.Input(shape=(2,))
    outputs = tf.keras.layers.Dense(2)(inputs)
    return tf.keras.Model(inputs, outputs)


class TestStrategyExtensionBoundary(unittest.TestCase):
    """One new strategy file + one registry entry = full integration."""

    def test_dummy_registers_and_builds(self):
        patched = {**STRATEGIES, "dummy": DummyStrategy}
        with mock.patch.object(_registry_module, "STRATEGIES", patched):
            strategy = build_strategy(
                "dummy",
                model=_make_model(),
                optimizer=tf.keras.optimizers.SGD(0.01),
                loss_fn=tf.keras.losses.CategoricalCrossentropy(),
                config={},
            )
        self.assertIsInstance(strategy, DummyStrategy)

    def test_unknown_strategy_raises_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            build_strategy("nonexistent", model=None, optimizer=None, loss_fn=None, config={})
        self.assertIn("nonexistent", str(ctx.exception))
        self.assertIn("supervised", str(ctx.exception))

    def test_strategy_receives_model_optimizer_loss_config(self):
        model = _make_model()
        optimizer = tf.keras.optimizers.SGD(0.01)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        config = {"key": "value"}

        patched = {**STRATEGIES, "dummy": DummyStrategy}
        with mock.patch.object(_registry_module, "STRATEGIES", patched):
            strategy = build_strategy(
                "dummy",
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                config=config,
            )

        self.assertIs(strategy.model, model)
        self.assertIs(strategy.optimizer, optimizer)
        self.assertIs(strategy.loss_fn, loss_fn)
        self.assertEqual(strategy.config["key"], "value")

    def test_train_epoch_returns_custom_metrics(self):
        patched = {**STRATEGIES, "dummy": DummyStrategy}
        with mock.patch.object(_registry_module, "STRATEGIES", patched):
            strategy = build_strategy(
                "dummy",
                model=_make_model(),
                optimizer=tf.keras.optimizers.SGD(0.01),
                loss_fn=tf.keras.losses.CategoricalCrossentropy(),
                config={},
            )

        logs = strategy.train_epoch(None, epoch=0)
        self.assertIn("custom_metric", logs)
        self.assertEqual(logs["custom_metric"], 42.0)

    def test_checkpoint_objects_exposes_extra_state(self):
        patched = {**STRATEGIES, "dummy": DummyStrategy}
        with mock.patch.object(_registry_module, "STRATEGIES", patched):
            strategy = build_strategy(
                "dummy",
                model=_make_model(),
                optimizer=tf.keras.optimizers.SGD(0.01),
                loss_fn=tf.keras.losses.CategoricalCrossentropy(),
                config={},
            )

        objects = strategy.checkpoint_objects()
        self.assertIn("step_counter", objects)

    def test_finalize_runs_cleanly(self):
        patched = {**STRATEGIES, "dummy": DummyStrategy}
        with mock.patch.object(_registry_module, "STRATEGIES", patched):
            strategy = build_strategy(
                "dummy",
                model=_make_model(),
                optimizer=tf.keras.optimizers.SGD(0.01),
                loss_fn=tf.keras.losses.CategoricalCrossentropy(),
                config={},
            )

        strategy.finalize()
        self.assertEqual(int(strategy._step_counter.numpy()), 999)

    def test_base_validate_epoch_not_implemented(self):
        base = TrainingStrategy()
        with self.assertRaises(NotImplementedError):
            base.train_epoch(None, epoch=0)

    def test_pipelines_train_not_modified(self):
        """Verifies train.py has no DummyStrategy or research-algorithm mentions."""
        import inspect
        from wingbeat_ml.pipelines import train
        source = inspect.getsource(train)
        for name in ("DummyStrategy", "adabn", "pseudo_label", "mean_teacher", "fixmatch"):
            self.assertNotIn(name, source)


if __name__ == "__main__":
    unittest.main()
