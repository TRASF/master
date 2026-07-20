"""Tests for the centralized training components."""

import unittest

import numpy as np
import tensorflow as tf

from wingbeat_ml.models import MosSongPlusModel
from wingbeat_ml.pipelines.train import (
    build_training_components,
    configure_trainable_layers,
    run_training,
)
from wingbeat_ml.registry import available_models, get_model_builder
from wingbeat_ml.training import Trainer


def make_model():
    inputs = tf.keras.layers.Input(shape=(2,))
    hidden = tf.keras.layers.Dense(4, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(hidden)
    return tf.keras.Model(inputs, outputs)


def make_dataset():
    x = np.ones((4, 2), dtype=np.float32)
    y = tf.one_hot([0, 1, 0, 1], depth=2)
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(2)


def make_config(mode="pretrain"):
    return {
        "training_mode": mode,
        "train": {"epochs": 1},
        "model": {"output_activation": "softmax"},
        "optimizer": {
            "name": "SGD",
            "learning_rate": 0.01,
        },
        "loss": {
            "name": "CategoricalCrossentropy",
            "from_logits": False,
        },
        "callbacks": {},
    }


class TestRegistry(unittest.TestCase):
    def test_canonical_and_legacy_model_ids(self):
        self.assertEqual(available_models(), ("mossong_plus",))
        self.assertIs(
            get_model_builder("mossong_plus"),
            MosSongPlusModel,
        )
        self.assertIs(
            get_model_builder("mossongplus"),
            MosSongPlusModel,
        )

    def test_unknown_model_fails_clearly(self):
        with self.assertRaises(ValueError):
            get_model_builder("unknown")


class TestTrainingModes(unittest.TestCase):
    def test_linear_probe_freezes_everything_except_head(self):
        model = make_model()
        configure_trainable_layers(model, "linear_probe")

        self.assertFalse(model.layers[0].trainable)
        self.assertTrue(model.layers[-1].trainable)

    def test_fine_tune_unfreezes_all_layers(self):
        model = make_model()
        for layer in model.layers:
            layer.trainable = False

        configure_trainable_layers(model, "fine_tune")
        self.assertTrue(all(layer.trainable for layer in model.layers))


class TestTrainingPipeline(unittest.TestCase):
    def test_builds_canonical_trainer(self):
        trainer, _, _, _, mode = build_training_components(
            make_model(),
            make_dataset(),
            make_config(),
        )

        self.assertIsInstance(trainer, Trainer)
        self.assertEqual(mode, "pretrain")

    def test_runs_one_epoch_and_returns_history(self):
        model = make_model()
        history = run_training(
            model,
            make_dataset(),
            make_config(),
            evaluate_epoch=lambda: {
                "loss": 0.5,
                "accuracy": 0.75,
            },
        )

        self.assertEqual(len(history), 1)
        self.assertIn("train_loss", history[0])
        self.assertEqual(history[0]["val_loss"], 0.5)
        self.assertEqual(history[0]["val_accuracy"], 0.75)

    def test_legacy_trainer_import_is_preserved(self):
        from src.framework.supervised.train_step import Train

        self.assertIs(Train, Trainer)


if __name__ == "__main__":
    unittest.main()
