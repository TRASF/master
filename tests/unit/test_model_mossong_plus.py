"""MosSongPlus architecture and checkpoint compatibility tests."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow.keras as keras
import yaml

from wingbeat_ml.models import MosSongPlusModel


class TestMosSongPlusModel(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(
            Path("configs/models/mossong_plus.yaml").read_text(
                encoding="utf-8"
            )
        )

    def build(self, config=None):
        keras.backend.clear_session()
        keras.utils.set_random_seed(45)
        return MosSongPlusModel(config or self.config).build(
            input_shape=(2400, 1),
            output_units=11,
            output_activation="softmax",
        )

    def test_canonical_model_builds_expected_output(self):
        model = self.build()
        self.assertEqual(model.name, "MosquitoSongPlus")
        self.assertEqual(model.output_shape, (None, 11))

    def test_legacy_configuration_key_remains_supported(self):
        legacy = copy.deepcopy(self.config)
        model_config = legacy["model"]
        model_config["mossongplus"] = model_config.pop("mossong_plus")

        model = self.build(legacy)
        self.assertEqual(model.output_shape, (None, 11))

    def test_legacy_import_is_the_canonical_builder(self):
        from model.mossongplus import MosSongPlusModel as LegacyBuilder

        self.assertIs(LegacyBuilder, MosSongPlusModel)

    def test_weights_round_trip_without_prediction_changes(self):
        source = self.build()
        inputs = np.ones((2, 2400, 1), dtype=np.float32)
        expected = source(inputs, training=False).numpy()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.weights.h5"
            source.save_weights(checkpoint)

            restored = self.build()
            restored.load_weights(checkpoint)
            actual = restored(inputs, training=False).numpy()

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-6,
            atol=1e-7,
        )


    def test_leaky_relu_layer_support(self):
        cfg = {
            "model": {
                "mossong_plus": {
                    "layers": [
                        {"type": "conv1d", "filters": 16, "kernel_size": 10, "activation": None},
                        {"type": "leaky_relu", "negative_slope": 0.1},
                        {"type": "flatten"},
                    ]
                }
            }
        }
        model = self.build(cfg)
        self.assertIn("leaky_re_lu", [l.name for l in model.layers])
        self.assertEqual(model.layers[2].negative_slope, 0.1)

    def test_relu6_layer_and_activation_support(self):
        cfg = {
            "model": {
                "mossong_plus": {
                    "layers": [
                        {"type": "conv1d", "filters": 16, "kernel_size": 10, "activation": "relu6"},
                        {"type": "conv1d", "filters": 16, "kernel_size": 10, "activation": None},
                        {"type": "relu", "max_value": 6},
                        {"type": "flatten"},
                    ]
                }
            }
        }
        model = self.build(cfg)
        self.assertEqual(model.output_shape, (None, 11))


if __name__ == "__main__":
    unittest.main()
