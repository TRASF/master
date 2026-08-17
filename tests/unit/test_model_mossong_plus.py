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

    def test_empty_layers_config_builds_linear_classifier(self):
        empty_cfg = copy.deepcopy(self.config)
        empty_cfg["model"]["mossong_plus"]["layers"] = []
        model = self.build(empty_cfg)
        self.assertEqual(model.output_shape, (None, 11))
        x = np.random.normal(size=(2, 2400, 1)).astype(np.float32)
        out = model.predict(x, verbose=0)
        self.assertEqual(out.shape, (2, 11))

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

    def test_conv1d_relu_flatten_dense_builds_via_registry(self):
        cfg = {
            "model": {
                "mossong_plus": {
                    "layers": [
                        {"type": "conv1d", "filters": 8, "kernel_size": 63, "strides": 4, "padding": "valid"},
                        {"type": "relu"},
                        {"type": "flatten"},
                        {"type": "dense", "units": 11},
                    ]
                }
            }
        }
        model = MosSongPlusModel(cfg).build(input_shape=(2400, 1), output_units=11)
        self.assertEqual(len(model.layers), 6)
        self.assertEqual(model.output_shape, (None, 11))

    def test_conv1d_relu_gap_dense_builds(self):
        cfg = {
            "model": {
                "mossong_plus": {
                    "layers": [
                        {"type": "conv1d", "filters": 8, "kernel_size": 63, "strides": 4, "padding": "valid"},
                        {"type": "relu"},
                        {"type": "global_avg_pool"},
                        {"type": "dense", "units": 11},
                    ]
                }
            }
        }
        model = MosSongPlusModel(cfg).build(input_shape=(2400, 1), output_units=11)
        self.assertEqual(len(model.layers), 6)
        self.assertEqual(model.output_shape, (None, 11))

    def test_mossong_plus_yaml_builds(self):
        model = self.build()
        self.assertGreater(len(model.layers), 5)
        self.assertEqual(model.output_shape, (None, 11))

    def test_relu_registration_is_used(self):
        from wingbeat_ml.models.registry import LAYER_REGISTRY
        self.assertTrue(LAYER_REGISTRY.contains("relu"))
        cfg = {
            "model": {
                "mossong_plus": {
                    "layers": [
                        {"type": "conv1d", "filters": 8, "kernel_size": 63, "padding": "valid"},
                        {"type": "relu"},
                        {"type": "flatten"},
                    ]
                }
            }
        }
        model = MosSongPlusModel(cfg).build(input_shape=(2400, 1), output_units=11)
        layer_classes = [l.__class__.__name__ for l in model.layers]
        self.assertIn("ReLU", layer_classes)

    def test_missing_layer_produces_registry_error(self):
        from wingbeat_ml.models.registry import LAYER_REGISTRY
        old = LAYER_REGISTRY._entries.pop("flatten", None)
        try:
            cfg = {
                "model": {
                    "mossong_plus": {
                        "layers": [
                            {"type": "flatten"},
                        ]
                    }
                }
            }
            with self.assertRaises(KeyError):
                MosSongPlusModel(cfg).build(input_shape=(2400, 1), output_units=11)
        finally:
            if old:
                LAYER_REGISTRY._entries["flatten"] = old

    def test_unknown_layer_fails_config_validation(self):
        from wingbeat_ml.config.schema import parse_layer_config
        with self.assertRaises(Exception):
            parse_layer_config({"type": "nonexistent_layer_xyz"})

    def test_invalid_kernel_size_fails_config_validation(self):
        from wingbeat_ml.config.schema import parse_layer_config
        with self.assertRaises(Exception):
            parse_layer_config({"type": "conv1d", "filters": 16, "kernel_size": -5})

    def test_sincconv1d_builds_via_registry(self):
        cfg = {
            "model": {
                "mossong_plus": {
                    "layers": [
                        {"type": "sincconv1d", "filters": 16, "kernel_size": 101, "sample_rate": 8000, "min_low_hz": 50},
                        {"type": "global_avg_pool"},
                    ]
                }
            }
        }
        model = MosSongPlusModel(cfg).build(input_shape=(2400, 1), output_units=11)
        self.assertEqual(model.output_shape, (None, 11))

    def test_repconv1d_builds_via_registry(self):
        cfg = {
            "model": {
                "mossong_plus": {
                    "layers": [
                        {"type": "repconv1d", "filters": 16, "kernel_size": 7, "strides": 2, "branches": 2},
                        {"type": "global_avg_pool"},
                    ]
                }
            }
        }
        model = MosSongPlusModel(cfg).build(input_shape=(2400, 1), output_units=11)
        self.assertEqual(model.output_shape, (None, 11))

    def test_misspelled_layer_field_fails_validation(self):
        from pydantic import ValidationError
        cfg = {
            "model": {
                "mossong_plus": {
                    "layers": [
                        {"type": "conv1d", "filter": 32, "kernel_size": 63},
                    ]
                }
            }
        }
        with self.assertRaises(ValidationError):
            MosSongPlusModel(cfg).build(input_shape=(2400, 1), output_units=11)

    def test_numerical_output_shape_remains_unchanged(self):
        model = self.build()
        inputs = np.ones((5, 2400, 1), dtype=np.float32)
        preds = model(inputs, training=False).numpy()
        self.assertEqual(preds.shape, (5, 11))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_concat_group_block_builds_via_registry(self):
        cfg = {
            "model": {
                "mossong_plus": {
                    "layers": [
                        {"type": "conv1d", "filters": 16, "kernel_size": 32, "strides": 2},
                        {
                            "type": "concat",
                            "layers": [
                                {"type": "global_avg_pool1d"},
                                {"type": "global_max_pool1d"},
                            ],
                        },
                        {"type": "dense", "units": 64},
                    ]
                }
            }
        }
        model = MosSongPlusModel(cfg).build(input_shape=(2400, 1), output_units=11)
        self.assertEqual(model.output_shape, (None, 11))
        x = np.ones((2, 2400, 1), dtype=np.float32)
        out = model(x, training=False).numpy()
        self.assertEqual(out.shape, (2, 11))

    def test_group_block_with_sequential_branches(self):
        cfg = {
            "model": {
                "mossong_plus": {
                    "layers": [
                        {"type": "conv1d", "filters": 16, "kernel_size": 32, "strides": 2},
                        {
                            "type": "group",
                            "layers": [
                                [{"type": "conv1d", "filters": 8, "kernel_size": 3, "padding": "same"}, {"type": "global_avg_pool1d"}],
                                {"type": "global_max_pool1d"},
                            ],
                        },
                    ]
                }
            }
        }
        model = MosSongPlusModel(cfg).build(input_shape=(2400, 1), output_units=11)
        self.assertEqual(model.output_shape, (None, 11))



if __name__ == "__main__":
    unittest.main()
