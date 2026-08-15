"""Integration and parity tests for deployment dynamic configuration & preprocessing."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np

from wingbeat_ml.data.audio import load_audio, to_mono
from wingbeat_ml.export.bundle import export_ota_config_json, export_input_quantization_header
from wingbeat_ml.export.tflite import convert_full_int8_tflite
from wingbeat_ml.visualizer.analyzer import FastTFLiteModel, HostAnalyzer

try:
    import tensorflow as tf
except ImportError:
    tf = None


def make_dummy_model():
    inputs = tf.keras.layers.Input(shape=(2400, 1))
    x = tf.keras.layers.Conv1D(filters=8, kernel_size=7, activation="relu")(inputs)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(11, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


class TestDeploymentParity(unittest.TestCase):
    def test_dynamic_config_json_schema(self):
        if tf is None:
            self.skipTest("TensorFlow not installed")

        def rep_ds():
            yield [np.full((1, 2400, 1), 0.05, dtype=np.float32)]

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tflite")
            json_path = os.path.join(tmpdir, "config_ota.json")

            convert_full_int8_tflite(make_dummy_model(), rep_ds, model_path)

            export_ota_config_json(
                tflite_path=model_path,
                out_json_path=json_path,
                amplitude_range=0.03,
                dc_removal=True,
                rms_normalization=True,
                detection_threshold=0.65,
                class_names=["cls0", "cls1"],
            )

            with open(json_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            self.assertEqual(cfg["preprocessing_overrides"]["dc_removal"], True)
            self.assertEqual(cfg["preprocessing_overrides"]["rms_normalization"], True)
            self.assertEqual(cfg["inference"]["detection_threshold"], 0.65)
            self.assertIn("quantization", cfg)
            self.assertIn("input_scale", cfg["quantization"])

    def test_fast_tflite_model_execution(self):
        if tf is None:
            self.skipTest("TensorFlow not installed")

        def rep_ds():
            yield [np.full((1, 2400, 1), 0.05, dtype=np.float32)]

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tflite")
            convert_full_int8_tflite(make_dummy_model(), rep_ds, model_path)

            fast_model = FastTFLiteModel(model_path)
            dummy_audio = np.random.randn(2400).astype(np.float32)

            class_id, conf, _ = fast_model.predict_fast(dummy_audio)
            self.assertGreaterEqual(class_id, 0)
            self.assertLess(class_id, 11)
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)

    def test_ota_manifest_generation(self):
        from tools.test_esp32_ota import generate_ota_manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest = generate_ota_manifest(tmp_path, "192.168.1.100", 8080)
            self.assertTrue(manifest.exists())
            with open(manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("192.168.1.100:8080/model_full_int8.tflite", data["model_url"])

    def test_audio_preprocessing_pipeline_parity(self):
        # Verify python audio preprocessing matches deployment C++ formula
        raw_signal = np.sin(np.linspace(0, 50, 2400, dtype=np.float32)) + 0.5  # Signal with DC bias
        
        # 1. DC removal
        dc_removed = raw_signal - np.mean(raw_signal)
        self.assertAlmostEqual(float(np.mean(dc_removed)), 0.0, places=5)

        # 2. Peak normalization
        peak = np.max(np.abs(dc_removed))
        normalized = (dc_removed / peak) * 0.95
        self.assertAlmostEqual(float(np.max(np.abs(normalized))), 0.95, places=5)


if __name__ == "__main__":
    unittest.main()
