"""Tests for canonical TFLite export components."""

import importlib
import importlib.util
import inspect
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import tensorflow as tf


def require_module(testcase, name):
    testcase.assertIsNotNone(
        importlib.util.find_spec(name),
        f"missing Phase 9 export module: {name}",
    )
    return importlib.import_module(name)


def make_model():
    inputs = tf.keras.layers.Input(
        batch_shape=(1, 2400, 1),
    )
    pooled = tf.keras.layers.GlobalAveragePooling1D()(inputs)
    outputs = tf.keras.layers.Dense(
        3,
        use_bias=False,
    )(pooled)
    model = tf.keras.Model(inputs, outputs)
    model.layers[-1].set_weights([
        np.array([[1.0, -1.0, 0.5]], dtype=np.float32)
    ])
    return model


def make_dataset():
    positive = np.full((2, 2400, 1), 0.08, dtype=np.float32)
    negative = np.full((2, 2400, 1), -0.08, dtype=np.float32)
    x = np.concatenate([positive, negative])
    y = tf.one_hot([0, 0, 1, 1], depth=3)
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(2)


class TestExportModules(unittest.TestCase):
    def test_canonical_modules_exist(self):
        for name in (
            "wingbeat_ml.export",
            "wingbeat_ml.export.tflite",
            "wingbeat_ml.export.verify",
            "wingbeat_ml.export.bundle",
            "wingbeat_ml.pipelines.export",
        ):
            require_module(self, name)

    def test_legacy_wrapper_exports_canonical_converter(self):
        canonical = require_module(
            self,
            "wingbeat_ml.export.tflite",
        )
        from src.quantization.tf_quantize import (
            convert_full_int8_tflite,
        )

        self.assertIs(
            convert_full_int8_tflite,
            canonical.convert_full_int8_tflite,
        )

    def test_float_and_int8_conversion_and_agreement(self):
        tflite = require_module(
            self,
            "wingbeat_ml.export.tflite",
        )
        verify = require_module(
            self,
            "wingbeat_ml.export.verify",
        )

        model = make_model()
        dataset = make_dataset()
        representative = tflite.make_representative_dataset(
            dataset,
            max_samples=4,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            float_path = tflite.convert_float_tflite(
                model,
                os.path.join(tmpdir, "float.tflite"),
            )
            int8_path = tflite.convert_full_int8_tflite(
                model,
                representative,
                os.path.join(tmpdir, "int8.tflite"),
            )

            details = tflite.inspect_tflite_io(int8_path)
            _, _, keras_scores = verify.predict_keras_dataset(
                model,
                dataset,
            )
            _, _, int8_scores = verify.predict_tflite_dataset(
                int8_path,
                dataset,
            )
            agreement = verify.compare_model_pair_agreement(
                keras_scores,
                int8_scores,
            )

            self.assertGreater(os.path.getsize(float_path), 0)
            self.assertGreater(os.path.getsize(int8_path), 0)
            self.assertEqual(
                details["inputs"][0]["dtype"],
                np.int8,
            )
            self.assertEqual(
                details["outputs"][0]["dtype"],
                np.int8,
            )
            self.assertGreaterEqual(
                agreement["top1_agreement"],
                0.75,
            )

    def test_bundle_generates_aligned_header(self):
        bundle = require_module(
            self,
            "wingbeat_ml.export.bundle",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tflite")
            header_path = os.path.join(tmpdir, "model.h")

            with open(model_path, "wb") as stream:
                stream.write(bytes([1, 2, 3, 255]))

            bundle.export_tflite_to_c_header(
                model_path,
                header_path,
                array_name="g_test_model",
            )

            with open(header_path, encoding="utf-8") as stream:
                content = stream.read()

        self.assertIn("aligned(16)", content)
        self.assertIn("g_test_model_len = 4", content)
        self.assertIn("0xff", content)

    def test_pipeline_requires_real_calibration_by_default(self):
        pipeline = require_module(
            self,
            "wingbeat_ml.pipelines.export",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(RuntimeError):
                pipeline.run_basic_quantization_suite(
                    keras_model=make_model(),
                    val_ds=None,
                    test_ds=None,
                    out_dir=tmpdir,
                )

    def test_export_modules_do_not_require_pandas(self):
        for name in (
            "wingbeat_ml.export.tflite",
            "wingbeat_ml.export.verify",
            "wingbeat_ml.pipelines.export",
        ):
            module = require_module(self, name)
            self.assertNotIn(
                "pandas",
                inspect.getsource(module),
            )

    def test_export_pipeline_uses_pipeline_helpers(self):
        pipeline = require_module(
            self,
            "wingbeat_ml.pipelines.export",
        )
        source = inspect.getsource(pipeline.export_from_weights)

        self.assertIn("build_dataset_bundle(", source)
        self.assertIn("build_model_component(", source)
        self.assertIn("prepare_export_runtime(", source)
        self.assertNotIn("build_datasets(", source)
        self.assertNotIn("build_model(", source)
        self.assertNotIn("configure_training_runtime(", source)
        self.assertNotIn("SupervisedDataset(", source)
        self.assertNotIn("MosSongPlusModel(", source)

    def test_cli_exposes_export_command(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "wingbeat_ml",
                "export",
                "--help",
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--weights", result.stdout)


if __name__ == "__main__":
    unittest.main()
