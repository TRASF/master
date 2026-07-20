"""Tests for canonical evaluation and quality gates."""

import importlib
import importlib.util
import inspect
import json
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
        f"canonical evaluation module is missing: {name}",
    )
    return importlib.import_module(name)


class TestCanonicalEvaluation(unittest.TestCase):
    def test_modules_exist(self):
        for name in (
            "wingbeat_ml.evaluation",
            "wingbeat_ml.evaluation.evaluator",
            "wingbeat_ml.evaluation.report",
            "wingbeat_ml.pipelines.evaluate",
            "wingbeat_ml.pipelines.validate",
            "wingbeat_ml.quality.gates",
            "wingbeat_ml.quality.report",
        ):
            require_module(self, name)

    def test_small_dataset_can_be_evaluated(self):
        module = require_module(
            self,
            "wingbeat_ml.pipelines.evaluate",
        )

        inputs = tf.keras.layers.Input(shape=(2,))
        outputs = tf.keras.layers.Dense(
            2,
            activation="softmax",
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )(inputs)
        model = tf.keras.Model(inputs, outputs)

        x = np.ones((4, 2), dtype=np.float32)
        y = tf.one_hot([0, 1, 0, 1], depth=2)
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)

        results = module.evaluate_model(
            model,
            dataset,
            ["female", "male"],
            return_predictions=True,
        )

        self.assertIn("metrics", results)
        self.assertIn("accuracy", results["metrics"])
        self.assertIn("macro_f1", results["metrics"])
        self.assertEqual(len(results["y_true"]), 4)

    def test_legacy_evaluator_and_report_are_wrappers(self):
        canonical_evaluator = require_module(
            self,
            "wingbeat_ml.evaluation.evaluator",
        )
        canonical_report = require_module(
            self,
            "wingbeat_ml.evaluation.report",
        )

        from src.evaluation.evaluate import ModelEvaluator
        from src.evaluation.report import report_results

        self.assertIs(
            ModelEvaluator,
            canonical_evaluator.ModelEvaluator,
        )
        self.assertIs(
            report_results,
            canonical_report.report_results,
        )

    def test_training_pipelines_use_canonical_evaluation(self):
        for name in (
            "wingbeat_ml.pipelines.pretrain",
            "wingbeat_ml.pipelines.linear_probe",
            "wingbeat_ml.pipelines.fine_tune",
        ):
            module = importlib.import_module(name)
            source = inspect.getsource(module)
            self.assertNotIn("src.evaluation", source)


class TestQualityGates(unittest.TestCase):
    def test_passes_when_all_minimums_are_met(self):
        gates = require_module(self, "wingbeat_ml.quality.gates")

        report = gates.evaluate_quality_gates(
            {"accuracy": 0.90, "macro_f1": 0.82},
            {"accuracy": 0.85, "macro_f1": 0.80},
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["failed"], [])

    def test_fails_for_low_or_missing_metric(self):
        gates = require_module(self, "wingbeat_ml.quality.gates")

        report = gates.evaluate_quality_gates(
            {"accuracy": 0.70},
            {"accuracy": 0.85, "macro_f1": 0.80},
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            report["failed"],
            ["accuracy", "macro_f1"],
        )

    def test_report_is_written_as_json(self):
        gates = require_module(self, "wingbeat_ml.quality.gates")
        report_module = require_module(
            self,
            "wingbeat_ml.quality.report",
        )

        report = gates.evaluate_quality_gates(
            {"macro_f1": 0.90},
            {"macro_f1": 0.80},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "quality.json")
            report_module.write_quality_report(report, path)

            with open(path, encoding="utf-8") as stream:
                stored = json.load(stream)

        self.assertTrue(stored["passed"])
        self.assertEqual(stored["checks"][0]["metric"], "macro_f1")

    def test_cli_uses_distinct_exit_code_for_gate_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            report_path = os.path.join(tmpdir, "quality.json")

            with open(metrics_path, "w", encoding="utf-8") as stream:
                json.dump(
                    {"metrics": {"accuracy": 0.90}},
                    stream,
                )

            passed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "wingbeat_ml",
                    "quality",
                    "validate",
                    "--metrics",
                    metrics_path,
                    "--minimum",
                    "accuracy=0.80",
                    "--output",
                    report_path,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertTrue(os.path.exists(report_path))

            failed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "wingbeat_ml",
                    "quality",
                    "validate",
                    "--metrics",
                    metrics_path,
                    "--minimum",
                    "accuracy=0.95",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(failed.returncode, 2)
            self.assertIn(
                "Quality gates failed",
                failed.stdout + failed.stderr,
            )


if __name__ == "__main__":
    unittest.main()
