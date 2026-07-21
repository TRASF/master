"""Contracts for training throughput and per-GPU W&B execution."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import yaml

from wingbeat_ml.config.runtime import resolve_class_weights
from wingbeat_ml.config.schema import validate_config


ROOT = Path(__file__).resolve().parents[2]
HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None


def minimal_config():
    return {
        "training_mode": "pretrain",
        "model": {"id": "mossong_plus", "input_shape": [2400, 1]},
        "audio": {"sample_rate": 8000, "segment_length": 2400},
        "train": {"epochs": 2, "batch_size": 2},
        "dataset": {"train_dir": "dataset"},
        "labels": {f"class_{index}": index for index in range(11)},
        "num_classes": 11,
        "performance": {
            "precision": "auto",
            "steps_per_call": 3,
            "jit_compile": False,
            "profiler": {"enabled": False, "start_step": 10, "num_steps": 10},
        },
        "logging": {
            "console": "normal",
            "epoch_interval": 1,
            "model_summary": False,
            "classification_report": "file",
            "jsonl": True,
        },
        "class_weights": {"mode": "auto"},
        "evaluation": {
            "sample_level": {"enabled": True},
            "file_level": {"enabled": False},
        },
    }


class TestClassWeightPolicy(unittest.TestCase):
    def test_auto_uses_dataset_weights(self):
        enabled, weights = resolve_class_weights(
            {"mode": "auto"},
            np.array([0.5, 2.0], dtype=np.float32),
            2,
            labels_dict={"female": 0, "male": 1},
        )
        self.assertTrue(enabled)
        np.testing.assert_array_equal(weights, [0.5, 2.0])

    def test_manual_resolves_names_through_label_map(self):
        enabled, weights = resolve_class_weights(
            {"mode": "manual", "values": {"female": 1.25, "male": 0.75}},
            np.ones(2, dtype=np.float32),
            2,
            labels_dict={"female": 0, "male": 1},
        )
        self.assertTrue(enabled)
        np.testing.assert_array_equal(weights, [1.25, 0.75])

    def test_manual_rejects_unknown_or_duplicate_class(self):
        with self.assertRaisesRegex(ValueError, "Unknown class weight name"):
            resolve_class_weights(
                {"mode": "manual", "values": {"femlae": 1.0, "male": 1.0}},
                np.ones(2, dtype=np.float32),
                2,
                labels_dict={"female": 0, "male": 1},
            )

        with self.assertRaisesRegex(ValueError, "Duplicate class weight"):
            resolve_class_weights(
                {"mode": "manual", "values": {"female": 1.0, "FEMALE": 2.0}},
                np.ones(2, dtype=np.float32),
                2,
                labels_dict={"female": 0, "male": 1},
            )

    def test_off_returns_no_weights(self):
        enabled, weights = resolve_class_weights(
            {"mode": "off"},
            np.ones(2, dtype=np.float32),
            2,
            labels_dict={"female": 0, "male": 1},
        )
        self.assertFalse(enabled)
        self.assertIsNone(weights)


class TestConfigurationContracts(unittest.TestCase):
    def test_approved_performance_configuration_is_valid(self):
        validate_config(minimal_config())

    def test_invalid_performance_and_logging_values_fail(self):
        for path, value, message in (
            (("performance", "precision"), "bf16", "performance.precision"),
            (("performance", "steps_per_call"), 0, "performance.steps_per_call"),
            (("logging", "console"), "chatty", "logging.console"),
            (("logging", "epoch_interval"), 0, "logging.epoch_interval"),
            (("class_weights", "mode"), "guess", "class_weights.mode"),
        ):
            config = minimal_config()
            config[path[0]][path[1]] = value
            with self.subTest(path=".".join(path)):
                with self.assertRaisesRegex(ValueError, message):
                    validate_config(config)

    def test_default_policy_is_sample_only_auto_weighted(self):
        config = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
        self.assertEqual(config["class_weights"]["mode"], "auto")
        self.assertTrue(config["evaluation"]["sample_level"]["enabled"])
        self.assertFalse(config["evaluation"]["file_level"]["enabled"])
        self.assertEqual(config["performance"]["steps_per_call"], 20)
        self.assertEqual(config["logging"]["console"], "normal")

    def test_fine_tune_has_no_warmup_subsystem(self):
        source = (ROOT / "src/wingbeat_ml/pipelines/fine_tune.py").read_text()
        defaults = (ROOT / "configs/defaults.yaml").read_text()
        self.assertNotIn("_run_warmup", source)
        self.assertNotIn("_build_warmup_dataset", source)
        self.assertNotIn("warmup_epochs", defaults)
        self.assertNotIn("warmup_augment_p", defaults)

    def test_auto_precision_selects_mixed_only_for_supported_gpu(self):
        from wingbeat_ml.config.runtime import configure_compute_policy

        tensorflow = mock.Mock()
        tensorflow.config.list_physical_devices.return_value = ["gpu:0"]
        tensorflow.config.experimental.get_device_details.return_value = {
            "compute_capability": (8, 9)
        }

        policy = configure_compute_policy(
            {"precision": "auto"},
            tf_module=tensorflow,
        )

        self.assertEqual(policy, "mixed_float16")
        tensorflow.keras.mixed_precision.set_global_policy.assert_called_once_with(
            "mixed_float16"
        )


class TestEvaluationAndLogging(unittest.TestCase):
    def test_disabled_file_level_evaluation_performs_no_work(self):
        from wingbeat_ml.pipelines.helpers.reporting import evaluate_training_run

        evaluator = mock.Mock()
        evaluator.evaluate_final_test.return_value = {
            "metrics": {"accuracy": 1.0, "macro_f1": 1.0},
            "report": {},
            "confusion_matrix": [[1]],
        }
        config = minimal_config()
        config["classes"] = ["female", "male"]
        config["wandb"] = {"enabled": False}
        builder = mock.Mock()

        report = mock.Mock()
        evaluation_module = types.ModuleType("wingbeat_ml.evaluation")
        evaluation_module.report_results = report
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(
                "sys.modules",
                {"wingbeat_ml.evaluation": evaluation_module},
            ):
                evaluate_training_run(
                    model=mock.Mock(),
                    evaluator=evaluator,
                    dataset_builder=builder,
                    config=config,
                    checkpoint_path=str(Path(tmpdir) / "missing.weights.h5"),
                    results_dir=tmpdir,
                    artifact_name="candidate",
                    validation_dataset=object(),
                    test_dataset=object(),
                )

        evaluator.evaluate_files.assert_not_called()
        self.assertIsNone(report.call_args.kwargs["file_results"])
        self.assertIsNone(report.call_args.kwargs["train_file_results"])

    def test_console_modes_and_epoch_interval(self):
        from wingbeat_ml.pipelines.helpers.reporting import make_epoch_printer

        config = minimal_config()
        logs = {
            "train_loss": 1.0,
            "train_accuracy": 0.5,
            "val_loss": 0.9,
            "val_accuracy": 0.6,
            "val_macro_f1": 0.4,
            "epoch_duration_seconds": 0.5,
        }

        config["logging"]["console"] = "quiet"
        stream = io.StringIO()
        with redirect_stdout(stream):
            make_epoch_printer(config)(0, logs)
        self.assertEqual(stream.getvalue(), "")

        config["logging"]["console"] = "normal"
        config["logging"]["epoch_interval"] = 2
        stream = io.StringIO()
        with redirect_stdout(stream):
            printer = make_epoch_printer(config)
            printer(0, logs)
            printer(1, logs)
        self.assertNotIn("Epoch 1/2", stream.getvalue())
        self.assertIn("Epoch 2/2", stream.getvalue())

    def test_jsonl_epoch_logger_is_append_only(self):
        from wingbeat_ml.pipelines.helpers.reporting import JsonlMetricLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            logger = JsonlMetricLogger(path)
            logger.log({"epoch": 0, "value": np.float32(1.5)})
            logger.log({"epoch": 1, "value": 2.5})
            rows = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual([row["epoch"] for row in rows], [0, 1])
        self.assertEqual(rows[0]["value"], 1.5)


class TestCacheAndGpuAgents(unittest.TestCase):
    def test_stable_cache_key_ignores_file_order(self):
        self.assertTrue((ROOT / "src/wingbeat_ml/data/cache.py").is_file())
        from wingbeat_ml.data.cache import stable_cache_key

        settings = {"sample_rate": 8000, "segment_length": 2400, "dc_removal": True}
        first = stable_cache_key(["b.wav", "a.wav"], settings, manifest_sha256="abc")
        second = stable_cache_key(["a.wav", "b.wav"], settings, manifest_sha256="abc")
        self.assertEqual(first, second)
        self.assertNotEqual(
            first,
            stable_cache_key(["a.wav", "b.wav"], settings, manifest_sha256="def"),
        )

    def test_gpu_discovery_and_specs_are_unique(self):
        self.assertTrue((ROOT / "src/wingbeat_ml/ops/gpu_agents.py").is_file())
        from wingbeat_ml.ops.gpu_agents import build_agent_specs, parse_gpu_query

        output = "GPU-aaa, NVIDIA RTX 4090\nGPU-bbb, NVIDIA RTX 4090\n"
        gpus = parse_gpu_query(output)
        specs = build_agent_specs(gpus, queue="wingbeat-training")

        self.assertEqual([gpu.uuid for gpu in gpus], ["GPU-aaa", "GPU-bbb"])
        self.assertEqual(len(specs), 2)
        self.assertEqual(
            {spec.environment["CUDA_VISIBLE_DEVICES"] for spec in specs},
            {"GPU-aaa", "GPU-bbb"},
        )
        self.assertTrue(all(spec.max_jobs == 1 for spec in specs))

    def test_manifest_checksum_mismatch_is_rejected(self):
        self.assertTrue((ROOT / "src/wingbeat_ml/ops/preflight.py").is_file())
        from wingbeat_ml.ops.preflight import require_manifest_checksum

        with self.assertRaisesRegex(RuntimeError, "Dataset manifest mismatch"):
            require_manifest_checksum("expected", "actual")

    def test_docker_queue_forwards_agent_specific_gpu_and_paths(self):
        config = json.loads(
            (ROOT / "ops/wandb/docker-queue-config.json").read_text()
        )
        self.assertEqual(config["gpus"], "all")
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "WINGBEAT_DATASET_DIR",
            "WINGBEAT_RUNTIME_ROOT",
            "WINGBEAT_DATASET_MANIFEST_SHA256",
        ):
            self.assertIn(name, config["env"])
        self.assertIn(
            "/srv/wingbeat/dataset:/data:ro",
            config["volume"],
        )

    def test_precision_benchmark_profiles_and_runner_exist(self):
        fp32_path = ROOT / "configs/profiles/benchmark_fp32.yaml"
        mixed_path = ROOT / "configs/profiles/benchmark_mixed.yaml"
        runner = ROOT / "ops/benchmarks/compare-precision.sh"
        self.assertTrue(fp32_path.is_file())
        self.assertTrue(mixed_path.is_file())
        self.assertTrue(runner.is_file())

        fp32 = yaml.safe_load(fp32_path.read_text())
        mixed = yaml.safe_load(mixed_path.read_text())
        self.assertEqual(fp32["performance"]["precision"], "float32")
        self.assertEqual(mixed["performance"]["precision"], "mixed_float16")
        self.assertEqual(fp32["reproducibility"], mixed["reproducibility"])

        source = runner.read_text()
        self.assertIn("prime-cache", source)
        self.assertIn("benchmark_fp32.yaml", source)
        self.assertIn("benchmark_mixed.yaml", source)
        self.assertIn("precision-benchmark.json", source)

    def test_precision_benchmark_disables_unconfigured_noise_overlay(self):
        from wingbeat_ml.config import load_config

        for profile_name in ("benchmark_fp32.yaml", "benchmark_mixed.yaml"):
            resolved = load_config(
                base_path=str(ROOT / "configs" / "base.yaml"),
                model_path=str(ROOT / "configs" / "models" / "mossong_plus.yaml"),
                experiment_path=str(
                    ROOT / "configs" / "experiments" / "pretrain.yaml"
                ),
                profile_path=str(ROOT / "configs" / "profiles" / profile_name),
            ).data
            with self.subTest(profile=profile_name):
                self.assertEqual(resolved["augment"]["noise_overlay"]["p"], 0)
                self.assertEqual(resolved["augment"]["noise_banks"], [])


@unittest.skipUnless(HAS_TENSORFLOW, "TensorFlow is not installed")
class TestTensorFlowPerformance(unittest.TestCase):
    def test_trainer_accounts_global_steps_and_configured_steps_per_call(self):
        import tensorflow as tf
        from wingbeat_ml.training.trainer import Trainer

        x = np.ones((10, 2), dtype=np.float32)
        y = tf.one_hot([0, 1] * 5, depth=2)
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(2, activation="softmax", dtype="float32"),
        ])
        trainer = Trainer(
            model,
            tf.keras.optimizers.SGD(0.01),
            tf.keras.losses.CategoricalCrossentropy(reduction="none"),
            dataset,
            steps_per_call=3,
        )

        metrics = trainer.train_epoch()

        self.assertEqual(metrics["batches"], 5)
        self.assertEqual(metrics["global_step"], 5)
        self.assertEqual(trainer.global_step, 5)

    def test_output_layer_is_float32_under_mixed_policy(self):
        import tensorflow as tf
        from wingbeat_ml.models.mossong_plus import MosSongPlusModel

        previous = tf.keras.mixed_precision.global_policy()
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            builder = MosSongPlusModel({
                "model": {"mossong_plus": {"layers": [{"type": "flatten"}]}}
            })
            model = builder.build((2, 1), 2, output_activation=None)
            self.assertEqual(model.output.dtype, "float32")
        finally:
            tf.keras.mixed_precision.set_global_policy(previous)


if __name__ == "__main__":
    unittest.main()
