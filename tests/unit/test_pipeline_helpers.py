"""Contracts for focused pipeline coordination helpers."""

import importlib
import inspect
import os
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[2]


class TestPipelineHelperBoundary(unittest.TestCase):
    def test_focused_helper_modules_exist(self):
        for name in (
            "wingbeat_ml.pipelines.helpers.configuration",
            "wingbeat_ml.pipelines.helpers.runtime",
            "wingbeat_ml.pipelines.helpers.components",
            "wingbeat_ml.pipelines.helpers.reporting",
        ):
            self.assertIsNotNone(importlib.util.find_spec(name), name)

    def test_pilot_policy_is_owned_by_yaml(self):
        profile = yaml.safe_load(
            (ROOT / "configs" / "profiles" / "pilot.yaml").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(profile["train"]["epochs"], 5)
        self.assertEqual(profile["train"]["batch_size"], 256)
        self.assertEqual(profile["augment"]["noise_overlay"]["p"], 0.0)
        self.assertFalse(profile["wandb"]["enabled"])

        pretrain = (
            ROOT / "src" / "wingbeat_ml" / "pipelines" / "pretrain.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('"epochs": 5', pretrain)
        self.assertNotIn('"batch_size": 256', pretrain)
        self.assertNotIn("/media/", pretrain)

    def test_tracked_configuration_has_no_machine_paths(self):
        paths = list((ROOT / "configs").rglob("*.yaml"))
        paths.extend((ROOT / "ops").rglob("*.sh"))
        for path in paths:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("/app/", source, str(path))
            self.assertNotIn("/media/", source, str(path))

    def test_runtime_and_dataset_environment_override_profile(self):
        configuration = importlib.import_module(
            "wingbeat_ml.pipelines.helpers.configuration"
        )

        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            project = temporary / "project"
            runtime = temporary / "runtime-from-env"
            dataset = temporary / "dataset-from-env"
            copytree(ROOT / "configs", project / "configs")
            dataset.mkdir()

            with mock.patch.dict(
                os.environ,
                {
                    "WINGBEAT_RUNTIME_ROOT": str(runtime),
                    "WINGBEAT_DATASET_DIR": str(dataset),
                },
                clear=False,
            ):
                resolved_path, _, execution_root = (
                    configuration.prepare_default_pilot(
                        project_root=project,
                    )
                )

            resolved = yaml.safe_load(
                resolved_path.read_text(encoding="utf-8")
            )
            self.assertEqual(resolved["dataset"]["train_dir"], str(dataset))
            self.assertEqual(execution_root.parent, runtime / "pilots")

    def test_experiment_output_root_comes_from_configuration(self):
        from wingbeat_ml.config.runtime import resolve_experiment_paths

        with TemporaryDirectory() as directory:
            paths = resolve_experiment_paths(
                {"runtime": {"experiments_dir": directory}},
                "configured-run",
            )

        self.assertTrue(
            paths["save_path"].startswith(
                str(Path(directory) / "configured-run")
            )
        )

    def test_missing_operational_setting_fails_before_runtime(self):
        configuration = importlib.import_module(
            "wingbeat_ml.pipelines.helpers.configuration"
        )
        config = {
            "dataset": {
                "train_dir": "dataset",
                "val_dir": None,
                "test_dir": None,
            },
            "train": {"epochs": 1, "batch_size": 2},
            "reproducibility": {"seed": 45},
            "wandb": {"enabled": False},
        }

        with self.assertRaisesRegex(
            ValueError,
            "runtime.experiments_dir",
        ):
            configuration.validate_pipeline_configuration(config)


if __name__ == "__main__":
    unittest.main()
