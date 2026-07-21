"""Tests for the zero-argument pretraining pilot."""
import os
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

test_env = os.environ.copy()
test_env.pop("WINGBEAT_DATASET_DIR", None)

import yaml

from wingbeat_ml.pipelines import pretrain


ROOT = Path(__file__).resolve().parents[2]


class TestDefaultPretrainPilot(TestCase):
    def test_prepares_canonical_five_epoch_configuration(self):
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            project = temporary / "project"
            runtime = temporary / "runtime"
            copytree(ROOT / "configs", project / "configs")
            dataset = project / "dataset" / "MSB" / "Indoor"
            dataset.mkdir(parents=True)

            resolved_path, model_path, runtime_path = (
                pretrain.prepare_default_pilot(
                    project_root=project,
                    runtime_root=runtime,
                )
            )

            config = yaml.safe_load(
                resolved_path.read_text(encoding="utf-8")
            )
            self.assertEqual(config["train"]["epochs"], 5)
            self.assertEqual(config["train"]["batch_size"], 256)
            self.assertEqual(config["dataset"]["train_dir"], str(dataset))
            self.assertEqual(config["augment"]["noise_overlay"]["p"], 0.0)
            self.assertFalse(config["wandb"]["enabled"])
            self.assertEqual(
                model_path,
                project / "configs" / "models" / "mossong_plus.yaml",
            )
            self.assertEqual(runtime_path.parent, runtime / "pilots")

    @patch.object(pretrain.os, "chdir")
    @patch.object(pretrain, "train_supervised")
    @patch.object(pretrain, "prepare_default_pilot")
    def test_bare_main_selects_pilot(
        self,
        prepare_default_pilot,
        train_supervised,
        change_directory,
    ):
        resolved = Path("/tmp/resolved.yaml")
        model = Path("/tmp/model.yaml")
        runtime = Path("/tmp/runtime")
        prepare_default_pilot.return_value = (resolved, model, runtime)

        with patch.dict(pretrain.os.environ, {}, clear=False):
            pretrain.main([])
            self.assertEqual(
                pretrain.os.environ["WINGBEAT_RUNTIME_ROOT"],
                str(runtime),
            )

        change_directory.assert_called_once_with(runtime)
        train_supervised.assert_called_once_with(
            defaults_path=resolved,
            model_cfg_path=model,
        )

    @patch.object(pretrain.os, "chdir")
    @patch.object(pretrain, "train_supervised")
    @patch.object(pretrain, "prepare_default_pilot")
    def test_explicit_paths_preserve_existing_behavior(
        self,
        prepare_default_pilot,
        train_supervised,
        change_directory,
    ):
        pretrain.main(
            [
                "--defaults_path",
                "custom-defaults.yaml",
                "--model_cfg_path",
                "custom-model.yaml",
            ]
        )

        prepare_default_pilot.assert_not_called()
        change_directory.assert_not_called()
        train_supervised.assert_called_once_with(
            defaults_path="custom-defaults.yaml",
            model_cfg_path="custom-model.yaml",
        )


if __name__ == "__main__":
    import unittest
    unittest.main()
