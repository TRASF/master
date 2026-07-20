import os
import unittest
import warnings
import tempfile
import yaml
from wingbeat_ml.config import resolve_config

class TestConfigParity(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_legacy_key_mapping_and_warnings(self):
        # We want to verify that using legacy flat keys in config file raises warnings
        # and correctly populates the canonical path (e.g. reproducibility.seed).
        legacy_yaml_path = os.path.join(self.temp_dir.name, "legacy.yaml")
        legacy_data = {
            "train.seed": 99,
            "optimizer.learning_rate": 0.005
        }
        with open(legacy_yaml_path, "w") as f:
            yaml.safe_dump(legacy_data, f)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = resolve_config(
                base_path="configs/base.yaml",
                experiment_path=legacy_yaml_path
            )

            # Check warning was raised
            deprecation_warnings = [
                warn for warn in w
                if issubclass(warn.category, DeprecationWarning)
            ]
            self.assertTrue(len(deprecation_warnings) >= 1)

            # Verify compatibility preserved and mapped
            self.assertEqual(cfg["reproducibility"]["seed"], 99)
            self.assertEqual(cfg["optimizer"]["learning_rate"], 0.005)

    def test_canonical_precedence_over_legacy(self):
        # If both legacy and canonical key are supplied in config file, canonical key must win.
        legacy_yaml_path = os.path.join(self.temp_dir.name, "legacy.yaml")
        legacy_data = {
            "train.seed": 99,
            "reproducibility": {
                "seed": 42
            }
        }
        with open(legacy_yaml_path, "w") as f:
            yaml.safe_dump(legacy_data, f)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = resolve_config(
                base_path="configs/base.yaml",
                experiment_path=legacy_yaml_path
            )

            # reproducibility.seed is canonical, should win
            self.assertEqual(cfg["reproducibility"]["seed"], 42)

            # Verify warning was raised about ignoring legacy key
            ignore_warns = [
                str(warn.message) for warn in w
                if "ignored in favor of" in str(warn.message)
            ]
            self.assertTrue(len(ignore_warns) >= 1)

    def test_three_mode_representation(self):
        # Check pretrain
        pretrain_cfg = resolve_config(
            base_path="configs/base.yaml",
            model_path="configs/models/mossong_plus.yaml",
            experiment_path="configs/experiments/pretrain.yaml",
            profile_path="configs/profiles/local.yaml"
        )
        self.assertEqual(pretrain_cfg["training_mode"], "pretrain")
        self.assertEqual(pretrain_cfg["model"]["id"], "mossong_plus")

        # Check linear_probe
        lp_cfg = resolve_config(
            base_path="configs/base.yaml",
            model_path="configs/models/mossong_plus.yaml",
            experiment_path="configs/experiments/linear_probe.yaml",
            profile_path="configs/profiles/local.yaml"
        )
        self.assertEqual(lp_cfg["training_mode"], "linear_probe")

        # Check fine_tune
        ft_cfg = resolve_config(
            base_path="configs/base.yaml",
            model_path="configs/models/mossong_plus.yaml",
            experiment_path="configs/experiments/fine_tune.yaml",
            profile_path="configs/profiles/local.yaml"
        )
        self.assertEqual(ft_cfg["training_mode"], "fine_tune")

if __name__ == "__main__":
    unittest.main()
