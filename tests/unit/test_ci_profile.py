"""Regression checks for the self-contained CI profile."""

from pathlib import Path
import unittest

from wingbeat_ml.config import resolve_config


ROOT = Path(__file__).resolve().parents[2]


class TestCiProfile(unittest.TestCase):
    def test_external_noise_overlay_is_disabled(self):
        config = resolve_config(
            base_path=ROOT / "configs/base.yaml",
            model_path=ROOT / "configs/models/mossong_plus.yaml",
            experiment_path=ROOT / "configs/experiments/pretrain.yaml",
            profile_path=ROOT / "configs/profiles/ci.yaml",
        )

        self.assertEqual(
            config["augment"]["noise_overlay"]["p"],
            0.0,
        )


    def test_model_output_activation_is_explicit(self):
        config = resolve_config(
            base_path=ROOT / "configs/base.yaml",
            model_path=ROOT / "configs/models/mossong_plus.yaml",
            experiment_path=ROOT / "configs/experiments/pretrain.yaml",
            profile_path=ROOT / "configs/profiles/ci.yaml",
        )

        self.assertIn("output_activation", config["model"])
        self.assertIsNone(config["model"]["output_activation"])


if __name__ == "__main__":
    unittest.main()
