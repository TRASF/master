import unittest
import subprocess
import sys
import os
import tempfile
import yaml

class TestConfigCLI(unittest.TestCase):
    def run_cli(self, args):
        cmd = [sys.executable, "-m", "wingbeat_ml"] + args
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res

    def test_help(self):
        res = self.run_cli(["--help"])
        self.assertEqual(res.returncode, 0)
        self.assertIn("wingbeat_ml", res.stdout)
        self.assertIn("config", res.stdout)

    def test_version(self):
        res = self.run_cli(["--version"])
        self.assertEqual(res.returncode, 0)
        self.assertIn("wingbeat_ml version", res.stdout)

    def test_validate_valid(self):
        res = self.run_cli([
            "config", "validate",
            "--base", "configs/base.yaml",
            "--model", "configs/models/mossong_plus.yaml",
            "--experiment", "configs/experiments/pretrain.yaml",
            "--profile", "configs/profiles/ci.yaml"
        ])
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("Configuration is valid", res.stdout)
        self.assertIn("Hash:", res.stdout)

    def test_validate_invalid_training_mode(self):
        res = self.run_cli([
            "config", "validate",
            "--base", "configs/base.yaml",
            "--model", "configs/models/mossong_plus.yaml",
            "--experiment", "configs/experiments/pretrain.yaml",
            "--profile", "configs/profiles/ci.yaml",
            "--set", "training_mode=invalid_mode"
        ])
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Invalid training mode", res.stderr + res.stdout)

    def test_unknown_override_failure(self):
        res = self.run_cli([
            "config", "validate",
            "--base", "configs/base.yaml",
            "--set", "nonexistent.key=123"
        ])
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("does not exist in the configuration", res.stderr + res.stdout)

    def test_resolve_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "resolved.yaml")
            res = self.run_cli([
                "config", "resolve",
                "--base", "configs/base.yaml",
                "--model", "configs/models/mossong_plus.yaml",
                "--experiment", "configs/experiments/pretrain.yaml",
                "--profile", "configs/profiles/ci.yaml",
                "--output", out_path
            ])
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertTrue(os.path.exists(out_path))
            self.assertTrue(os.path.exists(out_path.replace(".yaml", ".sha256")))
            self.assertIn("Resolved configuration saved to", res.stdout)

            with open(out_path, "r") as f:
                resolved_data = yaml.safe_load(f)
            self.assertEqual(resolved_data["training_mode"], "pretrain")

    def test_no_side_effects_imports(self):
        cmd = [
            sys.executable, "-c",
            "import sys; from wingbeat_ml.config import load_config; assert 'tensorflow' not in sys.modules; assert 'wandb' not in sys.modules"
        ]
        res = subprocess.run(cmd, capture_output=True)
        self.assertEqual(res.returncode, 0, res.stderr)

if __name__ == "__main__":
    unittest.main()
