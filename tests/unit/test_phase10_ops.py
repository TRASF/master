"""Contract tests for CI and W&B Launch operations."""

from pathlib import Path
import subprocess
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = (
    ".github/workflows/ci.yaml",
    "ops/wandb/train-pretrain.sh",
    "ops/wandb/launch-config.yaml",
    "ops/wandb/README.md",
    "README.md",
)


class TestPhase10Operations(unittest.TestCase):
    def test_required_files_exist(self):
        for name in REQUIRED:
            with self.subTest(path=name):
                self.assertTrue(
                    (ROOT / name).is_file(),
                    f"Phase 10 file is missing: {name}",
                )

    def test_ci_runs_package_quality_gates(self):
        path = ROOT / ".github/workflows/ci.yaml"
        self.assertTrue(path.is_file(), "Phase 10 file is missing: ci.yaml")
        workflow = yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=yaml.BaseLoader,
        )
        triggers = workflow["on"]
        for name in ("push", "pull_request", "workflow_dispatch"):
            self.assertIn(name, triggers)

        source = path.read_text(encoding="utf-8")
        for command in (
            "pytest -q --tb=short",
            "python -m build --wheel",
            "bash ops/wandb/train-pretrain.sh",
        ):
            self.assertIn(command, source)
        self.assertIn("CUDA_VISIBLE_DEVICES", source)
        self.assertIn("WANDB_MODE", source)

    def test_launch_agent_is_bounded(self):
        path = ROOT / "ops/wandb/launch-config.yaml"
        self.assertTrue(
            path.is_file(),
            "Phase 10 file is missing: launch-config.yaml",
        )
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(config["max_jobs"], 1)
        self.assertEqual(config["queues"], ["wingbeat-training"])
        self.assertEqual(config["builder"]["type"], "docker")

    def test_training_entrypoint_is_valid_and_canonical(self):
        path = ROOT / "ops/wandb/train-pretrain.sh"
        self.assertTrue(
            path.is_file(),
            "Phase 10 file is missing: train-pretrain.sh",
        )
        result = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        source = path.read_text(encoding="utf-8")
        self.assertIn("python -m wingbeat_ml config resolve", source)
        self.assertIn("python -m wingbeat_ml.pipelines.pretrain", source)
        self.assertIn("WINGBEAT_RUNTIME_ROOT", source)
        self.assertIn("WINGBEAT_DATASET_DIR", source)


if __name__ == "__main__":
    unittest.main()
