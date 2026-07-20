"""Tests for lineage tracking and W&B model promotion."""

import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest


def require_module(testcase, name):
    testcase.assertIsNotNone(
        importlib.util.find_spec(name),
        f"missing Phase 8 module: {name}",
    )
    return importlib.import_module(name)


class FakeArtifact:
    def __init__(self, name, type, metadata=None):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []
        self.waited = False
        self.qualified_name = f"entity/project/{name}:v0"

    def add_file(self, path):
        self.files.append(path)

    def wait(self):
        self.waited = True
        return self


class FakeRun:
    def __init__(self):
        self.logged = []
        self.used = []
        self.linked = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def log_artifact(self, artifact):
        self.logged.append(artifact)
        return artifact

    def use_artifact(self, reference):
        self.used.append(reference)
        artifact = FakeArtifact(reference, "model")
        artifact.qualified_name = reference
        return artifact

    def link_artifact(self, *, artifact, target_path, aliases):
        self.linked.append((artifact, target_path, aliases))
        return artifact


class FakeWandb:
    def __init__(self):
        self.run = FakeRun()
        self.config = {}
        self.init_calls = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self.run

    def Artifact(self, **kwargs):
        return FakeArtifact(**kwargs)


class TestLineage(unittest.TestCase):
    def test_lineage_contains_model_hash(self):
        lineage = require_module(
            self,
            "wingbeat_ml.tracking.lineage",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.keras")
            output_path = os.path.join(tmpdir, "lineage.json")

            with open(model_path, "wb") as stream:
                stream.write(b"model-contents")

            result = lineage.build_lineage(
                model_path=model_path,
                metrics={"macro_f1": 0.91},
                config_sha256="config-hash",
                dataset_sha256="dataset-hash",
                git_commit="abc1234",
            )
            lineage.write_lineage(result, output_path)

            with open(output_path, encoding="utf-8") as stream:
                stored = json.load(stream)

        self.assertEqual(stored["schema_version"], 1)
        self.assertEqual(stored["model"]["name"], "model.keras")
        self.assertEqual(len(stored["model"]["sha256"]), 64)
        self.assertEqual(stored["metrics"]["macro_f1"], 0.91)


class TestPromotion(unittest.TestCase):
    def test_training_run_uses_shared_metadata_and_sweep_overrides(self):
        tracking = require_module(
            self,
            "wingbeat_ml.tracking.wandb",
        )
        fake = FakeWandb()
        fake.config = {
            "optimizer.learning_rate": 0.02,
        }
        config = {
            "wandb": {
                "enabled": True,
                "project": "MosSongPlus",
                "group": "smoke",
                "tags": ["ci"],
                "job_type": "train",
            },
            "optimizer": {"learning_rate": 0.01},
        }

        run = tracking.initialize_training_run(
            config,
            wandb_module=fake,
        )

        self.assertIs(run, fake.run)
        self.assertEqual(fake.init_calls[0]["group"], "smoke")
        self.assertEqual(fake.init_calls[0]["tags"], ["ci"])
        self.assertEqual(fake.init_calls[0]["job_type"], "train")
        self.assertEqual(config["optimizer"]["learning_rate"], 0.02)

    def test_dry_run_does_not_initialize_wandb(self):
        promotion = require_module(
            self,
            "wingbeat_ml.pipelines.promote",
        )
        fake = FakeWandb()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.keras")
            with open(model_path, "wb") as stream:
                stream.write(b"model")

            result = promotion.promote_candidate(
                metrics={"macro_f1": 0.90},
                minimums={"macro_f1": 0.80},
                registry="Models",
                collection="mossong-plus",
                model_path=model_path,
                wandb_module=fake,
            )

        self.assertFalse(result["promoted"])
        self.assertTrue(result["quality"]["passed"])
        self.assertEqual(fake.init_calls, [])

    def test_failed_gate_prevents_remote_promotion(self):
        promotion = require_module(
            self,
            "wingbeat_ml.pipelines.promote",
        )
        fake = FakeWandb()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.keras")
            with open(model_path, "wb") as stream:
                stream.write(b"model")

            result = promotion.promote_candidate(
                metrics={"macro_f1": 0.60},
                minimums={"macro_f1": 0.80},
                registry="Models",
                collection="mossong-plus",
                model_path=model_path,
                project="MosSongPlus",
                execute=True,
                wandb_module=fake,
            )

        self.assertFalse(result["promoted"])
        self.assertEqual(fake.init_calls, [])

    def test_execute_links_approved_model(self):
        promotion = require_module(
            self,
            "wingbeat_ml.pipelines.promote",
        )
        fake = FakeWandb()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.keras")
            with open(model_path, "wb") as stream:
                stream.write(b"model")

            result = promotion.promote_candidate(
                metrics={"macro_f1": 0.90},
                minimums={"macro_f1": 0.80},
                registry="Models",
                collection="mossong-plus",
                model_path=model_path,
                project="MosSongPlus",
                aliases=["candidate"],
                execute=True,
                wandb_module=fake,
            )

        self.assertTrue(result["promoted"])
        self.assertEqual(len(fake.run.logged), 1)
        self.assertEqual(
            fake.run.linked[0][1],
            "wandb-registry-Models/mossong-plus",
        )
        self.assertEqual(fake.run.linked[0][2], ["candidate"])

    def test_cli_dry_run_and_gate_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.keras")
            metrics_path = os.path.join(tmpdir, "metrics.json")

            with open(model_path, "wb") as stream:
                stream.write(b"model")
            with open(metrics_path, "w", encoding="utf-8") as stream:
                json.dump(
                    {"metrics": {"macro_f1": 0.90}},
                    stream,
                )

            command = [
                sys.executable,
                "-m",
                "wingbeat_ml",
                "promote",
                "--model",
                model_path,
                "--metrics",
                metrics_path,
                "--registry",
                "Models",
                "--collection",
                "mossong-plus",
            ]

            passed = subprocess.run(
                command + ["--minimum", "macro_f1=0.80"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertIn("Promotion dry run passed", passed.stdout)

            failed = subprocess.run(
                command + ["--minimum", "macro_f1=0.95"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(failed.returncode, 2)
            self.assertIn(
                "Promotion blocked by quality gates",
                failed.stdout + failed.stderr,
            )


if __name__ == "__main__":
    unittest.main()
