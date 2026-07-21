"""Contracts for staging and centralizing completed training runs."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from wingbeat_ml.ops.artifact_handoff import (
    finalize_run,
    sanitize_run_id,
    sync_ready_runs,
    verify_run,
)


ROOT = Path(__file__).resolve().parents[2]


class TestArtifactHandoff(unittest.TestCase):
    def test_finalize_publishes_an_atomic_verifiable_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "run-123"
            (run_dir / "models").mkdir(parents=True)
            (run_dir / "models" / "best.weights.h5").write_bytes(b"weights")

            ready = finalize_run(run_dir, exit_code=0)
            result = verify_run(run_dir)

            self.assertEqual(ready.name, ".artifact-ready.json")
            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(result["file_count"], 2)
            self.assertTrue((run_dir / "run-status.json").is_file())

    def test_verification_rejects_a_changed_staged_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-123"
            run_dir.mkdir()
            model = run_dir / "best.weights.h5"
            model.write_bytes(b"original")
            finalize_run(run_dir, exit_code=1)

            model.write_bytes(b"changed")

            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                verify_run(run_dir)

    def test_local_sync_verifies_destination_and_preserves_staging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staging = root / "staging"
            destination = root / "miru4090s"
            run_dir = staging / "runs" / "run-123"
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.jsonl").write_text('{"epoch": 1}\n')
            finalize_run(run_dir, exit_code=0)

            synced = sync_ready_runs(staging, destination)

            self.assertEqual(synced, ["run-123"])
            self.assertTrue((destination / "runs" / "run-123" / ".artifact-verified.json").is_file())
            self.assertTrue((run_dir / ".artifact-synced.json").is_file())
            self.assertTrue((run_dir / "metrics.jsonl").is_file())
            verification = json.loads(
                (destination / "runs" / "run-123" / ".artifact-verified.json").read_text()
            )
            self.assertEqual(verification["run_id"], "run-123")

    def test_run_ids_cannot_escape_the_staging_directory(self):
        self.assertEqual(sanitize_run_id("sweep/run 123"), "sweep-run-123")
        with self.assertRaisesRegex(ValueError, "empty"):
            sanitize_run_id("../")

    def test_remote_sync_marks_source_only_after_rsync_and_ssh_verification(self):
        calls = []

        def run_command(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir) / "staging"
            run_dir = staging / "runs" / "run-123"
            run_dir.mkdir(parents=True)
            (run_dir / "model.keras").write_bytes(b"model")
            finalize_run(run_dir, exit_code=0)

            synced = sync_ready_runs(
                staging,
                "/media/miru4090s/New Volume1/MosSongPlus_experiments",
                owner="miru4090s@miru4090s",
                run_command=run_command,
            )

            self.assertEqual(synced, ["run-123"])
            self.assertEqual(calls[0][0][:3], ["ssh", "miru4090s@miru4090s", "python3"])
            self.assertIn("mkdir", calls[0][1]["input"])
            self.assertEqual(calls[1][0][0], "rsync")
            self.assertIn("--protect-args", calls[1][0])
            self.assertEqual(calls[2][0][:3], ["ssh", "miru4090s@miru4090s", "python3"])
            self.assertIn("New Volume1", calls[2][1]["input"])
            self.assertTrue((run_dir / ".artifact-synced.json").is_file())

    def test_failed_remote_verification_does_not_mark_source_synced(self):
        attempts = 0

        def run_command(command, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 3:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir) / "staging"
            run_dir = staging / "runs" / "run-123"
            run_dir.mkdir(parents=True)
            (run_dir / "model.keras").write_bytes(b"model")
            finalize_run(run_dir, exit_code=0)

            with self.assertRaises(subprocess.CalledProcessError):
                sync_ready_runs(
                    staging,
                    "/archive",
                    owner="miru4090s@miru4090s",
                    run_command=run_command,
                )

            self.assertFalse((run_dir / ".artifact-synced.json").exists())

    def test_training_entrypoint_isolates_and_finalizes_each_run(self):
        source = (ROOT / "ops" / "wandb" / "train-pretrain.sh").read_text()

        self.assertIn('RUNTIME_ROOT="$STAGING_ROOT/runs/$RUN_ID"', source)
        self.assertIn("artifact_handoff finalize", source)
        self.assertIn("trap finalize_artifacts EXIT", source)

    def test_artifact_sync_service_never_deletes_remote_staging(self):
        script = (ROOT / "ops" / "wandb" / "start-artifact-sync.sh").read_text()
        service = (ROOT / "ops" / "wandb" / "wingbeat-artifact-sync.service").read_text()

        self.assertIn("artifact_handoff sync", script)
        self.assertNotIn("rm -rf", script)
        self.assertIn("start-artifact-sync.sh", service)


if __name__ == "__main__":
    unittest.main()
