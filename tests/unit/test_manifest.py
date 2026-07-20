"""Unit tests for wingbeat_ml.data.manifest.

Tests cover:
  - generate_manifest determinism
  - Manifest paths are relative (not absolute)
  - manifest_sha256 excludes created_at and volatile fields
  - write_manifest / load_manifest round-trip
  - verify_manifest metadata mode
  - verify_manifest full-hash mode
  - Schema version is always 1
  - FileRecord fields are correct types
  - Empty manifest
"""

import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from wingbeat_ml.data.manifest import (
    DatasetManifest,
    FileRecord,
    generate_manifest,
    load_manifest,
    manifest_sha256,
    verify_manifest,
    write_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(path: str, nframes: int = 2400, sr: int = 8000) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        signal = (np.sin(np.linspace(0, np.pi, nframes)) * 16383).astype(np.int16)
        wf.writeframes(signal.tobytes())


def _make_fixture_dir(tmp_dir: str, n_files: int = 3) -> tuple[list[str], list[str], list[int], list[str]]:
    """Create a fixture dataset with WAV files in per-class subdirectories."""
    classes = ["Ae_aegypti_F", "Ae_aegypti_M"]
    root = Path(tmp_dir)
    file_paths, labels, label_indices, splits = [], [], [], []
    idx = 0
    for i, cls in enumerate(classes):
        (root / cls).mkdir(exist_ok=True)
        for j in range(n_files):
            p = str(root / cls / f"file_{j}.wav")
            _make_wav(p)
            file_paths.append(p)
            labels.append(cls)
            label_indices.append(i)
            splits.append("train" if idx % 2 == 0 else "test")
            idx += 1
    return file_paths, labels, label_indices, splits


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.file_paths, self.labels, self.label_indices, self.splits = (
            _make_fixture_dir(self.tmp.name)
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_schema_version_is_1(self):
        m = generate_manifest(
            self.tmp.name, self.file_paths, self.labels, self.label_indices, self.splits
        )
        self.assertEqual(m.schema_version, 1)

    def test_file_count(self):
        m = generate_manifest(
            self.tmp.name, self.file_paths, self.labels, self.label_indices, self.splits
        )
        self.assertEqual(len(m.files), len(self.file_paths))

    def test_paths_are_relative(self):
        m = generate_manifest(
            self.tmp.name, self.file_paths, self.labels, self.label_indices, self.splits
        )
        for rec in m.files:
            self.assertFalse(Path(rec.path).is_absolute(), f"Expected relative path: {rec.path}")

    def test_label_indices_correct(self):
        m = generate_manifest(
            self.tmp.name, self.file_paths, self.labels, self.label_indices, self.splits
        )
        for rec in m.files:
            self.assertIsInstance(rec.label_index, int)

    def test_records_sorted_by_split_label_path(self):
        m = generate_manifest(
            self.tmp.name, self.file_paths, self.labels, self.label_indices, self.splits
        )
        keys = [(r.split, r.label, r.path) for r in m.files]
        self.assertEqual(keys, sorted(keys))

    def test_determinism(self):
        m1 = generate_manifest(
            self.tmp.name, self.file_paths, self.labels, self.label_indices, self.splits
        )
        m2 = generate_manifest(
            self.tmp.name, self.file_paths, self.labels, self.label_indices, self.splits
        )
        self.assertEqual(
            [(r.path, r.sha256, r.label_index) for r in m1.files],
            [(r.path, r.sha256, r.label_index) for r in m2.files],
        )

    def test_no_hash_option(self):
        m = generate_manifest(
            self.tmp.name, self.file_paths, self.labels, self.label_indices, self.splits,
            full_hash=False,
        )
        for rec in m.files:
            self.assertEqual(rec.sha256, "")

    def test_size_bytes_positive(self):
        m = generate_manifest(
            self.tmp.name, self.file_paths, self.labels, self.label_indices, self.splits,
            full_hash=False,
        )
        for rec in m.files:
            self.assertGreater(rec.size_bytes, 0)


    def test_mismatched_columns_raise(self):
        with self.assertRaises(ValueError):
            generate_manifest(
                self.tmp.name,
                self.file_paths,
                self.labels[:-1],
                self.label_indices,
                self.splits,
            )

    def test_file_outside_dataset_root_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as outside:
            with self.assertRaises(ValueError):
                generate_manifest(
                    self.tmp.name,
                    [outside.name],
                    ["Ae_aegypti_F"],
                    [0],
                    ["train"],
                    full_hash=False,
                )


class TestManifestSha256(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.file_paths, self.labels, self.label_indices, self.splits = (
            _make_fixture_dir(self.tmp.name)
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _make(self) -> DatasetManifest:
        return generate_manifest(
            self.tmp.name, self.file_paths, self.labels, self.label_indices, self.splits,
            full_hash=False,
        )

    def test_hash_is_hex_string(self):
        m = self._make()
        h = manifest_sha256(m)
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)
        # Must be hex
        int(h, 16)

    def test_hash_excludes_created_at(self):
        m1 = self._make()
        m1.created_at = "2020-01-01T00:00:00+00:00"
        m2 = self._make()
        m2.created_at = "2099-12-31T23:59:59+00:00"
        self.assertEqual(manifest_sha256(m1), manifest_sha256(m2))

    def test_hash_changes_when_files_change(self):
        m1 = self._make()
        m2 = self._make()
        m2.files[0].label_index = 99
        self.assertNotEqual(manifest_sha256(m1), manifest_sha256(m2))

    def test_hash_deterministic(self):
        m1 = self._make()
        m2 = self._make()
        self.assertEqual(manifest_sha256(m1), manifest_sha256(m2))


class TestWriteLoadManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.file_paths, self.labels, self.label_indices, self.splits = (
            _make_fixture_dir(self.tmp.name)
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip(self):
        out_path = str(Path(self.tmp.name) / "manifest.json")
        m = generate_manifest(
            self.tmp.name, self.file_paths, self.labels, self.label_indices, self.splits,
            full_hash=False,
        )
        write_manifest(m, out_path)
        loaded = load_manifest(out_path)
        self.assertEqual(loaded.schema_version, m.schema_version)
        self.assertEqual(len(loaded.files), len(m.files))
        for orig, loaded_rec in zip(m.files, loaded.files):
            self.assertEqual(orig.path, loaded_rec.path)
            self.assertEqual(orig.label_index, loaded_rec.label_index)
            self.assertEqual(orig.split, loaded_rec.split)

    def test_written_json_has_required_keys(self):
        out_path = str(Path(self.tmp.name) / "manifest2.json")
        m = generate_manifest(
            self.tmp.name, self.file_paths, self.labels, self.label_indices, self.splits,
            full_hash=False,
        )
        write_manifest(m, out_path)
        data = json.loads(Path(out_path).read_text())
        for key in ("schema_version", "dataset_name", "dataset_version",
                    "sample_rate", "segment_length", "split_seed", "files"):
            self.assertIn(key, data)


class TestVerifyManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.file_paths, self.labels, self.label_indices, self.splits = (
            _make_fixture_dir(self.tmp.name)
        )
        self.manifest = generate_manifest(
            self.tmp.name, self.file_paths, self.labels, self.label_indices, self.splits,
            full_hash=True,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_issues_for_valid_dataset(self):
        issues = verify_manifest(self.manifest, self.tmp.name, full_hash=False)
        self.assertEqual(issues["missing"], [])
        self.assertEqual(issues["size_mismatch"], [])
        self.assertEqual(issues["hash_mismatch"], [])

    def test_missing_file_detected(self):
        m = generate_manifest(
            self.tmp.name, self.file_paths, self.labels, self.label_indices, self.splits,
            full_hash=False,
        )
        m.files[0].path = "nonexistent/file.wav"
        issues = verify_manifest(m, self.tmp.name, full_hash=False)
        self.assertIn("nonexistent/file.wav", issues["missing"])

    def test_full_hash_verification(self):
        issues = verify_manifest(self.manifest, self.tmp.name, full_hash=True)
        self.assertEqual(issues["hash_mismatch"], [])


class TestEmptyManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_files_list(self):
        m = generate_manifest(self.tmp.name, [], [], [], [], full_hash=False)
        self.assertEqual(m.files, [])
        h = manifest_sha256(m)
        self.assertEqual(len(h), 64)


if __name__ == "__main__":
    unittest.main()
