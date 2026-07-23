"""Unit tests for wingbeat_ml.data.splits.

Tests cover:
  - Same seed → same split
  - Different seeds → different splits (probabilistic)
  - Split ratios are approximately correct
  - Source-recording leakage is a known issue (not tested as error)
  - Class count handling (stratified vs. fallback)
  - Small class behavior (single sample → fallback)
  - DatasetSplits contains correct types
  - Empty input handling
  - Split labels match split paths
"""

import unittest

import numpy as np

from wingbeat_ml.data.splits import (
    DatasetSplits,
    source_recording_ids,
    split_files,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_file_paths_labels(n_per_class: int = 10, n_classes: int = 2):
    """Generate synthetic file paths and labels."""
    paths, labels = [], []
    for c in range(n_classes):
        for i in range(n_per_class):
            paths.append(f"/data/class{c}/file_{i:03d}.wav")
            labels.append(c)
    return paths, labels


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSplitFilesReproducibility(unittest.TestCase):
    def setUp(self):
        self.paths, self.labels = _make_file_paths_labels(20, 3)

    def test_same_seed_same_split(self):
        s1 = split_files(self.paths, self.labels, seed=42)
        s2 = split_files(self.paths, self.labels, seed=42)
        self.assertEqual(s1.train, s2.train)
        self.assertEqual(s1.validation, s2.validation)
        self.assertEqual(s1.test, s2.test)

    def test_different_seeds_may_differ(self):
        s1 = split_files(self.paths, self.labels, seed=42)
        s2 = split_files(self.paths, self.labels, seed=123)
        # They may occasionally agree by chance but shouldn't for 60 files
        self.assertNotEqual(set(s1.train), set(s2.train))

    def test_no_overlap_between_splits(self):
        s = split_files(self.paths, self.labels, seed=42)
        train_set = set(s.train)
        val_set = set(s.validation)
        test_set = set(s.test)
        self.assertEqual(len(train_set & val_set), 0)
        self.assertEqual(len(train_set & test_set), 0)
        self.assertEqual(len(val_set & test_set), 0)

    def test_all_files_accounted_for(self):
        s = split_files(self.paths, self.labels, seed=42)
        total = len(s.train) + len(s.validation) + len(s.test)
        self.assertEqual(total, len(self.paths))

    def test_source_recordings_do_not_cross_splits(self):
        paths, labels, recordings = [], [], []
        for label in range(2):
            for recording in range(10):
                source = f"class{label}/recording{recording}"
                for segment in range(3):
                    paths.append(f"/data/{source}/segment{segment}.wav")
                    labels.append(label)
                    recordings.append(source)

        splits = split_files(paths, labels, seed=42, source_recordings=recordings)
        path_to_source = dict(zip(paths, recordings))
        grouped = [
            {path_to_source[path] for path in part}
            for part in (splits.train, splits.validation, splits.test)
        ]
        self.assertTrue(grouped[0].isdisjoint(grouped[1]))
        self.assertTrue(grouped[0].isdisjoint(grouped[2]))
        self.assertTrue(grouped[1].isdisjoint(grouped[2]))


class TestSourceRecordingIds(unittest.TestCase):
    def test_flat_segment_names_share_source(self):
        root = "/data"
        paths = [
            "/data/class0/recording-cut-01.wav.wav",
            "/data/class0/recording-cut-02.wav.wav",
            "/data/class0/other-cut-01.wav.wav",
        ]
        self.assertEqual(
            source_recording_ids(paths, root).tolist(),
            ["class0/recording-cut", "class0/recording-cut", "class0/other-cut"],
        )


class TestSplitRatios(unittest.TestCase):
    def test_approximate_ratios(self):
        paths, labels = _make_file_paths_labels(50, 2)
        s = split_files(paths, labels, split=(0.8, 0.1, 0.1), seed=42)
        n = len(paths)
        self.assertGreater(len(s.train), 0.6 * n)
        self.assertGreater(len(s.validation), 0)
        self.assertGreater(len(s.test), 0)

    def test_invalid_ratio_raises(self):
        paths, labels = _make_file_paths_labels(10, 2)
        with self.assertRaises(ValueError):
            split_files(paths, labels, split=(0.5, 0.5, 0.5))

    def test_negative_ratio_raises(self):
        paths, labels = _make_file_paths_labels(10, 2)
        with self.assertRaises(ValueError):
            split_files(paths, labels, split=(0.8, 0.3, -0.1))

    def test_mismatched_paths_and_labels_raise(self):
        paths, labels = _make_file_paths_labels(10, 2)
        with self.assertRaises(ValueError):
            split_files(paths, labels[:-1])


class TestSplitLabels(unittest.TestCase):
    def test_label_count_matches_path_count(self):
        paths, labels = _make_file_paths_labels(20, 3)
        s = split_files(paths, labels, seed=42)
        self.assertEqual(len(s.train), len(s.train_labels))
        self.assertEqual(len(s.validation), len(s.validation_labels))
        self.assertEqual(len(s.test), len(s.test_labels))

    def test_label_types_are_int(self):
        paths, labels = _make_file_paths_labels(10, 2)
        s = split_files(paths, labels, seed=42)
        for label in s.train_labels + s.validation_labels + s.test_labels:
            self.assertIsInstance(label, int)


class TestSmallClassHandling(unittest.TestCase):
    def test_single_sample_class_falls_back(self):
        """One class with 1 sample; stratify must fail gracefully."""
        paths = ["/a/class0/f0.wav", "/a/class1/f0.wav", "/a/class1/f1.wav"]
        labels = [0, 1, 1]
        # Should not raise; falls back to non-stratified split
        s = split_files(paths, labels, split=(0.6, 0.2, 0.2), seed=42)
        total = len(s.train) + len(s.validation) + len(s.test)
        self.assertEqual(total, 3)


class TestEmptyInput(unittest.TestCase):
    def test_empty_input(self):
        s = split_files([], [], seed=42)
        self.assertEqual(s.train, ())
        self.assertEqual(s.validation, ())
        self.assertEqual(s.test, ())


class TestDatasetSplitsIsImmutable(unittest.TestCase):
    def test_frozen_dataclass(self):
        paths, labels = _make_file_paths_labels(10, 2)
        s = split_files(paths, labels, seed=42)
        with self.assertRaises((AttributeError, TypeError)):
            s.train = ()  # type: ignore


class TestDefaultSeed(unittest.TestCase):
    def test_default_seed_is_42(self):
        paths, labels = _make_file_paths_labels(20, 2)
        s1 = split_files(paths, labels, seed=42)
        s2 = split_files(paths, labels)  # default seed=42
        self.assertEqual(s1.train, s2.train)


if __name__ == "__main__":
    unittest.main()
