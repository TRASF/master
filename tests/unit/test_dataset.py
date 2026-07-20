"""Unit tests for wingbeat_ml.data.dataset (TF dataset construction).

Tests cover:
  - build_datasets returns three tf.data.Datasets
  - Tensor shapes match (batch, segment_length)
  - Tensor dtype is float32 for audio, int32 for labels (one-hot is float32)
  - Batch shape is correct
  - Label dtype (one-hot = float32)
  - Determinism with fixed seed

NOTE: These tests require TensorFlow and a real (tiny) dataset directory.
They skip if dataset creation fails due to missing data.
"""

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(path: str, nframes: int = 4800, sr: int = 8000) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        signal = (np.sin(np.linspace(0, np.pi * 10, nframes)) * 16383).astype(np.int16)
        wf.writeframes(signal.tobytes())


def _make_fixture_dataset(tmp_dir: str, n_per_class: int = 5, sr: int = 8000) -> dict:
    """Create a tiny per-class WAV fixture dataset."""
    classes = ["Ae_aegypti_F", "Ae_aegypti_M"]
    root = Path(tmp_dir)
    for cls in classes:
        (root / cls).mkdir()
        for i in range(n_per_class):
            _make_wav(str(root / cls / f"file_{i:02d}.wav"), nframes=9600, sr=sr)
    return {"root": str(root), "classes": classes}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildDatasets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fixture = _make_fixture_dataset(self.tmp.name, n_per_class=6)
        try:
            import tensorflow as tf
            self._tf_available = True
        except ImportError:
            self._tf_available = False

    def tearDown(self):
        self.tmp.cleanup()

    def _build(self, batch_size: int = 2, split=(0.6, 0.2, 0.2)):
        from wingbeat_ml.data.dataset import build_datasets
        config = {
            "reproducibility": {
                "seed": 42,
                "deterministic_data": True,
            },
            "train": {
                "batch_size": batch_size,
                "shuffle": False,
            },
            "audio": {
                "segment_length": 2400,
                "sample_rate": 8000,
            },
            "dataset": {
                "split_list": list(split),
            },
            "classes": self.fixture["classes"],
            "augment": {},
        }
        return build_datasets(self.fixture["root"], config)

    def test_returns_three_datasets(self):
        if not self._tf_available:
            self.skipTest("TensorFlow not available")
        import tensorflow as tf
        result = self._build()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        for ds in result:
            self.assertIsInstance(ds, tf.data.Dataset)

    def test_batch_audio_shape(self):
        if not self._tf_available:
            self.skipTest("TensorFlow not available")
        import tensorflow as tf
        train_ds, _, _ = self._build(batch_size=2)
        for batch in train_ds.take(1):
            audio, labels = batch
            # Model input shape is (segment_length, channel).
            self.assertEqual(tuple(audio.shape[-2:]), (2400, 1))
            self.assertEqual(audio.dtype, tf.float32)

    def test_label_dtype(self):
        if not self._tf_available:
            self.skipTest("TensorFlow not available")
        import tensorflow as tf
        train_ds, _, _ = self._build(batch_size=2)
        for batch in train_ds.take(1):
            audio, labels = batch
            # one_hot=True → float32 labels
            self.assertEqual(labels.dtype, tf.float32)

    def test_label_shape_one_hot(self):
        if not self._tf_available:
            self.skipTest("TensorFlow not available")
        import tensorflow as tf
        train_ds, _, _ = self._build(batch_size=2)
        n_classes = len(self.fixture["classes"])
        for batch in train_ds.take(1):
            audio, labels = batch
            self.assertEqual(labels.shape[-1], n_classes)

    def test_deterministic_with_same_seed(self):
        if not self._tf_available:
            self.skipTest("TensorFlow not available")
        import tensorflow as tf
        ds1, _, _ = self._build(batch_size=2)
        ds2, _, _ = self._build(batch_size=2)
        for (b1, l1), (b2, l2) in zip(ds1.take(2), ds2.take(2)):
            np.testing.assert_array_equal(l1.numpy(), l2.numpy())


if __name__ == "__main__":
    unittest.main()
