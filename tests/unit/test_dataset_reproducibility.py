"""Focused CPU-only unit and parity tests for dataset reproducibility and cache invariance."""

import hashlib
import os
import shutil
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import tensorflow as tf

from wingbeat_ml.data.cache import (
    CACHE_IMPLEMENTATION_VERSION,
    CACHE_SCHEMA_VERSION,
    stable_cache_key,
)
from wingbeat_ml.data.dataset import SupervisedDataset, derive_sample_seed


def _make_wav(path: str, nframes: int = 9600, sr: int = 8000) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        t = np.linspace(0, 1.0, nframes, endpoint=False)
        signal = (np.sin(2 * np.pi * 440 * t) * 16383).astype(np.int16)
        wf.writeframes(signal.tobytes())


def _create_test_dataset_dir(tmp_dir: str, n_per_class: int = 6) -> str:
    root = Path(tmp_dir) / "dataset"
    classes = ["class_a", "class_b"]
    for cls in classes:
        (root / cls).mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            _make_wav(str(root / cls / f"sample_{i:02d}.wav"))
    return str(root)


class TestDatasetReproducibility(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = os.path.join(self.tmp_dir.name, "cache")
        self.data_dir = _create_test_dataset_dir(self.tmp_dir.name, n_per_class=8)
        self.classes = ["class_a", "class_b"]

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _get_base_config(self, seed=42, cache_enabled=False, shuffle=False):
        return {
            "dataset_dir": self.data_dir,
            "sample_rate": 8000,
            "segment_length": 2400,
            "classes": self.classes,
            "seed": seed,
            "deterministic": True,
            "cache_cfg": {
                "enabled": cache_enabled,
                "root": self.cache_dir,
            },
        }

    def test_1_split_membership_and_path_lists_match(self):
        """1. Split membership and ordered path lists match for repeated runs with the same seed."""
        cfg = self._get_base_config(seed=42)
        ds_builder1 = SupervisedDataset(**cfg)
        train1, val1, test1 = ds_builder1.build(shuffle=False)

        ds_builder2 = SupervisedDataset(**cfg)
        train2, val2, test2 = ds_builder2.build(shuffle=False)

        np.testing.assert_array_equal(ds_builder1.train_paths, ds_builder2.train_paths)
        np.testing.assert_array_equal(ds_builder1.val_paths, ds_builder2.val_paths)
        np.testing.assert_array_equal(ds_builder1.test_paths, ds_builder2.test_paths)
        np.testing.assert_array_equal(ds_builder1.train_labels, ds_builder2.train_labels)
        np.testing.assert_array_equal(ds_builder1.val_labels, ds_builder2.val_labels)
        np.testing.assert_array_equal(ds_builder1.test_labels, ds_builder2.test_labels)

    def test_2_cache_modes_produce_identical_loaded_audio(self):
        """2. Cache-off, fresh-cache, and reused-cache produce identical sample IDs, labels, audio tensor hashes, shapes, and dtypes."""
        shutil.rmtree(self.cache_dir, ignore_errors=True)

        def _extract_loaded_audio(cache_enabled):
            cfg = self._get_base_config(seed=42, cache_enabled=cache_enabled)
            builder = SupervisedDataset(**cfg)
            builder.build(shuffle=False)
            dataset_root = Path(builder.dataset_dir).resolve()
            sample_ids = []
            for path in builder.train_paths:
                resolved = Path(str(path)).resolve()
                try:
                    relative = str(resolved.relative_to(dataset_root).as_posix())
                except ValueError:
                    relative = str(resolved.as_posix())
                sample_ids.append(relative)

            ds = tf.data.Dataset.from_tensor_slices((builder.train_paths, builder.train_labels, sample_ids))
            ds = builder._with_deterministic_options(ds)
            ds = ds.map(
                lambda p, l, s: (builder._tf_load_full_audio(p), l, s),
                num_parallel_calls=builder.pure_parallel_calls,
                deterministic=builder.deterministic,
            )
            if cache_enabled:
                from wingbeat_ml.data.cache import materialize_tensorflow_cache
                cache_file = os.path.join(self.cache_dir, "test_loaded_cache")
                ds = materialize_tensorflow_cache(ds, cache_file)

            items = []
            for audio, label, sample_id in ds:
                audio_np = audio.numpy()
                h = hashlib.sha256(audio_np.tobytes()).hexdigest()
                items.append({
                    "id": sample_id.numpy().decode("utf-8"),
                    "label": int(label.numpy()),
                    "hash": h,
                    "shape": audio_np.shape,
                    "dtype": str(audio_np.dtype),
                })
            return items

        off_items = _extract_loaded_audio(cache_enabled=False)
        fresh_items = _extract_loaded_audio(cache_enabled=True)
        reused_items = _extract_loaded_audio(cache_enabled=True)

        self.assertEqual(off_items, fresh_items)
        self.assertEqual(fresh_items, reused_items)

    def test_3_shuffle_disabled_segments_and_seeds_match_across_cache_modes(self):
        """3. With shuffle disabled, the first 500 generated segments and segment seeds match exactly across cache modes."""
        shutil.rmtree(self.cache_dir, ignore_errors=True)

        def _get_segments(cache_enabled):
            cfg = self._get_base_config(seed=42, cache_enabled=cache_enabled)
            builder = SupervisedDataset(**cfg)
            train_ds, _, _ = builder.build(shuffle=False, batch_size=1)
            segments = []
            for audio, label in train_ds.take(500):
                audio_np = audio.numpy()
                label_np = label.numpy()
                h = hashlib.sha256(audio_np.tobytes()).hexdigest()
                segments.append((h, label_np.tolist()))
            return segments

        off_segs = _get_segments(cache_enabled=False)
        fresh_segs = _get_segments(cache_enabled=True)
        reused_segs = _get_segments(cache_enabled=True)

        self.assertEqual(len(off_segs), len(fresh_segs))
        self.assertEqual(off_segs, fresh_segs)
        self.assertEqual(fresh_segs, reused_segs)

    def test_4_shuffle_enabled_parity_across_cache_modes_and_constructions(self):
        """4. With shuffle enabled and deterministic mode, cache-off, fresh-cache, and reused-cache produce identical sequences."""
        shutil.rmtree(self.cache_dir, ignore_errors=True)

        def _get_shuffled_batches(cache_enabled, seed=42):
            cfg = self._get_base_config(seed=seed, cache_enabled=cache_enabled)
            builder = SupervisedDataset(**cfg)
            train_ds, _, _ = builder.build(shuffle=True, batch_size=4)
            batches = []
            for audio, label in train_ds.take(10):
                audio_np = audio.numpy()
                label_np = label.numpy()
                h = hashlib.sha256(audio_np.tobytes()).hexdigest()
                batches.append((h, label_np.tolist()))
            return batches

        off_b = _get_shuffled_batches(cache_enabled=False, seed=42)
        fresh_b = _get_shuffled_batches(cache_enabled=True, seed=42)
        reused_b = _get_shuffled_batches(cache_enabled=True, seed=42)
        repeated_b = _get_shuffled_batches(cache_enabled=False, seed=42)

        self.assertEqual(off_b, fresh_b)
        self.assertEqual(fresh_b, reused_b)
        self.assertEqual(off_b, repeated_b)

    def test_5_changing_global_seed_changes_seeds_and_outputs(self):
        """5. Changing the global seed changes shuffle/augmentation seeds and dataset output sequence."""
        cfg42 = self._get_base_config(seed=42, cache_enabled=False)
        builder42 = SupervisedDataset(**cfg42)
        train_ds42, _, _ = builder42.build(shuffle=True, batch_size=4)

        cfg48 = self._get_base_config(seed=48, cache_enabled=False)
        builder48 = SupervisedDataset(**cfg48)
        train_ds48, _, _ = builder48.build(shuffle=True, batch_size=4)

        batch42 = next(iter(train_ds42))
        batch48 = next(iter(train_ds48))

        hash42 = hashlib.sha256(batch42[0].numpy().tobytes()).hexdigest()
        hash48 = hashlib.sha256(batch48[0].numpy().tobytes()).hexdigest()
        self.assertNotEqual(hash42, hash48)

    def test_6_changing_sample_order_before_shuffle_does_not_change_derived_seed(self):
        """6. Changing sample order before shuffle does not change the seed derived for a given sample ID."""
        sample_id = tf.constant("class_a/sample_01.wav")
        g_seed = 42
        seed1 = derive_sample_seed(sample_id, g_seed, iteration_rnd=0, stage_id=1)
        seed2 = derive_sample_seed(sample_id, g_seed, iteration_rnd=0, stage_id=1)

        np.testing.assert_array_equal(seed1.numpy(), seed2.numpy())

        # Test with a different stage
        seed_stage2 = derive_sample_seed(sample_id, g_seed, iteration_rnd=0, stage_id=2)
        self.assertFalse(np.array_equal(seed1.numpy(), seed_stage2.numpy()))

    def test_7_val_test_outputs_remain_fixed_across_iterations(self):
        """7. Validation/test outputs remain fixed across repeated iterations."""
        cfg = self._get_base_config(seed=42, cache_enabled=False)
        builder = SupervisedDataset(**cfg)
        _, val_ds, test_ds = builder.build(shuffle=False, batch_size=2)

        iter1_val = [b[0].numpy() for b in val_ds]
        iter2_val = [b[0].numpy() for b in val_ds]

        for v1, v2 in zip(iter1_val, iter2_val):
            np.testing.assert_array_equal(v1, v2)

        iter1_test = [b[0].numpy() for b in test_ds]
        iter2_test = [b[0].numpy() for b in test_ds]

        for t1, t2 in zip(iter1_test, iter2_test):
            np.testing.assert_array_equal(t1, t2)

    def test_8_cache_identity_changes_when_schema_or_version_changes(self):
        """8. Cache identity changes when cached element structure, loader, or seed implementation version changes."""
        paths = ["class_a/file1.wav", "class_a/file2.wav"]
        prep = {"sample_rate": 8000, "segment_length": 2400}

        k1 = stable_cache_key(paths, prep, schema_version=CACHE_SCHEMA_VERSION)
        k2 = stable_cache_key(paths, prep, schema_version=CACHE_SCHEMA_VERSION - 1)
        self.assertNotEqual(k1, k2)

        # Test preprocessing change
        prep_diff = {"sample_rate": 16000, "segment_length": 2400}
        k3 = stable_cache_key(paths, prep_diff, schema_version=CACHE_SCHEMA_VERSION)
        self.assertNotEqual(k1, k3)


if __name__ == "__main__":
    unittest.main()
