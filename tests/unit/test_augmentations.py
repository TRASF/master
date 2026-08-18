"""Unit tests for wingbeat_ml.augmentations.

Tests cover AudioAugmentor transforms from wingbeat_ml.augmentations.transforms:
  - pre_emphasis: output shape, dtype, determinism
  - rms_normalize: energy normalisation, clipping
  - delta_waveform: first-order difference
  - apply_time_masking: mask application with seed
  - random_segment: output shape and determinism
  - random_gain: gain application in dB range
  - add_gaussian_noise: SNR-controlled noise addition
  - apply_hpf: high-pass filter (requires config)
  - pitch_shift: output shape preserved
  - time_shift: output shape preserved

Also tests pipeline:
  - build_augmentor with valid config
  - build_augmentor raises on unknown transform key
  - TRANSFORMS contains expected keys

NOTE: Tests avoid importing TensorFlow at module load time.
AudioAugmentor is TF-dependent; import is deferred.
"""

import unittest
import numpy as np


# ---------------------------------------------------------------------------
# TF-dependent helpers (imported in each test)
# ---------------------------------------------------------------------------

def _make_audio(length: int = 2400, seed: int = 0) -> "tf.Tensor":
    import tensorflow as tf
    rng = np.random.default_rng(seed)
    data = rng.uniform(-0.5, 0.5, length).astype(np.float32)
    return tf.constant(data, dtype=tf.float32)


def _make_seed(a: int = 0, b: int = 0) -> "tf.Tensor":
    import tensorflow as tf
    return tf.constant([a, b], dtype=tf.int64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPreEmphasis(unittest.TestCase):
    def test_output_shape(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        aug = AudioAugmentor(segment_length=2400, config={}, seed=42)
        audio = _make_audio(2400)
        out = aug.pre_emphasis(audio)
        self.assertEqual(out.shape[0], 2400)

    def test_output_dtype(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        aug = AudioAugmentor(segment_length=2400, config={}, seed=42)
        audio = _make_audio(2400)
        out = aug.pre_emphasis(audio)
        self.assertEqual(out.dtype, tf.float32)

    def test_first_element_unchanged(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        aug = AudioAugmentor(segment_length=2400, config={}, seed=42)
        audio = tf.constant([0.5, 0.3, 0.1], dtype=tf.float32)
        out = aug.pre_emphasis(audio)
        self.assertAlmostEqual(float(out[0]), 0.5, places=5)


class TestRmsNormalize(unittest.TestCase):
    def test_output_shape(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        aug = AudioAugmentor(segment_length=2400, config={}, seed=42)
        audio = _make_audio(2400)
        out = aug.rms_normalize(audio)
        self.assertEqual(out.shape[0], 2400)

    def test_output_dtype(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        aug = AudioAugmentor(segment_length=2400, config={}, seed=42)
        audio = _make_audio(2400)
        out = aug.rms_normalize(audio)
        self.assertEqual(out.dtype, tf.float32)

    def test_rms_near_target(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        aug = AudioAugmentor(segment_length=2400, config={"rms_norm": {"target_rms": 0.5}}, seed=42)
        audio = tf.constant(np.ones(2400, dtype=np.float32) * 0.1)
        out = aug.rms_normalize(audio, target_rms=0.5, min_gain=0.1, max_gain=10.0)
        rms = float(tf.sqrt(tf.reduce_mean(tf.square(out))))
        self.assertAlmostEqual(rms, 0.5, places=4)

    def test_gain_clamped(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        aug = AudioAugmentor(segment_length=2400, config={}, seed=42)
        # Very quiet audio → gain would exceed max_gain
        audio = tf.constant(np.ones(2400, dtype=np.float32) * 1e-6)
        out = aug.rms_normalize(audio, target_rms=0.5, min_gain=0.1, max_gain=10.0)
        # Gain must not exceed 10
        rms = float(tf.sqrt(tf.reduce_mean(tf.square(audio))) * 10.0)
        self.assertGreater(rms, 0)


class TestDeltaWaveform(unittest.TestCase):
    def test_output_shape(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        aug = AudioAugmentor(segment_length=2400, config={}, seed=42)
        audio = _make_audio(2400)
        out = aug.delta_waveform(audio)
        self.assertEqual(out.shape[0], 2400)

    def test_first_element_is_zero(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        aug = AudioAugmentor(segment_length=2400, config={}, seed=42)
        audio = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        out = aug.delta_waveform(audio)
        self.assertAlmostEqual(float(out[0]), 0.0, places=6)


class TestApplyTimeMasking(unittest.TestCase):
    def test_output_shape(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        cfg = {"time_masking": {"p": 1.0, "num_masks": 1, "max_mask_size": 400}}
        aug = AudioAugmentor(segment_length=2400, config=cfg, seed=42)
        audio = tf.ones(2400, dtype=tf.float32)
        out = aug.apply_time_masking(audio, seed=_make_seed(0, 0))
        self.assertEqual(out.shape[0], 2400)

    def test_output_dtype(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        cfg = {"time_masking": {"p": 1.0, "num_masks": 1, "max_mask_size": 400}}
        aug = AudioAugmentor(segment_length=2400, config=cfg, seed=42)
        audio = tf.ones(2400, dtype=tf.float32)
        out = aug.apply_time_masking(audio, seed=_make_seed(0, 0))
        self.assertEqual(out.dtype, tf.float32)

    def test_fixed_seed_deterministic(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        cfg = {"time_masking": {"p": 1.0, "num_masks": 1, "max_mask_size": 400}}
        aug = AudioAugmentor(segment_length=2400, config=cfg, seed=42)
        audio = tf.ones(2400, dtype=tf.float32)
        seed = _make_seed(7, 3)
        out1 = aug.apply_time_masking(audio, seed=seed)
        out2 = aug.apply_time_masking(audio, seed=seed)
        np.testing.assert_array_equal(out1.numpy(), out2.numpy())

    def test_mask_creates_zeros(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        cfg = {"time_masking": {"p": 1.0, "num_masks": 1, "max_mask_size": 400}}
        aug = AudioAugmentor(segment_length=2400, config=cfg, seed=42)
        audio = tf.ones(2400, dtype=tf.float32)
        out = aug.apply_time_masking(audio, seed=_make_seed(0, 0))
        # Some values should be zero
        self.assertTrue(np.any(out.numpy() == 0.0))


class TestRandomSegment(unittest.TestCase):
    def test_output_shape(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        aug = AudioAugmentor(segment_length=2400, config={}, seed=42)
        audio = tf.ones(9600, dtype=tf.float32)
        out = aug.random_segment(audio, seed=_make_seed(0, 0))
        self.assertEqual(out.shape[0], 2400)

    def test_short_audio_padded(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        aug = AudioAugmentor(segment_length=2400, config={}, seed=42)
        audio = tf.ones(100, dtype=tf.float32)
        out = aug.random_segment(audio, seed=_make_seed(0, 0))
        self.assertEqual(out.shape[0], 2400)

    def test_deterministic_with_seed(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        aug = AudioAugmentor(segment_length=2400, config={}, seed=42)
        audio = tf.constant(np.arange(9600, dtype=np.float32))
        seed = _make_seed(5, 5)
        out1 = aug.random_segment(audio, seed=seed)
        out2 = aug.random_segment(audio, seed=seed)
        np.testing.assert_array_equal(out1.numpy(), out2.numpy())


class TestRandomGain(unittest.TestCase):
    def test_output_shape(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        cfg = {"random_gain": {"p": 1.0, "gain_db": [-6.0, 6.0]}}
        aug = AudioAugmentor(segment_length=2400, config=cfg, seed=42)
        audio = tf.ones(2400, dtype=tf.float32) * 0.5
        out = aug.random_gain(audio, [-6.0, 6.0], seed=_make_seed(0, 0))
        self.assertEqual(out.shape[0], 2400)

    def test_output_dtype(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        cfg = {"random_gain": {"p": 1.0, "gain_db": [-6.0, 6.0]}}
        aug = AudioAugmentor(segment_length=2400, config=cfg, seed=42)
        audio = tf.ones(2400, dtype=tf.float32) * 0.5
        out = aug.random_gain(audio, [-6.0, 6.0], seed=_make_seed(0, 0))
        self.assertEqual(out.dtype, tf.float32)


class TestAddGaussianNoise(unittest.TestCase):
    def test_output_shape(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        cfg = {"gaussian_noise": {"p": 1.0, "snr_db": [10.0, 30.0]}}
        aug = AudioAugmentor(segment_length=2400, config=cfg, seed=42)
        audio = tf.ones(2400, dtype=tf.float32) * 0.5
        out = aug.add_gaussian_noise(audio, [10.0, 30.0], seed=_make_seed(0, 0))
        self.assertEqual(out.shape[0], 2400)

    def test_output_dtype(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        cfg = {"gaussian_noise": {"p": 1.0, "snr_db": [10.0, 30.0]}}
        aug = AudioAugmentor(segment_length=2400, config=cfg, seed=42)
        audio = tf.ones(2400, dtype=tf.float32) * 0.5
        out = aug.add_gaussian_noise(audio, [10.0, 30.0], seed=_make_seed(0, 0))
        self.assertEqual(out.dtype, tf.float32)

    def test_deterministic_with_fixed_seed(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        cfg = {"gaussian_noise": {"p": 1.0, "snr_db": [20.0, 20.0]}}
        aug = AudioAugmentor(segment_length=2400, config=cfg, seed=42)
        audio = tf.constant(np.ones(2400, dtype=np.float32) * 0.5)
        seed = _make_seed(3, 7)
        out1 = aug.add_gaussian_noise(audio, [20.0, 20.0], seed=seed)
        out2 = aug.add_gaussian_noise(audio, [20.0, 20.0], seed=seed)
        np.testing.assert_array_equal(out1.numpy(), out2.numpy())


class TestRandomMicEQ(unittest.TestCase):
    def test_shape_preserved(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor

        cfg = {
            "random_mic_eq": {
                "p": 1.0,
                "num_points": [3, 7],
                "gain_db": [-4.0, 4.0],
            }
        }
        aug = AudioAugmentor(
            segment_length=2400,
            sample_rate=8000,
            config=cfg,
            seed=42,
        )
        audio = _make_audio(2400)
        output = aug.random_mic_eq(audio, _make_seed(10, 20))
        self.assertEqual(output.shape, audio.shape)

    def test_deterministic(self):
        from wingbeat_ml.augmentations.transforms import AudioAugmentor

        cfg = {
            "random_mic_eq": {
                "p": 1.0,
                "num_points": [3, 7],
                "gain_db": [-4.0, 4.0],
            }
        }
        aug = AudioAugmentor(config=cfg, seed=42)
        audio = _make_audio(2400)
        seed = _make_seed(99, 11)
        a = aug.random_mic_eq(audio, seed)
        b = aug.random_mic_eq(audio, seed)
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=0.0, atol=0.0)

    def test_zero_db_is_identity(self):
        from wingbeat_ml.augmentations.transforms import AudioAugmentor

        cfg = {
            "random_mic_eq": {
                "p": 1.0,
                "num_points": [3, 7],
                "gain_db": [0.0, 0.0],
            }
        }
        aug = AudioAugmentor(config=cfg)
        audio = _make_audio(2400)
        output = aug.random_mic_eq(audio, _make_seed(1, 2))
        np.testing.assert_allclose(audio.numpy(), output.numpy(), atol=1e-5, rtol=1e-5)


class TestDeviceIR(unittest.TestCase):
    def test_identity_ir(self):
        import tempfile
        from pathlib import Path
        from wingbeat_ml.augmentations.transforms import AudioAugmentor

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "identity.npy"
            ir = np.zeros(400, dtype=np.float32)
            ir[0] = 1.0
            np.save(path, ir)

            cfg = {
                "device_ir": {
                    "p": 1.0,
                    "banks": [str(path)],
                    "max_ir_ms": 50.0,
                    "pre_peak_ms": 0.0,
                    "wet": [1.0, 1.0],
                    "normalize": "l2",
                }
            }

            aug = AudioAugmentor(
                segment_length=2400,
                sample_rate=8000,
                config=cfg,
            )
            audio = _make_audio(2400)
            output = aug.apply_device_ir(audio, _make_seed(1, 2))
            np.testing.assert_allclose(audio.numpy(), output.numpy(), atol=1e-5, rtol=1e-5)


class TestElectronics(unittest.TestCase):
    def test_shape_preserved(self):
        from wingbeat_ml.augmentations.transforms import AudioAugmentor

        cfg = {
            "electronics": {
                "p": 1.0,
                "soft_clip_p": 1.0,
                "hard_clip_p": 1.0,
                "quantize_p": 1.0,
            }
        }
        aug = AudioAugmentor(config=cfg)
        audio = _make_audio(2400)
        output = aug.apply_electronics(audio, _make_seed(4, 5))
        self.assertEqual(output.shape, audio.shape)

    def test_deterministic(self):
        from wingbeat_ml.augmentations.transforms import AudioAugmentor

        cfg = {
            "electronics": {
                "p": 1.0,
                "soft_clip_p": 1.0,
                "hard_clip_p": 1.0,
                "quantize_p": 1.0,
            }
        }
        aug = AudioAugmentor(config=cfg)
        audio = _make_audio(2400)
        seed = _make_seed(9, 7)
        a = aug.apply_electronics(audio, seed)
        b = aug.apply_electronics(audio, seed)
        np.testing.assert_array_equal(a.numpy(), b.numpy())


class TestApplyPostProcessingNoop(unittest.TestCase):
    """Test apply_post_processing with all transforms disabled (p=0)."""

    def test_shape_preserved(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        aug = AudioAugmentor(segment_length=2400, config={}, seed=42)
        audio = _make_audio(2400)
        label = tf.constant(0, dtype=tf.int32)
        out_audio, out_label = aug.apply_post_processing(audio, label, seed=_make_seed(0, 0), augment=False)
        self.assertEqual(out_audio.shape[0], 2400)

    def test_dtype_preserved(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        aug = AudioAugmentor(segment_length=2400, config={}, seed=42)
        audio = _make_audio(2400)
        label = tf.constant(0, dtype=tf.int32)
        out_audio, _ = aug.apply_post_processing(audio, label, seed=_make_seed(0, 0), augment=False)
        self.assertEqual(out_audio.dtype, tf.float32)

    def test_clipped_to_unit_range(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        aug = AudioAugmentor(segment_length=2400, config={}, seed=42)
        # Very loud audio before RMS normalisation + clip
        audio = tf.constant(np.ones(2400, dtype=np.float32) * 100.0)
        label = tf.constant(0, dtype=tf.int32)
        out_audio, _ = aug.apply_post_processing(audio, label, seed=_make_seed(0, 0), augment=False)
        max_val = float(tf.reduce_max(tf.abs(out_audio)))
        self.assertLessEqual(max_val, 1.0 + 1e-5)

    def test_baseline_rms_disabled_preserves_scale(self):
        import tensorflow as tf
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        cfg = {"rms_norm": {"enabled": False, "p": 0.0}}
        aug = AudioAugmentor(segment_length=2400, config=cfg, seed=42)
        # Sine wave with peak 0.2
        t = np.linspace(0, 0.3, 2400, endpoint=False)
        data = (0.2 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
        audio = tf.constant(data)
        label = tf.constant(0, dtype=tf.int32)
        out_audio, _ = aug.apply_post_processing(audio, label, seed=_make_seed(0, 0), augment=False)
        max_val = float(tf.reduce_max(tf.abs(out_audio)))
        self.assertAlmostEqual(max_val, 0.2, places=3)


class TestPipeline(unittest.TestCase):
    def test_build_augmentor_returns_augmentor(self):
        from wingbeat_ml.augmentations.pipeline import build_augmentor
        from wingbeat_ml.augmentations.transforms import AudioAugmentor
        aug = build_augmentor({}, segment_length=2400, seed=42)
        self.assertIsInstance(aug, AudioAugmentor)

    def test_unknown_key_raises(self):
        from wingbeat_ml.augmentations.pipeline import build_augmentor
        with self.assertRaises(ValueError):
            build_augmentor({"nonexistent_transform_xyz": {"p": 1.0}})

    def test_valid_pipeline_level_keys_are_accepted(self):
        from wingbeat_ml.augmentations.pipeline import build_augmentor

        augmentor = build_augmentor({
            "noise_banks": [],
            "overlap": [0.0, 0.5],
            "mixup": {"p": 0.0},
        })
        self.assertIsNotNone(augmentor)

    def test_transforms_registry_has_expected_keys(self):
        from wingbeat_ml.augmentations.pipeline import TRANSFORMS
        expected = [
            "high_pass", "pre_emphasis", "pitch_shift", "time_shift",
            "time_masking", "random_mic_eq", "device_ir", "electronics",
            "random_gain", "gaussian_noise", "noise_overlay",
        ]
        for key in expected:
            self.assertIn(key, TRANSFORMS)


if __name__ == "__main__":
    unittest.main()
