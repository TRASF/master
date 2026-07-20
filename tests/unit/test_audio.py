"""Unit tests for wingbeat_ml.data.audio.

Tests cover:
  - WAV loading to mono float32
  - NPY loading
  - Dtype conversion (always float32 output)
  - to_mono() channel conversion
  - resample_audio() via librosa
  - load_audio() end-to-end
  - segment_audio() shape, overlap, padding, max_segments
  - preprocess_audio() pass-through and resampling
  - Empty / invalid audio handling
  - Determinism (same call → same result)
"""

import io
import struct
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from wingbeat_ml.data.audio import (
    load_audio,
    load_raw,
    preprocess_audio,
    resample_audio,
    segment_audio,
    to_mono,
)


# ---------------------------------------------------------------------------
# Helpers to create synthetic WAV files
# ---------------------------------------------------------------------------

def _make_wav(path: str, nframes: int = 4800, sr: int = 8000, nchannels: int = 1, sampwidth: int = 2) -> None:
    """Write a short sine-wave WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        t = np.linspace(0, nframes / sr, nframes, endpoint=False)
        signal = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        if nchannels > 1:
            signal = np.stack([signal] * nchannels, axis=1)
        wf.writeframes(signal.tobytes())


def _make_stereo_wav(path: str, nframes: int = 4800, sr: int = 8000) -> None:
    _make_wav(path, nframes=nframes, sr=sr, nchannels=2)


def _make_npy(path: str, length: int = 2400) -> np.ndarray:
    data = np.random.default_rng(0).uniform(-0.5, 0.5, length).astype(np.float32)
    np.save(path, data)
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadRaw(unittest.TestCase):
    def test_returns_float32(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        _make_wav(path)
        data, sr, subtype = load_raw(path)
        self.assertEqual(data.dtype, np.float32)
        self.assertEqual(sr, 8000)
        self.assertIsInstance(subtype, str)

    def test_raises_if_missing(self):
        with self.assertRaises(FileNotFoundError):
            load_raw("/tmp/nonexistent_audio_file_zzz.wav")


class TestToMono(unittest.TestCase):
    def test_mono_unchanged(self):
        arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        out = to_mono(arr)
        np.testing.assert_array_equal(out, arr)
        self.assertEqual(out.dtype, np.float32)

    def test_stereo_averaged(self):
        arr = np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        out = to_mono(arr)
        np.testing.assert_allclose(out, [0.5, 0.5], rtol=1e-6)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.ndim, 1)

    def test_output_is_float32(self):
        arr = np.array([1, 2, 3], dtype=np.int16)
        out = to_mono(arr)
        self.assertEqual(out.dtype, np.float32)


class TestResampleAudio(unittest.TestCase):
    def test_no_resample_when_same_rate(self):
        arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        out = resample_audio(arr, 8000, 8000)
        np.testing.assert_array_equal(out, arr)
        self.assertEqual(out.dtype, np.float32)

    def test_upsample_changes_length(self):
        arr = np.ones(8000, dtype=np.float32)
        out = resample_audio(arr, 8000, 16000)
        self.assertEqual(len(out), 16000)
        self.assertEqual(out.dtype, np.float32)

    def test_downsample_changes_length(self):
        arr = np.ones(16000, dtype=np.float32)
        out = resample_audio(arr, 16000, 8000)
        self.assertEqual(len(out), 8000)
        self.assertEqual(out.dtype, np.float32)

    def test_deterministic(self):
        arr = np.sin(np.linspace(0, np.pi, 16000)).astype(np.float32)
        out1 = resample_audio(arr.copy(), 16000, 8000)
        out2 = resample_audio(arr.copy(), 16000, 8000)
        np.testing.assert_array_equal(out1, out2)


class TestLoadAudio(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.wav_path = str(Path(self.tmp.name) / "test.wav")
        self.npy_path = str(Path(self.tmp.name) / "test.npy")
        _make_wav(self.wav_path, nframes=4800, sr=8000)
        self.npy_data = _make_npy(self.npy_path, length=2400)

    def tearDown(self):
        self.tmp.cleanup()

    def test_wav_output_dtype(self):
        out = load_audio(self.wav_path)
        self.assertEqual(out.dtype, np.float32)

    def test_wav_output_is_1d(self):
        out = load_audio(self.wav_path)
        self.assertEqual(out.ndim, 1)

    def test_wav_mono_length(self):
        out = load_audio(self.wav_path, target_sample_rate=8000)
        # 4800 frames at 8000 Hz
        self.assertEqual(len(out), 4800)

    def test_npy_loading(self):
        out = load_audio(self.npy_path)
        np.testing.assert_array_equal(out, self.npy_data)
        self.assertEqual(out.dtype, np.float32)

    def test_stereo_wav_becomes_mono(self):
        stereo_path = str(Path(self.tmp.name) / "stereo.wav")
        _make_stereo_wav(stereo_path)
        out = load_audio(stereo_path)
        self.assertEqual(out.ndim, 1)

    def test_deterministic(self):
        out1 = load_audio(self.wav_path)
        out2 = load_audio(self.wav_path)
        np.testing.assert_array_equal(out1, out2)

    def test_resamples_from_16khz(self):
        wav16k = str(Path(self.tmp.name) / "test16k.wav")
        _make_wav(wav16k, nframes=16000, sr=16000)
        out = load_audio(wav16k, target_sample_rate=8000)
        self.assertEqual(len(out), 8000)
        self.assertEqual(out.dtype, np.float32)


class TestSegmentAudio(unittest.TestCase):
    def test_basic_shape(self):
        wav = np.ones(9600, dtype=np.float32)
        segs = segment_audio(wav, segment_length=2400)
        self.assertEqual(segs.shape[1], 2400)
        self.assertEqual(segs.dtype, np.float32)
        # 9600 / 2400 = 4 segments
        self.assertEqual(segs.shape[0], 4)

    def test_short_audio_padded(self):
        wav = np.ones(1000, dtype=np.float32)
        segs = segment_audio(wav, segment_length=2400)
        self.assertEqual(segs.shape, (1, 2400))
        # Tail should be zero-padded
        np.testing.assert_array_equal(segs[0, 1000:], np.zeros(1400))

    def test_empty_audio(self):
        wav = np.array([], dtype=np.float32)
        segs = segment_audio(wav, segment_length=2400)
        self.assertEqual(segs.shape, (0, 2400))

    def test_overlap(self):
        wav = np.ones(4800, dtype=np.float32)
        segs = segment_audio(wav, segment_length=2400, overlap=0.5)
        # step = 1200; starts at 0, 1200, 2400, 3600 → 4 segments (last padded)
        self.assertGreater(segs.shape[0], 2)

    def test_max_segments(self):
        wav = np.ones(24000, dtype=np.float32)
        segs = segment_audio(wav, segment_length=2400, max_segments=3)
        self.assertEqual(segs.shape[0], 3)

    def test_output_dtype(self):
        wav = np.ones(4800, dtype=np.float64)
        segs = segment_audio(wav.astype(np.float32), segment_length=2400)
        self.assertEqual(segs.dtype, np.float32)

    def test_no_overlap_deterministic(self):
        wav = np.arange(4800, dtype=np.float32)
        s1 = segment_audio(wav, segment_length=2400, overlap=0.0)
        s2 = segment_audio(wav, segment_length=2400, overlap=0.0)
        np.testing.assert_array_equal(s1, s2)


class TestPreprocessAudio(unittest.TestCase):
    def test_passthrough_same_rate(self):
        wav = np.ones(4800, dtype=np.float32)
        out = preprocess_audio(wav, source_rate=8000, config={"sample_rate": 8000})
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(len(out), 4800)

    def test_resampling(self):
        wav = np.ones(16000, dtype=np.float32)
        out = preprocess_audio(wav, source_rate=16000, config={"sample_rate": 8000})
        self.assertEqual(len(out), 8000)

    def test_default_config(self):
        wav = np.ones(4800, dtype=np.float32)
        out = preprocess_audio(wav, source_rate=8000)
        self.assertEqual(len(out), 4800)
        self.assertEqual(out.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
