"""Unit tests for model analysis, visualizer, spectrogram, and Grad-CAM features."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np

from wingbeat_ml.visualizer.spectrogram import compute_spectrogram, analyze_harmonics
from wingbeat_ml.visualizer.exporter import export_anomaly_frame
from wingbeat_ml.visualizer.analyzer import HostAnalyzer, AnalysisResult

try:
    import tensorflow as tf
    from wingbeat_ml.evaluation.gradcam import compute_gradcam
except ImportError:
    tf = None
    compute_gradcam = None


class TestSpectrogramAnalysis(unittest.TestCase):
    def test_compute_spectrogram_dimensions(self):
        audio = np.random.randn(2400).astype(np.float32)
        freqs, times, spec = compute_spectrogram(audio, sample_rate=8000, n_fft=512, hop_length=128)
        self.assertGreater(len(freqs), 0)
        self.assertGreater(len(times), 0)
        self.assertEqual(spec.shape[0], len(freqs))
        self.assertEqual(spec.shape[1], len(times))

    def test_analyze_harmonics_peak_detection(self):
        sr = 8000
        t = np.linspace(0, 0.3, int(sr * 0.3), endpoint=False)
        # Synthetic 450 Hz mosquito wingbeat tone
        audio = (0.8 * np.sin(2 * np.pi * 450.0 * t)).astype(np.float32)

        res = analyze_harmonics(audio, sample_rate=sr, freq_range=(150.0, 1000.0))
        self.assertAlmostEqual(res["f0_hz"], 450.0, delta=25.0)
        self.assertGreater(res["peak_power_db"], -40.0)


class TestExporter(unittest.TestCase):
    def test_export_anomaly_frame(self):
        audio = np.random.randn(2400).astype(np.float32)
        meta = {"seq": 42, "mcu_class_id": 0, "mcu_confidence": 0.30, "discrepancy": True}

        with tempfile.TemporaryDirectory() as tmpdir:
            frame_dir = export_anomaly_frame(audio, 8000, meta, output_dir=tmpdir)
            self.assertTrue(frame_dir.exists())
            self.assertTrue((frame_dir / "audio.wav").exists())
            self.assertTrue((frame_dir / "metadata.json").exists())


class TestGradCAM(unittest.TestCase):
    def test_gradcam_computation(self):
        if tf is None:
            self.skipTest("TensorFlow not installed")

        # Construct minimal Keras Conv1D model
        inp = tf.keras.layers.Input(shape=(2400, 1))
        x = tf.keras.layers.Conv1D(filters=8, kernel_size=7, activation="relu", name="conv1d_target")(inp)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        out = tf.keras.layers.Dense(11, activation="softmax")(x)
        model = tf.keras.Model(inputs=inp, outputs=out)

        dummy_audio = np.random.randn(1, 2400, 1).astype(np.float32)
        heatmap, cls_idx, conf = compute_gradcam(model, dummy_audio)

        self.assertIsInstance(heatmap, np.ndarray)
        self.assertGreaterEqual(cls_idx, 0)
        self.assertLess(cls_idx, 11)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)


class TestHostAnalyzer(unittest.TestCase):
    def test_analyzer_worker_thread(self):
        analyzer = HostAnalyzer(model_path=None, sample_rate=8000, enable_gradcam=False, export_anomalies=False)
        analyzer.start()

        dummy_i16 = (np.random.randn(2400) * 1000).astype(np.int16)
        analyzer.submit_packet(seq=1, mcu_class_id=0, mcu_confidence=0.8, audio_i16=dummy_i16, received_at=100.0)

        # Retrieve output result
        res = analyzer.output_queue.get(timeout=2.0)
        self.assertIsInstance(res, AnalysisResult)
        self.assertEqual(res.seq, 1)
        self.assertGreaterEqual(res.f0_hz, 0.0)

        analyzer.stop_event.set()
        analyzer.join(timeout=1.0)

    def test_analyzer_weights_h5_loading(self):
        if tf is None:
            self.skipTest("TensorFlow not installed")

        from wingbeat_ml.models import MosSongPlusModel
        import yaml

        with open("configs/models/mossong_plus.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        m = MosSongPlusModel(cfg).build((2400, 1), 11, "softmax")
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_file = str(Path(tmpdir) / "best_model.weights.h5")
            m.save_weights(weights_file)

            analyzer = HostAnalyzer(model_path=weights_file, sample_rate=8000, enable_gradcam=True)
            self.assertIsNotNone(analyzer.model)

    def test_analyzer_fast_tflite_loading(self):
        if tf is None:
            self.skipTest("TensorFlow not installed")

        from wingbeat_ml.export.tflite import convert_float_tflite
        from wingbeat_ml.visualizer.analyzer import FastTFLiteModel

        m = make_model() if "make_model" in globals() else None
        if m is None:
            inp = tf.keras.layers.Input(batch_shape=(1, 2400, 1))
            out = tf.keras.layers.Dense(11)(tf.keras.layers.GlobalAveragePooling1D()(inp))
            m = tf.keras.Model(inp, out)

        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = str(Path(tmpdir) / "model.tflite")
            convert_float_tflite(m, tflite_path)

            fast_model = FastTFLiteModel(tflite_path)
            cls_id, conf = fast_model.predict_fast(np.random.randn(2400).astype(np.float32))
            self.assertGreaterEqual(cls_id, 0)
            self.assertLess(cls_id, 11)


class TestModelDiagnostics(unittest.TestCase):
    def test_analyze_model_sample(self):
        if tf is None:
            self.skipTest("TensorFlow not installed")

        from wingbeat_ml.evaluation.diagnostics import analyze_model_sample, DiagnosticResult

        inp = tf.keras.layers.Input(shape=(2400, 1))
        x = tf.keras.layers.Conv1D(filters=8, kernel_size=7, activation="relu", name="conv1d_target")(inp)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        emb = tf.keras.layers.Dense(16, activation="relu", name="dense_emb")(x)
        out = tf.keras.layers.Dense(11, activation="softmax", name="dense_out")(emb)
        model = tf.keras.Model(inputs=inp, outputs=out)

        dummy_audio = np.random.randn(2400).astype(np.float32)
        res = analyze_model_sample(model, dummy_audio)

        self.assertIsInstance(res, DiagnosticResult)
        self.assertEqual(len(res.dense_embedding), 16)
        self.assertEqual(res.class_contributions.shape, (16, 11))
        self.assertEqual(len(res.top_positive_features), 5)


if __name__ == "__main__":
    unittest.main()
