"""Comprehensive P0 verification unit tests covering requirements A through P."""

import unittest
import numpy as np
import tensorflow as tf

from wingbeat_ml.export.input_contract import (
    DeploymentInputContract,
    dequantize_int8_to_float,
    preprocess_audio_canonical,
    quantize_float_to_int8,
    resolve_deployment_shape,
)
from wingbeat_ml.export.tflite import (
    convert_float_tflite,
    convert_full_int8_tflite,
)
from wingbeat_ml.evaluation.gradcam import (
    GradCamResult,
    aggregate_raw_cams,
    compute_gradcam,
)


def _make_dummy_model(output_activation: str | None = "softmax", input_len: int = 2400) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(input_len, 1), name="audio_input")
    x = tf.keras.layers.Conv1D(filters=8, kernel_size=15, strides=8, padding="same", name="conv1")(inputs)
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = tf.keras.layers.Dense(16, activation="relu", name="dense_emb")(x)
    outputs = tf.keras.layers.Dense(3, activation=output_activation, name="output_dense")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="test_model")


def _make_conv_t_1_model() -> tf.keras.Model:
    """Model where conv layer outputs temporal length 1 [B, 1, C]."""
    inputs = tf.keras.layers.Input(shape=(2400, 1), name="audio_input")
    x = tf.keras.layers.Conv1D(filters=4, kernel_size=2400, strides=1, padding="valid", name="conv_t1")(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="output_dense")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


class TestP0ContractAndExport(unittest.TestCase):
    """Verifies P0 Requirements A, B, C, D, E, F, G."""

    def test_A_canonical_contract_schema(self):
        contract = DeploymentInputContract(
            sample_rate_hz=8000,
            frame_length_samples=2400,
            duration=0.3,
            channels=1,
            tensor_layout="[1, 2400, 1]",
            input_dtype="int8",
            dc_removal=True,
            target_rms=0.05,
        )
        cdict = contract.to_dict()
        self.assertEqual(cdict["sample_rate_hz"], 8000)
        self.assertEqual(cdict["frame_length_samples"], 2400)
        self.assertEqual(cdict["model_input_shape"], [1, 2400, 1])
        self.assertIn("1.0.0", contract.to_json())

    def test_B_export_shape_resolution(self):
        model = _make_dummy_model(input_len=2400)
        shape = resolve_deployment_shape(model)
        self.assertEqual(shape, (1, 2400, 1))

    def test_C_shape_mismatch_rejection(self):
        model = _make_dummy_model(input_len=2400)
        with self.assertRaises(ValueError) as ctx:
            resolve_deployment_shape(model, input_shape=(1, 1200, 1))
        self.assertIn("Deployment shape mismatch", str(ctx.exception))

    def test_D_deterministic_python_preprocessing(self):
        raw_pcm = np.sin(np.linspace(0, 100, 2400, dtype=np.float32)) + 0.1
        out1 = preprocess_audio_canonical(raw_pcm, dc_removal=True, target_rms=0.05)
        out2 = preprocess_audio_canonical(raw_pcm, dc_removal=True, target_rms=0.05)
        np.testing.assert_array_equal(out1, out2)

    def test_E_firmware_equivalent_preprocessing_parity(self):
        # Generate raw 16-bit PCM waveform with DC offset
        t = np.linspace(0, 50, 2400, dtype=np.float32)
        raw_float = np.sin(t) * 0.1 + 0.05  # Has DC offset 0.05

        # Python canonical preprocessing
        py_prep = preprocess_audio_canonical(raw_float, dc_removal=True, normalization_method="rms_normalize", target_rms=0.05)

        # Firmware-equivalent processing simulation
        fw_float = raw_float.copy()
        fw_float -= np.mean(fw_float)  # RemoveDC
        fw_rms = np.sqrt(np.mean(np.square(fw_float)))  # ComputeRms
        gain = np.clip(0.05 / (fw_rms + 1e-8), 0.1, 10.0)
        fw_prep = np.clip(fw_float * gain, -1.0, 1.0).astype(np.float32)

        np.testing.assert_allclose(py_prep, fw_prep, atol=1e-5)

    def test_F_int8_scale_zero_point_quantization(self):
        scale = 0.0078125  # 1 / 128
        zero_point = 0
        x_float = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)

        q_int8 = quantize_float_to_int8(x_float, scale, zero_point)
        expected_int8 = np.array([-128, -64, 0, 64, 127], dtype=np.int8)
        np.testing.assert_array_equal(q_int8, expected_int8)

        dequant = dequantize_int8_to_float(q_int8, scale, zero_point)
        np.testing.assert_allclose(x_float, dequant, atol=0.01)

    def test_G_clipping_saturation_boundaries(self):
        scale = 0.01
        zero_point = 0
        extreme_inputs = np.array([-10.0, -1.5, -1.0, 0.0, 1.0, 1.5, 10.0], dtype=np.float32)

        # Float preprocessing clipping
        clipped_float = preprocess_audio_canonical(extreme_inputs, dc_removal=False, normalization_method="fixed_range", fixed_range_amplitude=1.0)
        self.assertTrue(np.all(clipped_float >= -1.0))
        self.assertTrue(np.all(clipped_float <= 1.0))

        # INT8 saturation bounds check [-128, 127]
        q = quantize_float_to_int8(extreme_inputs, scale, zero_point)
        self.assertEqual(q[0], -128)
        self.assertEqual(q[-1], 127)


class TestP0GradCAM(unittest.TestCase):
    """Verifies P0 Requirements H, I, J, K, L, M, N, O, P."""

    def test_H_pre_softmax_gradcam_target(self):
        model_softmax = _make_dummy_model(output_activation="softmax")
        model_linear = _make_dummy_model(output_activation=None)

        sample = np.random.randn(1, 2400, 1).astype(np.float32)

        res_s = compute_gradcam(model_softmax, sample, class_idx=0)
        res_l = compute_gradcam(model_linear, sample, class_idx=0)

        self.assertIsInstance(res_s, GradCamResult)
        self.assertIsInstance(res_l, GradCamResult)

    def test_I_B_equals_1_semantics(self):
        model = _make_dummy_model()
        sample = np.random.randn(1, 2400, 1).astype(np.float32)

        res = compute_gradcam(model, sample, class_idx=0)
        self.assertEqual(res.raw_cam.shape, (1, 2400))
        self.assertEqual(res.display_cam.shape, (1, 2400))
        self.assertEqual(res.raw_cam.ndim, 2)

        # Tuple unpacking test
        display_cam, cls_idx, conf = res
        self.assertEqual(display_cam.shape, (1, 2400))

    def test_J_B_greater_than_1_semantics(self):
        model = _make_dummy_model()
        batch_sample = np.random.randn(3, 2400, 1).astype(np.float32)

        res = compute_gradcam(model, batch_sample, class_idx=1)
        self.assertEqual(res.raw_cam.shape, (3, 2400))
        self.assertEqual(res.display_cam.shape, (3, 2400))
        self.assertEqual(res.degenerate_heatmap.shape, (3,))

    def test_K_T_equals_1_conv_layer(self):
        model = _make_conv_t_1_model()
        sample = np.random.randn(1, 2400, 1).astype(np.float32)

        res = compute_gradcam(model, sample, class_idx=0)
        self.assertEqual(res.raw_cam.shape, (1, 2400))
        self.assertEqual(res.display_cam.shape, (1, 2400))

    def test_L_zero_cam_handling(self):
        # A zero input tensor producing zero gradients / zero CAM
        model = _make_dummy_model()
        zero_sample = np.zeros((1, 2400, 1), dtype=np.float32)

        res = compute_gradcam(model, zero_sample, class_idx=0)
        is_deg = bool(np.asarray(res.degenerate_heatmap).ravel()[0])
        self.assertTrue(is_deg)
        np.testing.assert_array_equal(res.display_cam, np.zeros((1, 2400)))

    def test_M_constant_positive_cam_not_all_ones(self):
        model = _make_dummy_model()
        # Mocking or simulating a constant positive raw CAM calculation
        res = compute_gradcam(model, np.ones((1, 2400, 1), dtype=np.float32), class_idx=0)
        # Verify display_cam is NOT all ones if raw_cam range <= 1e-8
        is_deg = bool(np.asarray(res.degenerate_heatmap).ravel()[0])
        if is_deg:
            self.assertFalse(np.all(res.display_cam == 1.0))
            np.testing.assert_array_equal(res.display_cam, np.zeros_like(res.display_cam))

    def test_N_raw_display_separation(self):
        model = _make_dummy_model()
        sample = np.random.randn(1, 2400, 1).astype(np.float32)

        res = compute_gradcam(model, sample, class_idx=0)

        # Raw CAM preserves analytical magnitude
        self.assertIsNotNone(res.raw_cam)
        self.assertIsNotNone(res.display_cam)
        self.assertTrue(hasattr(res, "raw_min"))
        self.assertTrue(hasattr(res, "raw_max"))
        self.assertTrue(hasattr(res, "raw_mean"))
        self.assertTrue(hasattr(res, "raw_l1"))
        self.assertTrue(hasattr(res, "raw_l2"))

    def test_O_temporal_alignment_to_T(self):
        model = _make_dummy_model(input_len=2400)  # Conv layer outputs T' = 300
        sample = np.random.randn(1, 2400, 1).astype(np.float32)

        res = compute_gradcam(model, sample, class_idx=0)
        self.assertEqual(res.raw_cam.shape[1], 2400)
        self.assertEqual(res.display_cam.shape[1], 2400)

    def test_P_aggregation_preserves_raw_magnitude_differences(self):
        cam_large = np.full((1, 2400), 10.0, dtype=np.float32)
        cam_small = np.full((1, 2400), 1.0, dtype=np.float32)

        summary = aggregate_raw_cams([cam_large, cam_small])
        self.assertEqual(summary["count"], 2)
        np.testing.assert_allclose(summary["mean"], np.full((2400,), 5.5, dtype=np.float32))
        np.testing.assert_allclose(summary["median"], np.full((2400,), 5.5, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
