"""Host-side model analysis background executor for visualizer telemetry streams.

Performs host inference, Grad-CAM heatmap extraction, harmonic analysis,
and MCU-Host discrepancy detection.
"""

from __future__ import annotations

import queue
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np

from wingbeat_ml.analysis.signal.spectrum import analyze_harmonics
from wingbeat_ml.visualizer.exporter import export_anomaly_frame

try:
    import tensorflow as tf
except Exception as err:  # pragma: no cover
    tf = None
    TF_IMPORT_ERROR = err
else:
    TF_IMPORT_ERROR = None

try:
    from wingbeat_ml.analysis.model.gradcam import compute_gradcam
except Exception as err:  # pragma: no cover
    compute_gradcam = None
    GRADCAM_IMPORT_ERROR = err
else:
    GRADCAM_IMPORT_ERROR = None

try:
    from wingbeat_ml.classification.evaluation.diagnostics import analyze_model_sample
except Exception as err:  # pragma: no cover
    analyze_model_sample = None
    DIAGNOSTICS_IMPORT_ERROR = err
else:
    DIAGNOSTICS_IMPORT_ERROR = None


from wingbeat_ml.deployment.runtime.tflite import FastTFLiteModel, TFLitePredictor


def _load_or_build_model(model_path: str):
    if tf is None:
        return None

    path_obj = Path(model_path).resolve()
    if not path_obj.exists():
        print(f"[HostAnalyzer Error]: Model file not found at '{model_path}' (Resolved: '{path_obj}')")
        return None

    path = str(path_obj)
    if path.endswith(".tflite"):
        try:
            return FastTFLiteModel(path)
        except Exception as err:
            print(f"[HostAnalyzer Error]: Failed to load TFLite model '{path}': {err}")

    if not path.endswith(".weights.h5"):
        try:
            return tf.keras.models.load_model(path)
        except Exception:
            pass

    try:
        from wingbeat_ml.classification.models import MosSongPlusModel
        import yaml

        cfg_path = Path("configs/models/mossong_plus.yaml")
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                model_cfg = yaml.safe_load(f)
        else:
            model_cfg = {
                "model": {
                    "mossong_plus": {
                        "layers": [
                            {"type": "conv1d", "filters": 32, "kernel_size": 100, "strides": 4, "activation": "relu", "padding": "valid", "batch_norm": True},
                            {"type": "conv1d", "filters": 32, "kernel_size": 64, "strides": 4, "activation": "relu", "padding": "valid", "batch_norm": True},
                            {"type": "conv1d", "filters": 64, "kernel_size": 64, "strides": 3, "activation": "relu", "padding": "valid", "batch_norm": True},
                            {"type": "maxpool1d", "pool_size": 3, "strides": 3, "padding": "valid"},
                            {"type": "flatten"},
                            {"type": "dropout", "rate": 0.5},
                            {"type": "dense", "units": 256, "activation": "relu", "batch_norm": True},
                            {"type": "dropout", "rate": 0.5},
                        ]
                    }
                }
            }

        builder = MosSongPlusModel(model_cfg)
        model = builder.build(input_shape=(2400, 1), output_units=11, output_activation=None)
        model.load_weights(path)
        return model
    except Exception as err:
        print(f"[HostAnalyzer] Failed to load model weights from {path}: {err}")
        return None


@dataclass
class AnalysisResult:
    seq: int
    audio: np.ndarray
    mcu_class_id: int
    mcu_confidence: float
    host_class_id: Optional[int]
    host_confidence: Optional[float]
    discrepancy: bool
    f0_hz: float
    peak_power_db: float
    heatmap: Optional[np.ndarray]
    timestamp: float
    dense_embedding: Optional[np.ndarray] = None
    host_class_probability: Optional[np.ndarray] = None


class HostAnalyzer(threading.Thread):
    """Asynchronous analysis engine processing incoming telemetry frames."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        sample_rate: int = 8000,
        enable_gradcam: bool = False,
        enable_dense: bool = False,
        export_anomalies: bool = False,
        anomaly_output_dir: str = "output/misclassifications",
    ) -> None:
        super().__init__(name="host-analyzer", daemon=True)
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.enable_gradcam = enable_gradcam
        self.export_anomalies = export_anomalies
        self.anomaly_output_dir = anomaly_output_dir

        self.input_queue: queue.Queue[Tuple[int, int, float, np.ndarray, float]] = queue.Queue(maxsize=1)
        self.output_queue: queue.Queue[AnalysisResult] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()

        # Low-latency host-analysis state.
        self.processed_packets = 0
        self.deep_analysis_stride = 3
        self.harmonic_analysis_stride = 2
        self.last_harmonics = {
            "f0_hz": 0.0,
            "peak_power_db": -120.0,
        }
        self._compiled_inference = None

        # WINGBEAT_REALTIME_PATCH
        self.enable_dense = enable_dense
        self.heavy_every = max(
            1, int(os.environ.get("WINGBEAT_HEAVY_EVERY", "7"))
        )
        self.harmonic_every = max(
            1, int(os.environ.get("WINGBEAT_HARMONIC_EVERY", "3"))
        )
        self._analysis_counter = 0
        self._last_f0_hz = 0.0
        self._last_peak_power_db = -120.0
        self._infer_fn = None
        self._heavy_error_reported = False
        self.model = None
        self.model_loaded = False
        self.model_error: Optional[str] = None

        if self.model_path:
            if tf is None:
                self.model_error = (
                    f"TensorFlow import failed: {TF_IMPORT_ERROR!r}"
                )
                print(f"[HostAnalyzer Error] {self.model_error}")
            else:
                try:
                    self.model = _load_or_build_model(self.model_path)

                    if self.model is None:
                        raise RuntimeError(
                            "_load_or_build_model returned None; inspect the "
                            "preceding HostAnalyzer error."
                        )

                    self.model_loaded = True
                    print(
                        "[HostAnalyzer] MODEL LOADED: "
                        f"{Path(self.model_path).expanduser().resolve()}"
                    )
                    print(
                        "[HostAnalyzer] Model type: "
                        f"{type(self.model).__name__}"
                    )

                    if self.enable_gradcam and compute_gradcam is None:
                        print(
                            "[HostAnalyzer Warning] Grad-CAM helper unavailable: "
                            f"{GRADCAM_IMPORT_ERROR!r}"
                        )

                    if analyze_model_sample is None:
                        print(
                            "[HostAnalyzer Warning] Diagnostics helper unavailable: "
                            f"{DIAGNOSTICS_IMPORT_ERROR!r}"
                        )

                except Exception as err:
                    self.model = None
                    self.model_loaded = False
                    self.model_error = f"{type(err).__name__}: {err}"
                    print(
                        "[HostAnalyzer Error] MODEL LOAD FAILED: "
                        f"{self.model_error}"
                    )

    def _predict_host(
        self,
        inp_tensor: np.ndarray,
    ) -> Tuple[int, float]:
        """Fast class inference without Grad-CAM or diagnostic extraction."""
        if self._infer_fn is None:
            @tf.function(reduce_retracing=True)
            def infer_fn(x):
                return self.model(x, training=False)

            self._infer_fn = infer_fn

            # Compile/warm the TensorFlow graph once.
            self._infer_fn(
                tf.zeros((1, 2400, 1), dtype=tf.float32)
            )

        logits = self._infer_fn(inp_tensor)

        if isinstance(logits, dict):
            logits = next(iter(logits.values()))
        elif isinstance(logits, (list, tuple)):
            logits = logits[0]

        probabilities = tf.nn.softmax(
            logits[0],
            axis=-1,
        ).numpy()

        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])
        return class_id, confidence

    def submit_packet(
        self,
        seq: int,
        mcu_class_id: int,
        mcu_confidence: float,
        audio_i16: np.ndarray,
        received_at: float,
    ) -> None:
        """Publish only the newest telemetry frame."""

        packet = (
            seq,
            mcu_class_id,
            mcu_confidence,
            audio_i16.copy(),
            received_at,
        )

        try:
            self.input_queue.put_nowait(packet)
            return
        except queue.Full:
            pass

        try:
            self.input_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self.input_queue.put_nowait(packet)
        except queue.Full:
            pass

    def run(self) -> None:
        """Process current telemetry without accumulating stale frames."""

        while not self.stop_event.is_set():
            try:
                packet = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # If another frame arrived while waking up, process only the newest.
            while True:
                try:
                    packet = self.input_queue.get_nowait()
                except queue.Empty:
                    break

            try:
                (
                    seq,
                    mcu_class_id,
                    mcu_confidence,
                    audio_i16,
                    received_at,
                ) = packet

                self.processed_packets += 1

                do_harmonics = (
                    self.processed_packets == 1
                    or self.processed_packets
                    % self.harmonic_analysis_stride
                    == 0
                )

                do_deep_analysis = (
                    self.enable_gradcam
                    and (
                        self.processed_packets == 1
                        or self.processed_packets
                        % self.deep_analysis_stride
                        == 0
                    )
                )

                # Harmonic analysis is staggered because adjacent windows
                # overlap heavily and usually contain nearly identical data.
                if do_harmonics:
                    self.last_harmonics = analyze_harmonics(
                        audio_i16,
                        sample_rate=self.sample_rate,
                    )

                f0_hz = float(self.last_harmonics["f0_hz"])
                peak_power_db = float(
                    self.last_harmonics["peak_power_db"]
                )

                host_class_id = None
                host_confidence = None
                host_class_probability = None
                heatmap = None
                dense_embedding = None

                audio_float = (
                    audio_i16.astype(np.float32) / 32768.0
                )
                # 1. DC removal
                audio_float -= np.mean(audio_float)

                # 2. Standardized RMS normalization (matches training & ESP32 deployment)
                rms = float(np.sqrt(np.mean(np.square(audio_float)) + 1e-8))
                target_rms = 0.05
                min_gain = 0.1
                max_gain = 10.0
                gain = float(np.clip(target_rms / rms, min_gain, max_gain))
                audio_float *= gain

                # 3. Final clipping to [-1, 1]
                audio_float = np.clip(audio_float, -1.0, 1.0)

                inp_tensor = audio_float.reshape(1, -1, 1)

                if self.model is not None:
                    if isinstance(self.model, FastTFLiteModel):
                        (
                            host_class_id,
                            host_confidence,
                            host_class_probability,
                        ) = self.model.predict_fast(audio_float)

                    else:
                        # Compile the ordinary classification path once.
                        if self._compiled_inference is None:
                            self._compiled_inference = tf.function(
                                lambda value: self.model(
                                    value,
                                    training=False,
                                ),
                                reduce_retracing=True,
                            )

                        output = self._compiled_inference(inp_tensor)

                        if isinstance(output, (list, tuple)):
                            output = output[0]

                        logits = np.asarray(output)[0]
                        probabilities = tf.nn.softmax(
                            logits
                        ).numpy()

                        host_class_id = int(
                            np.argmax(probabilities)
                        )
                        host_confidence = float(
                            probabilities[host_class_id]
                        )
                        host_class_probability = probabilities

                        # Grad-CAM and embedding extraction are much more
                        # expensive than classification, so run them less often.
                        if do_deep_analysis:
                            if analyze_model_sample is not None:
                                try:
                                    diagnostic = analyze_model_sample(
                                        self.model,
                                        inp_tensor,
                                    )

                                    if (
                                        diagnostic.predicted_class_id
                                        is not None
                                    ):
                                        host_class_id = int(
                                            diagnostic.predicted_class_id
                                        )

                                    if (
                                        diagnostic.predicted_confidence
                                        is not None
                                    ):
                                        host_confidence = float(
                                            diagnostic.predicted_confidence
                                        )

                                    heatmap = (
                                        diagnostic.gradcam_heatmap
                                    )
                                    dense_embedding = (
                                        diagnostic.dense_embedding
                                    )

                                except Exception as err:
                                    print(
                                        "[HostAnalyzer] Deep analysis "
                                        f"failed: {err}"
                                    )

                            elif compute_gradcam is not None:
                                try:
                                    (
                                        heatmap,
                                        deep_class_id,
                                        deep_confidence,
                                    ) = compute_gradcam(
                                        self.model,
                                        inp_tensor,
                                        class_idx=host_class_id,
                                    )

                                    if deep_class_id is not None:
                                        host_class_id = int(
                                            deep_class_id
                                        )

                                    if deep_confidence is not None:
                                        host_confidence = float(
                                            deep_confidence
                                        )

                                except Exception as err:
                                    print(
                                        "[HostAnalyzer] Grad-CAM "
                                        f"failed: {err}"
                                    )

                discrepancy = False

                if host_class_id is not None:
                    confidence_difference = abs(
                        float(host_confidence)
                        - float(mcu_confidence)
                    )

                    discrepancy = (
                        host_class_id != mcu_class_id
                        or confidence_difference > 0.35
                    )

                elif mcu_confidence < 0.50:
                    discrepancy = True

                if self.export_anomalies and discrepancy:
                    metadata = {
                        "seq": seq,
                        "mcu_class_id": mcu_class_id,
                        "mcu_confidence": mcu_confidence,
                        "host_class_id": host_class_id,
                        "host_confidence": host_confidence,
                        "f0_hz": f0_hz,
                        "peak_power_db": peak_power_db,
                        "discrepancy": discrepancy,
                        "timestamp": received_at,
                    }

                    export_anomaly_frame(
                        audio_i16,
                        self.sample_rate,
                        metadata,
                        output_dir=self.anomaly_output_dir,
                        heatmap=heatmap,
                    )

                result = AnalysisResult(
                    seq=seq,
                    audio=audio_i16,
                    mcu_class_id=mcu_class_id,
                    mcu_confidence=mcu_confidence,
                    host_class_id=host_class_id,
                    host_confidence=host_confidence,
                    discrepancy=discrepancy,
                    f0_hz=f0_hz,
                    peak_power_db=peak_power_db,
                    heatmap=heatmap,
                    timestamp=received_at,
                    dense_embedding=dense_embedding,
                    host_class_probability=host_class_probability,
                )

                # Never leave an old result waiting for the UI.
                try:
                    self.output_queue.put_nowait(result)
                except queue.Full:
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass

                    try:
                        self.output_queue.put_nowait(result)
                    except queue.Full:
                        pass

            except Exception as err:
                print(
                    "[HostAnalyzer Worker Exception]: "
                    f"{type(err).__name__}: {err}"
                )
