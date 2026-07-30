"""Host-side model analysis background executor for visualizer telemetry streams.

Performs host inference, Grad-CAM heatmap extraction, harmonic analysis,
and MCU-Host discrepancy detection.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np

from wingbeat_ml.visualizer.spectrogram import analyze_harmonics
from wingbeat_ml.visualizer.exporter import export_anomaly_frame

try:
    import tensorflow as tf
    from wingbeat_ml.evaluation.gradcam import compute_gradcam
    from wingbeat_ml.evaluation.diagnostics import analyze_model_sample
except ImportError:  # pragma: no cover
    tf = None
    compute_gradcam = None
    analyze_model_sample = None


def _load_or_build_model(model_path: str):
    if tf is None:
        return None

    path_obj = Path(model_path).resolve()
    if not path_obj.exists():
        print(f"[HostAnalyzer Error]: Model file not found at '{model_path}' (Resolved: '{path_obj}')")
        return None

    path = str(path_obj)
    if not path.endswith(".weights.h5"):
        try:
            return tf.keras.models.load_model(path)
        except Exception:
            pass

    try:
        from wingbeat_ml.models import MosSongPlusModel
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


class HostAnalyzer(threading.Thread):
    """Asynchronous analysis engine processing incoming telemetry frames."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        sample_rate: int = 8000,
        enable_gradcam: bool = False,
        export_anomalies: bool = False,
        anomaly_output_dir: str = "output/misclassifications",
    ) -> None:
        super().__init__(name="host-analyzer", daemon=True)
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.enable_gradcam = enable_gradcam
        self.export_anomalies = export_anomalies
        self.anomaly_output_dir = anomaly_output_dir

        self.input_queue: queue.Queue[Tuple[int, int, float, np.ndarray, float]] = queue.Queue(maxsize=10)
        self.output_queue: queue.Queue[AnalysisResult] = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.model = None

        if self.model_path and tf is not None:
            self.model = _load_or_build_model(self.model_path)

    def submit_packet(
        self,
        seq: int,
        mcu_class_id: int,
        mcu_confidence: float,
        audio_i16: np.ndarray,
        received_at: float,
    ) -> None:
        try:
            self.input_queue.put_nowait((seq, mcu_class_id, mcu_confidence, audio_i16.copy(), received_at))
        except queue.Full:
            pass  # ponytail: drop stale frame if processing queue is full

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                packet = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                seq, mcu_class_id, mcu_confidence, audio_i16, received_at = packet

                # 1. Harmonic analysis
                harmonics = analyze_harmonics(audio_i16, sample_rate=self.sample_rate)
                f0_hz = harmonics["f0_hz"]
                peak_power_db = harmonics["peak_power_db"]

                # 2. Host model inference & Grad-CAM & Dense Analysis
                host_class_id = None
                host_confidence = None
                heatmap = None
                dense_embedding = None

                audio_float = audio_i16.astype(np.float32) / 32768.0
                audio_float = audio_float - np.mean(audio_float)
                peak = np.max(np.abs(audio_float))
                if peak > 1e-6:
                    audio_float = (audio_float / peak) * 0.95

                if self.model is not None:
                    inp_tensor = np.reshape(audio_float, (1, -1, 1))
                    if analyze_model_sample is not None:
                        try:
                            diag = analyze_model_sample(self.model, inp_tensor)
                            host_class_id = diag.predicted_class_id
                            host_confidence = diag.predicted_confidence
                            heatmap = diag.gradcam_heatmap
                            dense_embedding = diag.dense_embedding
                        except Exception as err:
                            if self.enable_gradcam and compute_gradcam is not None:
                                heatmap, host_class_id, host_confidence = compute_gradcam(
                                    self.model, inp_tensor, class_idx=None
                                )
                            else:
                                logits = self.model.predict(inp_tensor, verbose=0)[0]
                                probs = tf.nn.softmax(logits).numpy()
                                host_class_id = int(np.argmax(probs))
                                host_confidence = float(probs[host_class_id])
                    elif self.enable_gradcam and compute_gradcam is not None:
                        try:
                            heatmap, host_class_id, host_confidence = compute_gradcam(
                                self.model, inp_tensor, class_idx=None
                            )
                        except Exception:
                            logits = self.model.predict(inp_tensor, verbose=0)[0]
                            probs = tf.nn.softmax(logits).numpy()
                            host_class_id = int(np.argmax(probs))
                            host_confidence = float(probs[host_class_id])
                    else:
                        logits = self.model.predict(inp_tensor, verbose=0)[0]
                        probs = tf.nn.softmax(logits).numpy()
                        host_class_id = int(np.argmax(probs))
                        host_confidence = float(probs[host_class_id])

                # 3. Discrepancy detection
                discrepancy = False
                if host_class_id is not None:
                    if host_class_id != mcu_class_id or abs(host_confidence - mcu_confidence) > 0.35:
                        discrepancy = True
                elif mcu_confidence < 0.50:
                    discrepancy = True

                # 4. Anomaly Export
                if self.export_anomalies and discrepancy:
                    meta = {
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
                        meta,
                        output_dir=self.anomaly_output_dir,
                        heatmap=heatmap,
                    )

                res = AnalysisResult(
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
                )

                try:
                    self.output_queue.put_nowait(res)
                except queue.Full:
                    pass
            except Exception as err:
                print(f"[HostAnalyzer Worker Exception]: {err}")
