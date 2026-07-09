#!/usr/bin/env python3
"""
ESP32-S3 mosquito audio + classification visualizer.

Matches the current firmware protocol:

    0x00 | COBS(
        uint32 seq                  # little-endian
        uint32 audio_timestamp_us
        uint8  predicted_class
        float32 confidence
        uint32 inference_time_us
        uint32 class_age_ms
        uint32 classifier_seq
        int16  audio[AUDIO_SAMPLE_COUNT]
    ) | 0x00

Dependencies:
    pip install numpy matplotlib pyserial

Example:
    python esp32_mosquito_visualizer.py
    python esp32_mosquito_visualizer.py --port COM6
    python esp32_mosquito_visualizer.py --port /dev/ttyUSB0
    python esp32_mosquito_visualizer.py --n-fft 1024 --hop-length 128
    python esp32_mosquito_visualizer.py --packet-hop-samples 512 --refresh-ms 20
"""

from __future__ import annotations

import argparse
import collections
import queue
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from serial.tools import list_ports

CLASS_NAMES = [
    "Ae_aegypti_Female",
    "Ae_aegypti_Male",
    "Ae_albopictus_Female",
    "Ae_albopictus_Male",
    "An_dirus_Female",
    "An_dirus_Male",
    "An_minimus_Female",
    "An_minimus_Male",
    "Cx_quin_Female",
    "Cx_quin_Male",
    "No_Mos",
]

CLASS_COLORS = {
    "Ae": "#00d6b4",
    "An": "#ff4d73",
    "Cx": "#f2c94c",
    "No": "#808080",
    "Unknown": "#ffffff",
}

DEFAULT_PORT_CANDIDATES = [
    "/dev/ttyUSB0",
    "/dev/ttyUSB1",
    "/dev/ttyACM0",
    "/dev/ttyACM1",
]


def class_color(class_name: str) -> str:
    if class_name == "Unknown":
        return CLASS_COLORS["Unknown"]
    return CLASS_COLORS.get(class_name[:2], "#ffffff")


def cobs_decode(encoded: bytes) -> bytes:
    """Decode one COBS frame without its 0x00 delimiter."""
    if not encoded:
        return b""

    decoded = bytearray()
    index = 0
    encoded_len = len(encoded)

    while index < encoded_len:
        code = encoded[index]
        if code == 0:
            raise ValueError("COBS frame contains an unexpected zero byte")

        index += 1
        block_end = index + code - 1

        if block_end > encoded_len:
            raise ValueError("COBS code exceeds remaining frame length")

        decoded.extend(encoded[index:block_end])
        index = block_end

        if code != 0xFF and index < encoded_len:
            decoded.append(0)

    return bytes(decoded)


HEADER_FORMAT = "<IIBfIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


@dataclass(frozen=True)
class TelemetryPacket:
    seq: int
    audio_timestamp_us: int
    class_id: int
    confidence: float
    inference_time_us: int
    class_age_ms: int
    classifier_seq: int
    audio_i16: np.ndarray
    received_at: float


@dataclass
class ReaderStats:
    bytes_received: int = 0
    valid_packets: int = 0
    empty_frames: int = 0
    cobs_errors: int = 0
    length_errors: int = 0
    value_errors: int = 0
    queue_drops: int = 0
    buffer_resets: int = 0
    last_error: str = ""


class TelemetryReader(threading.Thread):
    """Read, frame, decode, and validate telemetry away from the UI thread."""

    def __init__(
        self,
        serial_port: serial.Serial,
        sample_count: int,
        output_queue: queue.Queue[TelemetryPacket],
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="telemetry-reader", daemon=True)
        self.serial_port = serial_port
        self.sample_count = sample_count
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.stats = ReaderStats()
        self.expected_payload_size = HEADER_SIZE + sample_count * 2
        self.max_encoded_size = (
            self.expected_payload_size + self.expected_payload_size // 254 + 2
        )
        self.rx_buffer = bytearray()

    def _publish_latest(self, packet: TelemetryPacket) -> None:
        try:
            self.output_queue.put_nowait(packet)
        except queue.Full:
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                pass
            self.stats.queue_drops += 1
            self.output_queue.put_nowait(packet)

    def _decode_frame(self, frame: bytes) -> None:
        try:
            payload = cobs_decode(frame)
        except ValueError:
            self.stats.cobs_errors += 1
            return

        if len(payload) != self.expected_payload_size:
            self.stats.length_errors += 1
            return

        (
            seq,
            audio_timestamp_us,
            class_id,
            confidence,
            inference_time_us,
            class_age_ms,
            classifier_seq,
        ) = struct.unpack_from(HEADER_FORMAT, payload, 0)

        if not np.isfinite(confidence):
            self.stats.value_errors += 1
            return

        audio = np.frombuffer(
            payload,
            dtype="<i2",
            count=self.sample_count,
            offset=HEADER_SIZE,
        ).copy()

        packet = TelemetryPacket(
            seq=seq,
            audio_timestamp_us=audio_timestamp_us,
            class_id=class_id,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            inference_time_us=inference_time_us,
            class_age_ms=class_age_ms,
            classifier_seq=classifier_seq,
            audio_i16=audio,
            received_at=time.monotonic(),
        )
        self.stats.valid_packets += 1
        self._publish_latest(packet)

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                waiting = self.serial_port.in_waiting
                chunk = self.serial_port.read(waiting if waiting > 0 else 1)

                if not chunk:
                    continue

                self.stats.bytes_received += len(chunk)
                self.rx_buffer.extend(chunk)

                # Firmware uses 0x00 as both the leading and trailing delimiter.
                while True:
                    delimiter = self.rx_buffer.find(0)
                    if delimiter < 0:
                        break

                    frame = bytes(self.rx_buffer[:delimiter])
                    del self.rx_buffer[: delimiter + 1]

                    if not frame:
                        self.stats.empty_frames += 1
                        continue

                    self._decode_frame(frame)

                # Recover if connection starts mid-frame or data becomes corrupted.
                if len(self.rx_buffer) > self.max_encoded_size * 3:
                    self.rx_buffer.clear()
                    self.stats.buffer_resets += 1

        except (serial.SerialException, OSError) as exc:
            self.stats.last_error = str(exc)
            self.stop_event.set()


def choose_port(explicit_port: Optional[str]) -> str:
    if explicit_port:
        return explicit_port

    available = list(list_ports.comports())
    available_names = {port.device for port in available}

    for candidate in DEFAULT_PORT_CANDIDATES:
        if candidate in available_names:
            return candidate

    # Prefer USB serial devices when the operating system exposes metadata.
    preferred_terms = ("USB", "UART", "CP210", "CH340", "JTAG", "ESP")
    for port in available:
        description = f"{port.description} {port.manufacturer or ''}".upper()
        if any(term in description for term in preferred_terms):
            return port.device

    if available:
        return available[0].device

    raise RuntimeError("No serial ports were found")


def open_serial(port: str, baud: int) -> serial.Serial:
    connection = serial.Serial(
        port=port,
        baudrate=baud,
        timeout=0.01,
        write_timeout=0.2,
    )
    connection.reset_input_buffer()
    return connection


def compute_stft_db(
    samples: np.ndarray,
    n_fft: int,
    hop_length: int,
    floor_db: float,
) -> np.ndarray:
    """
    Return frequency x time STFT magnitude in dBFS.

    The signal is centered with half-window padding so the number of output
    columns tracks elapsed audio time more closely. Per-frame mean removal
    suppresses DC/very-low-frequency smear.
    """
    samples = np.asarray(samples, dtype=np.float32)

    if samples.size == 0:
        return np.full((n_fft // 2 + 1, 1), floor_db, dtype=np.float32)

    pad = n_fft // 2
    if samples.size > 1:
        samples = np.pad(samples, (pad, pad), mode="reflect")
    else:
        samples = np.pad(samples, (pad, pad), mode="constant")

    if len(samples) < n_fft:
        samples = np.pad(samples, (0, n_fft - len(samples)))

    frame_count = 1 + (len(samples) - n_fft) // hop_length
    shape = (frame_count, n_fft)
    strides = (samples.strides[0] * hop_length, samples.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=shape,
        strides=strides,
        writeable=False,
    )

    window = np.hanning(n_fft).astype(np.float32)
    detrended = frames - np.mean(frames, axis=1, keepdims=True)
    spectrum = np.fft.rfft(detrended * window, n=n_fft, axis=1)

    # Scaling gives an approximately full-scale referenced magnitude.
    scale = max(float(window.sum()) / 2.0, 1.0)
    magnitude = np.abs(spectrum) / scale
    db = 20.0 * np.log10(np.maximum(magnitude, 10.0 ** (floor_db / 20.0)))
    return np.maximum(db, floor_db).T.astype(np.float32)


class Visualizer:
    def __init__(
        self,
        packet_queue: queue.Queue[TelemetryPacket],
        reader: TelemetryReader,
        stop_event: threading.Event,
        port: str,
        baud: int,
        sample_rate: int,
        sample_count: int,
        history_seconds: float,
        detection_threshold: float,
        min_frequency: float,
        max_frequency: float,
        packet_hop_samples: int,
        live_wave_seconds: float,
        n_fft: int,
        hop_length: int,
        floor_db: float,
        ceiling_db: float,
        refresh_ms: int,
    ) -> None:
        self.packet_queue = packet_queue
        self.reader = reader
        self.stop_event = stop_event
        self.port = port
        self.baud = baud
        self.fs = sample_rate
        self.sample_count = sample_count
        self.window_seconds = sample_count / sample_rate
        self.history_seconds = history_seconds
        self.detection_threshold = detection_threshold
        self.min_frequency = max(0.0, min_frequency)
        self.max_frequency = min(max_frequency, sample_rate / 2)
        self.packet_hop_samples = min(packet_hop_samples, sample_count)
        self.packet_advance_seconds = self.packet_hop_samples / sample_rate
        self.live_wave_seconds = live_wave_seconds

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.floor_db = floor_db
        self.ceiling_db = ceiling_db
        self.refresh_ms = refresh_ms

        self.freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.fs)
        self.spec_columns = max(
            50, int(round(self.history_seconds * self.fs / self.hop_length))
        )
        self.spec_matrix = np.full(
            (len(self.freqs), self.spec_columns),
            self.floor_db,
            dtype=np.float32,
        )
        self.spec_column_remainder = 0.0
        self.live_wave_sample_count = max(
            self.sample_count,
            int(round(self.live_wave_seconds * self.fs)),
        )
        self.live_wave_buffer = np.zeros(self.live_wave_sample_count, dtype=np.float32)

        self.spec_annotations = []

        self.last_packet_time: Optional[float] = None
        self.last_stats_time = time.monotonic()
        self.last_stats_bytes = 0
        self.last_stats_packets = 0
        self.bytes_per_second = 0.0
        self.packets_per_second = 0.0

        self._build_figure()

    def _build_figure(self) -> None:
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(15, 8.5))
        grid = self.fig.add_gridspec(2, 1, height_ratios=[1.0, 2.1])

        self.ax_wave = self.fig.add_subplot(grid[0])
        self.ax_spec = self.fig.add_subplot(grid[1])

        self.fig.canvas.manager.set_window_title("ESP32 Mosquito Edge-ML Monitor")
        self.fig.suptitle("Waiting for telemetry...", fontsize=16, fontweight="bold")

        # Waveform
        wave_time = (
            np.arange(self.live_wave_sample_count, dtype=np.float32)
            - self.live_wave_sample_count
            + 1
        ) / self.fs
        (self.wave_line,) = self.ax_wave.plot(
            wave_time,
            self.live_wave_buffer,
            linewidth=0.9,
        )
        self.ax_wave.set_xlim(wave_time[0], 0.0)
        self.ax_wave.set_ylim(-1.05, 1.05)
        self.ax_wave.set_title("Live rolling waveform")
        self.ax_wave.set_xlabel("Time (s)")
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.grid(True, alpha=0.18)
        self.wave_info = self.ax_wave.text(
            0.012,
            0.93,
            "",
            transform=self.ax_wave.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
        )

        # Spectrogram
        self.spec_image = self.ax_spec.imshow(
            self.spec_matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[-self.history_seconds, 0.0, 0.0, self.fs / 2.0],
            cmap="inferno",
            vmin=self.floor_db,
            vmax=self.ceiling_db,
        )
        self.ax_spec.set_ylim(self.min_frequency, self.max_frequency)
        self.ax_spec.set_title("Scrolling STFT spectrogram")
        self.ax_spec.set_xlabel("History (seconds)")
        self.ax_spec.set_ylabel("Frequency (Hz)")
        colorbar = self.fig.colorbar(
            self.spec_image,
            ax=self.ax_spec,
            pad=0.01,
            fraction=0.025,
        )
        colorbar.set_label("Magnitude (dBFS)")

        self.status_text = self.fig.text(
            0.01,
            0.008,
            "",
            ha="left",
            va="bottom",
            fontsize=9,
            family="monospace",
        )

        self.fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.96))
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _on_close(self, _event) -> None:
        self.stop_event.set()

    def _drain_packets(self) -> list[TelemetryPacket]:
        packets = []
        while True:
            try:
                packets.append(self.packet_queue.get_nowait())
            except queue.Empty:
                return packets

    def _peak_frequency(self, audio: np.ndarray) -> float:
        window = np.hanning(len(audio))
        spectrum = np.abs(np.fft.rfft(audio * window))
        frequencies = np.fft.rfftfreq(len(audio), d=1.0 / self.fs)

        band = (frequencies >= 150.0) & (frequencies <= min(1500.0, self.fs / 2))
        if not np.any(band):
            return 0.0

        band_indices = np.flatnonzero(band)
        peak_index = band_indices[int(np.argmax(spectrum[band]))]
        return float(frequencies[peak_index])

    def _append_spectrogram(self, audio: np.ndarray) -> None:
        new_columns = compute_stft_db(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            floor_db=self.floor_db,
        )

        # When firmware sends overlapping inference windows, append only the
        # newest stride instead of duplicating the whole window in history.
        exact_columns = (
            self.packet_hop_samples / self.hop_length + self.spec_column_remainder
        )
        expected_columns = int(np.floor(exact_columns))
        self.spec_column_remainder = exact_columns - expected_columns

        if expected_columns <= 0:
            return

        column_count = min(
            expected_columns,
            new_columns.shape[1],
            self.spec_columns,
        )
        newest = new_columns[:, -column_count:]

        if column_count >= self.spec_columns:
            self.spec_matrix[:, :] = newest[:, -self.spec_columns :]
        else:
            self.spec_matrix[:, :-column_count] = self.spec_matrix[:, column_count:]
            self.spec_matrix[:, -column_count:] = newest

        self.spec_image.set_data(self.spec_matrix)

    def _append_live_waveform(self, audio: np.ndarray) -> None:
        newest = audio[-self.packet_hop_samples :]
        count = min(len(newest), self.live_wave_sample_count)
        if count <= 0:
            return

        self.live_wave_buffer[:-count] = self.live_wave_buffer[count:]
        self.live_wave_buffer[-count:] = newest[-count:]
        self.wave_line.set_ydata(self.live_wave_buffer)

    def _age_spec_annotations(self) -> None:
        """Shift existing spectrogram boxes left as new windows arrive."""
        kept = []
        for rect, text in self.spec_annotations:
            new_x = rect.get_x() - self.packet_advance_seconds
            rect.set_x(new_x)

            tx, ty = text.get_position()
            text.set_position((tx - self.packet_advance_seconds, ty))

            if new_x + rect.get_width() < -self.history_seconds:
                rect.remove()
                text.remove()
            else:
                kept.append((rect, text))

        self.spec_annotations = kept

    def _short_class_label(self, class_name: str) -> str:
        if class_name == "Unknown":
            return "Unknown"
        if class_name == "No_Mos":
            return "NoMos"

        parts = class_name.split("_")
        if len(parts) >= 3:
            species = parts[1]
            sex = parts[2][0]
            return f"{species} {sex}"
        return class_name

    def _detection_band(
        self, class_name: str, peak_frequency: float
    ) -> tuple[float, float]:
        """
        Return a reasonable frequency band for the bounding box.
        Uses dominant frequency when it is plausible, otherwise falls back.
        """
        visible_span = self.max_frequency - self.min_frequency
        fallback_low = max(
            self.min_frequency,
            min(250.0, self.max_frequency - 0.6 * visible_span),
        )
        fallback_high = min(
            self.max_frequency,
            max(900.0, fallback_low + 0.25 * visible_span),
        )

        if peak_frequency < 200.0 or peak_frequency > self.max_frequency:
            return fallback_low, fallback_high

        if "Male" in class_name:
            half_band = 120.0
        elif "Female" in class_name:
            half_band = 180.0
        else:
            half_band = 150.0

        low = max(self.min_frequency, 150.0, peak_frequency - half_band)
        high = min(self.max_frequency, peak_frequency + half_band)

        if high - low < 120.0:
            low = max(self.min_frequency, 150.0, peak_frequency - 60.0)
            high = min(self.max_frequency, peak_frequency + 60.0)

        return low, high

    def _add_spec_box(
        self,
        class_name: str,
        confidence: float,
        color: str,
        peak_frequency: float,
    ) -> None:
        """Draw a box over the newest spectrogram segment."""
        low_f, high_f = self._detection_band(class_name, peak_frequency)

        x0 = -self.packet_advance_seconds
        width = self.packet_advance_seconds
        height = high_f - low_f

        rect = patches.Rectangle(
            (x0, low_f),
            width,
            height,
            linewidth=1.8,
            edgecolor=color,
            facecolor="none",
            alpha=0.95,
        )
        self.ax_spec.add_patch(rect)

        label = f"{self._short_class_label(class_name)}  {confidence * 100:.0f}%"
        text_y = min(high_f + 25.0, self.max_frequency - 20.0)
        text = self.ax_spec.text(
            x0 + 0.01,
            text_y,
            label,
            color=color,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 1.5},
        )

        self.spec_annotations.append((rect, text))

    def _update_rate_stats(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_stats_time
        if elapsed < 1.0:
            return

        stats = self.reader.stats
        self.bytes_per_second = (stats.bytes_received - self.last_stats_bytes) / elapsed
        self.packets_per_second = (
            stats.valid_packets - self.last_stats_packets
        ) / elapsed

        self.last_stats_time = now
        self.last_stats_bytes = stats.bytes_received
        self.last_stats_packets = stats.valid_packets

    def _render_packet(self, packet: TelemetryPacket) -> None:
        class_name = (
            CLASS_NAMES[packet.class_id]
            if 0 <= packet.class_id < len(CLASS_NAMES)
            else "Unknown"
        )
        color = class_color(class_name)
        audio = packet.audio_i16.astype(np.float32) / 32768.0

        self._age_spec_annotations()

        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio)))
        peak_frequency = self._peak_frequency(audio)

        is_detection = (
            class_name not in ("No_Mos", "Unknown")
            and packet.confidence >= self.detection_threshold
        )

        self._append_live_waveform(audio)
        self.wave_line.set_color(color if is_detection else "#a0a0a0")

        if is_detection:
            title = (
                f"DETECTED: {class_name}  |  "
                f"confidence {packet.confidence * 100.0:.1f}%"
            )
            self.fig.suptitle(title, color=color, fontsize=16, fontweight="bold")
            self.ax_wave.set_facecolor("#151515")
        elif class_name == "No_Mos":
            self.fig.suptitle(
                f"No mosquito  |  confidence {packet.confidence * 100.0:.1f}%",
                color="#a0a0a0",
                fontsize=15,
                fontweight="normal",
            )
            self.ax_wave.set_facecolor("#101010")
        else:
            self.fig.suptitle(
                f"Uncertain: {class_name}  |  confidence {packet.confidence * 100.0:.1f}%",
                color="#dddddd",
                fontsize=15,
                fontweight="normal",
            )
            self.ax_wave.set_facecolor("#101010")

        infer_ms = packet.inference_time_us / 1000.0
        class_age = (
            "none" if packet.class_age_ms == 0xFFFFFFFF
            else f"{packet.class_age_ms} ms"
        )
        self.wave_info.set_text(
            f"class={packet.class_id}: {class_name}\n"
            f"confidence={packet.confidence:.3f}   "
            f"infer={infer_ms:.2f} ms   class_age={class_age}   "
            f"class_seq={packet.classifier_seq}   audio_seq={packet.seq}\n"
            f"RMS={rms:.4f}   peak={peak:.4f}   "
            f"dominant≈{peak_frequency:.1f} Hz"
        )
        self.wave_info.set_color(color if is_detection else "white")

        self._append_spectrogram(audio)
        if is_detection:
            self._add_spec_box(
                class_name=class_name,
                confidence=packet.confidence,
                color=color,
                peak_frequency=peak_frequency,
            )

        self.last_packet_time = packet.received_at

    def update(self, _frame):
        for packet in self._drain_packets():
            self._render_packet(packet)

        self._update_rate_stats()
        stats = self.reader.stats

        age_text = "never"
        if self.last_packet_time is not None:
            age_text = f"{time.monotonic() - self.last_packet_time:.2f}s"

        status = (
            f"{self.port} @ {self.baud:,} baud | "
            f"{self.fs} Hz, {self.sample_count} samples/window "
            f"({self.window_seconds * 1000.0:.0f} ms), "
            f"step={self.packet_hop_samples} "
            f"({self.packet_advance_seconds * 1000.0:.0f} ms) | "
            f"valid={stats.valid_packets}  "
            f"rate={self.packets_per_second:.2f} pkt/s  "
            f"serial={self.bytes_per_second / 1024.0:.1f} KiB/s | "
            f"COBS_err={stats.cobs_errors}  "
            f"len_err={stats.length_errors}  "
            f"dropped={stats.queue_drops}  "
            f"last={age_text}"
        )
        if stats.last_error:
            status += f" | SERIAL ERROR: {stats.last_error}"

        self.status_text.set_text(status)

        return (
            self.wave_line,
            self.spec_image,
            self.wave_info,
            self.status_text,
        )

    def run(self) -> None:
        # UI refresh can be faster than packet arrival; each tick drains queued packets.
        self.animation = FuncAnimation(
            self.fig,
            self.update,
            interval=self.refresh_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize COBS-framed ESP32 audio and classification telemetry."
    )
    parser.add_argument("--port", help="Serial port, for example COM6 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--fs", type=int, default=8_000)
    parser.add_argument("--samples", type=int, default=2_400)
    parser.add_argument("--history", type=float, default=12.0)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--min-frequency", type=float, default=0.0)
    parser.add_argument("--max-frequency", type=float, default=4_000.0)
    parser.add_argument(
        "--packet-hop-samples",
        type=int,
        default=None,
        help=(
            "Number of new audio samples between packets. Use a smaller value "
            "when firmware sends overlapping inference windows. Defaults to "
            "half the window for the current firmware's 50% overlap."
        ),
    )
    parser.add_argument(
        "--live-wave-seconds",
        type=float,
        default=2.0,
        help="Seconds of rolling waveform to show in the live view.",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=1024,
        help="FFT size. Larger values improve frequency resolution.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=128,
        help="STFT hop in samples. Smaller values produce denser time columns.",
    )
    parser.add_argument("--floor-db", type=float, default=-100.0)
    parser.add_argument("--ceiling-db", type=float, default=-20.0)
    parser.add_argument("--refresh-ms", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.fs <= 0 or args.samples <= 0:
        raise ValueError("--fs and --samples must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.n_fft <= 0 or args.n_fft & (args.n_fft - 1):
        raise ValueError("--n-fft must be a positive power of two")
    if not 0 < args.hop_length <= args.n_fft:
        raise ValueError("--hop-length must be between 1 and --n-fft")
    if args.floor_db >= args.ceiling_db:
        raise ValueError("--floor-db must be below --ceiling-db")
    if args.refresh_ms <= 0:
        raise ValueError("--refresh-ms must be positive")
    if args.live_wave_seconds <= 0:
        raise ValueError("--live-wave-seconds must be positive")
    if not 0.0 <= args.min_frequency < args.max_frequency <= args.fs / 2:
        raise ValueError("frequency limits must satisfy 0 <= min < max <= Nyquist")

    packet_hop_samples = (
        args.samples // 2 if args.packet_hop_samples is None else args.packet_hop_samples
    )
    if not 0 < packet_hop_samples <= args.samples:
        raise ValueError("--packet-hop-samples must be between 1 and --samples")

    port = choose_port(args.port)
    serial_port = open_serial(port, args.baud)
    print(f"Connected to {port} at {args.baud:,} baud")

    packet_queue: queue.Queue[TelemetryPacket] = queue.Queue(maxsize=32)
    stop_event = threading.Event()
    reader = TelemetryReader(
        serial_port=serial_port,
        sample_count=args.samples,
        output_queue=packet_queue,
        stop_event=stop_event,
    )
    reader.start()

    visualizer = Visualizer(
        packet_queue=packet_queue,
        reader=reader,
        stop_event=stop_event,
        port=port,
        baud=args.baud,
        sample_rate=args.fs,
        sample_count=args.samples,
        history_seconds=args.history,
        detection_threshold=args.threshold,
        min_frequency=args.min_frequency,
        max_frequency=args.max_frequency,
        packet_hop_samples=packet_hop_samples,
        live_wave_seconds=args.live_wave_seconds,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        floor_db=args.floor_db,
        ceiling_db=args.ceiling_db,
        refresh_ms=args.refresh_ms,
    )

    try:
        visualizer.run()
    finally:
        stop_event.set()
        reader.join(timeout=1.0)
        if serial_port.is_open:
            serial_port.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
