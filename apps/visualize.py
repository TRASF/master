from __future__ import annotations

import argparse
import os
import queue
import re
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Ensure 'src' is in sys.path for direct script execution
_repo_src = Path(__file__).resolve().parents[1] / "src"
if _repo_src.exists() and str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from serial.tools import list_ports

HOST_ANALYZER_IMPORT_ERROR = None

try:
    from wingbeat_ml.visualizer.analyzer import HostAnalyzer
except Exception as exc:
    import traceback

    HostAnalyzer = None
    HOST_ANALYZER_IMPORT_ERROR = exc
    print("[Visualizer] HostAnalyzer import failed:")
    traceback.print_exc()

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

# Restrained semantic colors. Species identity remains visible without making
# the interface look like a neon/cyber dashboard.
CLASS_COLORS = {
    "Ae": "#5B8FF9",
    "An": "#D97588",
    "Cx": "#CDA349",
    "No": "#7E8794",
    "Unknown": "#AEB6C2",
}


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


HEADER_FORMAT = "<IIBf11fIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


@dataclass(frozen=True)
class TelemetryPacket:
    seq: int
    audio_timestamp_us: int
    class_id: int
    confidence: float
    class_probability: tuple[float, ...]
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

        unpacked = struct.unpack_from(HEADER_FORMAT, payload, 0)
        seq = unpacked[0]
        audio_timestamp_us = unpacked[1]
        class_id = unpacked[2]
        confidence = unpacked[3]
        class_probability = unpacked[4:15]
        inference_time_us = unpacked[15]
        class_age_ms = unpacked[16]
        classifier_seq = unpacked[17]

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
            class_probability=class_probability,
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


# USB vendor IDs are used only as ranking hints. They are never mandatory,
# because replica ESP32 boards may use many different USB-to-UART bridges.
KNOWN_USB_SERIAL_VIDS = {
    0x303A,  # Espressif native USB/JTAG/Serial
    0x10C4,  # Silicon Labs CP210x
    0x1A86,  # WCH CH340/CH341/CH910x
    0x0403,  # FTDI
    0x067B,  # Prolific PL2303
}

ESP_SERIAL_HINTS = (
    "esp32",
    "espressif",
    "usb jtag",
    "jtag/serial",
    "usb serial",
    "usb-serial",
    "usb uart",
    "uart bridge",
    "cp210",
    "silicon labs",
    "ch340",
    "ch341",
    "ch910",
    "wch",
    "ftdi",
    "pl2303",
)

UNWANTED_PORT_HINTS = (
    "bluetooth",
    "infrared",
    "irda",
    "dial-up modem",
)


def parse_usb_id(value: str) -> int:
    """Parse a 16-bit USB VID/PID in decimal or hexadecimal notation."""
    text = value.strip().lower()
    try:
        if text.startswith("0x") or any(char in "abcdef" for char in text):
            parsed = int(text.removeprefix("0x"), 16)
        else:
            parsed = int(text, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid USB ID {value!r}; use values such as 0x303A or 303A"
        ) from exc

    if not 0 <= parsed <= 0xFFFF:
        raise argparse.ArgumentTypeError("USB IDs must be between 0x0000 and 0xFFFF")
    return parsed


def _port_text(port) -> str:
    """Combine available cross-platform serial metadata into searchable text."""
    values = (
        getattr(port, "device", None),
        getattr(port, "name", None),
        getattr(port, "description", None),
        getattr(port, "hwid", None),
        getattr(port, "manufacturer", None),
        getattr(port, "product", None),
        getattr(port, "serial_number", None),
        getattr(port, "location", None),
        getattr(port, "interface", None),
    )
    return " ".join(str(value) for value in values if value).casefold()


def _usb_id_text(value: Optional[int]) -> str:
    return "----" if value is None else f"{value:04X}"


def _port_identity(port) -> tuple:
    """Create a physical-device identity used to collapse Linux symlink copies."""
    vid = getattr(port, "vid", None)
    pid = getattr(port, "pid", None)
    serial_number = getattr(port, "serial_number", None)
    location = getattr(port, "location", None)

    interface = getattr(port, "interface", None)
    if vid is not None and pid is not None and (serial_number or location):
        # Location is included because low-cost clone adapters sometimes reuse
        # the same serial number across multiple physical boards.
        return ("usb", vid, pid, serial_number, location, interface)
    return ("device", getattr(port, "device", ""))


def _path_preference(port) -> int:
    """Prefer stable or connection-oriented device paths when duplicates exist."""
    device = getattr(port, "device", "").casefold()
    if "/dev/serial/by-id/" in device:
        return 50
    if "/dev/serial/by-path/" in device:
        return 40
    if device.startswith("/dev/cu."):
        return 30
    if device.startswith("/dev/ttyacm") or device.startswith("/dev/ttyusb"):
        return 20
    if re.fullmatch(r"com\d+", device):
        return 20
    return 0


def list_serial_ports() -> list:
    """Enumerate serial ports on Windows, macOS, Linux, and BSD."""
    discovered = list(list_ports.comports(include_links=True))

    # include_links=True can return both a Linux tty path and one or more
    # symlinks for the same physical interface. Keep the most stable path.
    unique = {}
    for port in discovered:
        key = _port_identity(port)
        current = unique.get(key)
        if current is None or _path_preference(port) > _path_preference(current):
            unique[key] = port

    return sorted(unique.values(), key=lambda port: port.device.casefold())


def is_unwanted_port(port) -> bool:
    text = _port_text(port)
    return any(hint in text for hint in UNWANTED_PORT_HINTS)


def score_port(port) -> int:
    """Rank likely ESP32 ports without excluding unknown replica-board adapters."""
    score = 0
    text = _port_text(port)
    device = port.device.casefold()
    vid = getattr(port, "vid", None)

    if vid == 0x303A:
        score += 120
    elif vid in KNOWN_USB_SERIAL_VIDS:
        score += 60

    for hint in ESP_SERIAL_HINTS:
        if hint in text:
            score += 12

    if getattr(port, "serial_number", None):
        score += 8
    if getattr(port, "location", None):
        score += 4

    if "/dev/serial/by-id/" in device:
        score += 35
    elif "/dev/serial/by-path/" in device:
        score += 25
    elif device.startswith("/dev/cu."):
        score += 20
    elif device.startswith("/dev/ttyacm"):
        score += 18
    elif device.startswith("/dev/ttyusb"):
        score += 16
    elif re.fullmatch(r"com\d+", device):
        score += 10

    # Physical motherboard UARTs are less likely than USB serial devices.
    if device.startswith("/dev/ttys") and vid is None:
        score -= 25

    if is_unwanted_port(port):
        score -= 1000

    return score


def format_port(port) -> str:
    """Format one serial device for --list-ports and error diagnostics."""
    description = (
        getattr(port, "product", None)
        or getattr(port, "description", None)
        or "unknown device"
    )
    manufacturer = getattr(port, "manufacturer", None) or "-"
    serial_number = getattr(port, "serial_number", None) or "-"
    location = getattr(port, "location", None) or "-"
    return (
        f"{port.device:<28} "
        f"score={score_port(port):>4}  "
        f"VID:PID={_usb_id_text(getattr(port, 'vid', None))}:"
        f"{_usb_id_text(getattr(port, 'pid', None))}  "
        f"manufacturer={manufacturer}  product={description}  "
        f"serial={serial_number}  location={location}"
    )


def _matches_port_filters(
    port,
    target_vid: Optional[int],
    target_pid: Optional[int],
    target_serial: Optional[str],
    port_match: Optional[str],
) -> bool:
    if target_vid is not None and getattr(port, "vid", None) != target_vid:
        return False
    if target_pid is not None and getattr(port, "pid", None) != target_pid:
        return False
    if target_serial is not None:
        actual = (getattr(port, "serial_number", None) or "").casefold()
        if actual != target_serial.casefold():
            return False
    if port_match is not None and port_match.casefold() not in _port_text(port):
        return False
    return True


def open_serial(
    port: str,
    baud: int,
    *,
    timeout: float = 0.01,
    clear_input: bool = True,
) -> serial.Serial:
    """Open a serial port while minimizing ESP32 auto-reset line toggling."""
    connection = serial.Serial(
        port=None,
        baudrate=baud,
        timeout=timeout,
        write_timeout=0.2,
        rtscts=False,
        dsrdtr=False,
    )
    connection.port = port

    # Set inactive line states before opening. Some ESP32 boards connect DTR
    # and RTS to EN/BOOT, so careless line changes can reset the board.
    connection.dtr = False
    connection.rts = False
    connection.open()

    if clear_input:
        try:
            connection.reset_input_buffer()
        except (serial.SerialException, OSError):
            connection.close()
            raise

    return connection


def _valid_telemetry_payload(payload: bytes, sample_count: int) -> bool:
    """Validate enough of a decoded frame to identify this firmware protocol."""
    expected_size = HEADER_SIZE + sample_count * 2
    if len(payload) != expected_size:
        return False

    try:
        unpacked = struct.unpack_from(HEADER_FORMAT, payload, 0)
        class_id = unpacked[2]
        confidence = unpacked[3]
    except (struct.error, IndexError):
        return False

    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return False

    # Current and future class sets are expected to stay far below 255. This
    # rejects random binary streams without tying discovery to CLASS_NAMES.
    if class_id > 63:
        return False

    audio = np.frombuffer(
        payload,
        dtype="<i2",
        count=sample_count,
        offset=HEADER_SIZE,
    )
    return audio.size == sample_count


def probe_telemetry_port(
    port: str,
    baud: int,
    sample_count: int,
    timeout_seconds: float,
) -> tuple[Optional[serial.Serial], str]:
    """
    Open a candidate and return its live connection after one valid frame.

    Keeping the successful connection open avoids probing the board and then
    opening it a second time, which could trigger another ESP32 reset.
    """
    connection: Optional[serial.Serial] = None
    expected_payload_size = HEADER_SIZE + sample_count * 2
    max_encoded_size = expected_payload_size + expected_payload_size // 254 + 2
    rx_buffer = bytearray()
    deadline = time.monotonic() + timeout_seconds
    success = False

    try:
        connection = open_serial(
            port,
            baud,
            timeout=min(0.05, max(timeout_seconds / 20.0, 0.01)),
            clear_input=False,
        )

        while time.monotonic() < deadline:
            waiting = connection.in_waiting
            chunk = connection.read(waiting if waiting > 0 else 1)
            if not chunk:
                continue

            rx_buffer.extend(chunk)

            while True:
                delimiter = rx_buffer.find(0)
                if delimiter < 0:
                    break

                frame = bytes(rx_buffer[:delimiter])
                del rx_buffer[: delimiter + 1]

                if not frame:
                    continue

                try:
                    payload = cobs_decode(frame)
                except ValueError:
                    continue

                if _valid_telemetry_payload(payload, sample_count):
                    # The local probe buffer may contain a partial later frame.
                    # Discard it and let TelemetryReader start at a clean packet.
                    connection.reset_input_buffer()
                    connection.timeout = 0.01
                    success = True
                    return connection, "valid telemetry frame"

            if len(rx_buffer) > max_encoded_size * 3:
                rx_buffer.clear()

        return None, f"no valid frame within {timeout_seconds:.1f}s"

    except PermissionError as exc:
        return None, f"permission denied: {exc}"
    except (serial.SerialException, OSError) as exc:
        return None, str(exc)
    finally:
        # Keep the successful connection open; close every failed probe.
        if not success and connection is not None and connection.is_open:
            connection.close()


def discover_serial_connection(
    explicit_port: Optional[str],
    baud: int,
    sample_count: int,
    probe_timeout: float,
    target_vid: Optional[int] = None,
    target_pid: Optional[int] = None,
    target_serial: Optional[str] = None,
    port_match: Optional[str] = None,
    probe_all_ports: bool = False,
    strict_probe: bool = False,
) -> tuple[str, serial.Serial]:
    """Select and open the telemetry device on any supported desktop OS.

    Port discovery is intentionally separate from application-protocol
    validation. Replica ESP32 boards and USB-UART bridges may reset on open,
    start streaming slowly, or expose incomplete USB metadata. Therefore:

      * an explicit --port is always opened directly;
      * a single suitable serial candidate is opened directly by default;
      * multiple candidates are probed to disambiguate them;
      * if probing fails, a clearly superior candidate is used as a fallback;
      * --strict-probe restores fail-closed protocol validation.
    """
    if explicit_port:
        return explicit_port, open_serial(explicit_port, baud)

    ports = list_serial_ports()
    if not ports:
        raise RuntimeError(
            "No serial ports were found. Check the USB data cable, driver, "
            "device power, and operating-system permissions."
        )

    filtered = [
        port
        for port in ports
        if _matches_port_filters(
            port,
            target_vid=target_vid,
            target_pid=target_pid,
            target_serial=target_serial,
            port_match=port_match,
        )
    ]

    if not filtered:
        available = "\n".join(f"  {format_port(port)}" for port in ports)
        raise RuntimeError(
            "No serial port matched the requested filters.\n\n"
            f"Available ports:\n{available}"
        )

    if not probe_all_ports:
        non_unwanted = [port for port in filtered if not is_unwanted_port(port)]
        if non_unwanted:
            filtered = non_unwanted

    ranked = sorted(
        filtered,
        key=lambda port: (-score_port(port), port.device.casefold()),
    )

    # If there is only one usable serial interface, discovery has already done
    # its job. Do not reject a replica board merely because it did not emit a
    # complete application frame during a short probe window.
    if len(ranked) == 1 and not strict_probe:
        candidate = ranked[0]
        print(
            f"Selected {candidate.device}: "
            f"{candidate.description or 'unknown serial device'} "
            "(only suitable candidate; protocol validation deferred to reader)"
        )
        return candidate.device, open_serial(candidate.device, baud)

    failures = []
    for candidate in ranked:
        print(
            f"Probing {candidate.device}: {candidate.description or 'unknown device'}"
        )
        connection, reason = probe_telemetry_port(
            port=candidate.device,
            baud=baud,
            sample_count=sample_count,
            timeout_seconds=probe_timeout,
        )
        if connection is not None:
            return candidate.device, connection
        failures.append((candidate, reason))

    # Metadata fallback for clone boards that reset on open or need more time
    # than the probe window. Only use it when the winner is unambiguous.
    if not strict_probe and ranked:
        best = ranked[0]
        best_score = score_port(best)
        second_score = score_port(ranked[1]) if len(ranked) > 1 else -10_000
        filters_used = any(
            value is not None
            for value in (target_vid, target_pid, target_serial, port_match)
        )
        clearly_best = best_score >= 35 and (best_score - second_score) >= 20

        if filters_used or clearly_best:
            print(
                f"Warning: no candidate produced a valid telemetry frame within "
                f"{probe_timeout:.1f}s. Falling back to {best.device} based on "
                "USB metadata. Telemetry validation will continue in the reader."
            )
            return best.device, open_serial(best.device, baud)

    diagnostics = "\n".join(
        f"  {format_port(port)}\n      probe: {reason}" for port, reason in failures
    )
    raise RuntimeError(
        "Serial ports were found, but none emitted a valid telemetry packet and "
        "no unique metadata-based fallback was safe.\n"
        "Confirm that the firmware is streaming, the baud rate and --samples "
        "match the firmware, or select the port explicitly.\n\n"
        f"Candidates:\n{diagnostics}\n\n"
        "Use --port COMx to select a device directly, or omit --strict-probe."
    )


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
        auto_wave_x: bool,
        auto_wave_y: bool,
        wave_y_min: float,
        wave_y_max: float,
        wave_y_percentile: float,
        wave_y_headroom: float,
        wave_y_release: float,
        n_fft: int,
        hop_length: int,
        floor_db: float,
        ceiling_db: float,
        refresh_ms: int,
        local_model: Optional[str] = None,
        enable_gradcam: bool = False,
        enable_dense: bool = False,
        export_anomalies: bool = False,
        anomaly_dir: str = "output/misclassifications",
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

        # Waveform-axis behavior. X grows with the amount of received audio
        # until the configured rolling history is full. Y uses a robust peak
        # estimate, expands immediately, and contracts gradually to avoid
        # distracting axis flicker.
        self.auto_wave_x = auto_wave_x
        self.auto_wave_y = auto_wave_y
        self.wave_y_min = wave_y_min
        self.wave_y_max = wave_y_max
        self.wave_y_percentile = wave_y_percentile
        self.wave_y_headroom = wave_y_headroom
        self.wave_y_release = wave_y_release
        self.current_wave_y_limit = min(max(0.05, wave_y_min), wave_y_max)
        self.valid_live_wave_samples = 0

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.floor_db = floor_db
        self.ceiling_db = ceiling_db
        self.refresh_ms = refresh_ms
        self.enable_gradcam = enable_gradcam
        self.enable_dense = enable_dense or (local_model is not None)

        self.host_analyzer = None
        if HostAnalyzer is not None and (
            local_model or enable_gradcam or export_anomalies
        ):
            self.host_analyzer = HostAnalyzer(
                model_path=local_model,
                sample_rate=sample_rate,
                enable_gradcam=enable_gradcam,
                enable_dense=self.enable_dense,
                export_anomalies=export_anomalies,
                anomaly_output_dir=anomaly_dir,
            )
            self.host_analyzer.start()

        self.freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.fs)
        self.spec_columns = max(
            50, int(round(self.history_seconds * self.fs / self.hop_length))
        )
        self.spec_matrix = np.full(
            (len(self.freqs), self.spec_columns),
            self.floor_db,
            dtype=np.float32,
        )
        self.gradcam_matrix = np.zeros(
            (len(self.freqs), self.spec_columns),
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

        # Visual annotation throttling prevents overlapping labels from turning
        # the spectrogram into a wall of boxes during sustained detections.
        self.last_annotation_class: Optional[str] = None
        self.last_annotation_at = 0.0

        self._build_figure()

    def _style_plot_axis(self, axis, title: str) -> None:
        """Apply one quiet, consistent instrument-panel visual style."""
        axis.set_facecolor(self.ui["panel"])
        axis.set_title(
            title,
            loc="left",
            color=self.ui["text"],
            fontsize=11,
            fontweight="medium",
            pad=10,
        )
        axis.tick_params(colors=self.ui["muted"], labelsize=8)
        axis.xaxis.label.set_color(self.ui["muted"])
        axis.yaxis.label.set_color(self.ui["muted"])
        axis.grid(True, color=self.ui["grid"], linewidth=0.65, alpha=0.70)
        for spine in axis.spines.values():
            spine.set_color(self.ui["border"])
            spine.set_linewidth(0.8)

    def _build_figure(self) -> None:
        self.ui = {
            "canvas": "#F8F9FA",
            "panel": "#FFFFFF",
            "panel_alt": "#F1F3F5",
            "border": "#DEE2E6",
            "grid": "#E9ECEF",
            "text": "#212529",
            "muted": "#6C757D",
            "accent": "#0D6EFD",
            "success": "#198754",
            "warning": "#F5A623",
            "danger": "#DC3545",
        }

        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 10,
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
            }
        )

        self.fig = plt.figure(figsize=(16, 9), facecolor=self.ui["canvas"])
        grid = self.fig.add_gridspec(
            3,
            2,
            height_ratios=[0.55, 1.75, 1.0],
            width_ratios=[3.0, 1.35],
            left=0.05,
            right=0.965,
            bottom=0.08,
            top=0.96,
            hspace=0.28,
            wspace=0.22,
        )

        self.ax_wave = self.fig.add_subplot(grid[0, 0])
        if self.enable_gradcam:
            self.ax_spec = self.fig.add_subplot(grid[1, 0])
            self.ax_gradcam = self.fig.add_subplot(grid[2, 0])
        else:
            self.ax_spec = self.fig.add_subplot(grid[1:3, 0])
            self.ax_gradcam = None

        self.ax_info = self.fig.add_subplot(grid[0, 1])

        if self.enable_dense:
            self.ax_probs = self.fig.add_subplot(grid[1, 1])
            self.ax_emb = self.fig.add_subplot(grid[2, 1])
        else:
            self.ax_probs = self.fig.add_subplot(grid[1:3, 1])
            self.ax_emb = None

        self.fig.canvas.manager.set_window_title("Telemetry")

        # Header: one stable title and one compact state indicator. The plots no
        # longer flash or resize when the predicted class changes.
        self.fig.text(
            0.055,
            0.944,
            "",
            color=self.ui["text"],
            fontsize=18,
            fontweight="medium",
            ha="left",
            va="center",
        )
        self.fig.text(
            0.055,
            0.912,
            "",
            color=self.ui["muted"],
            fontsize=9.5,
            ha="left",
            va="center",
        )
        self.header_status = self.fig.text(
            0.955,
            0.928,
            "WAITING",
            color=self.ui["muted"],
            fontsize=9,
            fontweight="medium",
            ha="right",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.38,rounding_size=0.9",
                "facecolor": self.ui["panel_alt"],
                "edgecolor": self.ui["border"],
                "linewidth": 0.8,
            },
        )
        self.header_status.set_visible(False)
        # Header divider removed for compact layout

        # Waveform
        self.wave_time = (
            np.arange(self.live_wave_sample_count, dtype=np.float32)
            - self.live_wave_sample_count
            + 1
        ) / self.fs
        (self.wave_line,) = self.ax_wave.plot(
            self.wave_time,
            self.live_wave_buffer,
            color=self.ui["muted"],
            linewidth=1.05,
            solid_capstyle="round",
        )
        self.ax_wave.set_xlim(self.wave_time[0], 0.0)
        self.ax_wave.set_ylim(-self.current_wave_y_limit, self.current_wave_y_limit)
        self._style_plot_axis(self.ax_wave, "Waveform")
        self.ax_wave.set_xlabel("Time (s)")
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.axhline(0.0, color=self.ui["border"], linewidth=0.75)

        # Spectrogram
        self.spec_image = self.ax_spec.imshow(
            self.spec_matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[-self.history_seconds, 0.0, 0.0, self.fs / 2.0],
            cmap="magma",
            vmin=self.floor_db,
            vmax=self.ceiling_db,
        )
        self.ax_spec.set_ylim(self.min_frequency, self.max_frequency)
        self._style_plot_axis(self.ax_spec, "Frequency history")
        self.ax_spec.set_xlabel("History (s)")
        self.ax_spec.set_ylabel("Frequency (Hz)")
        colorbar = self.fig.colorbar(
            self.spec_image,
            ax=self.ax_spec,
            pad=0.012,
            fraction=0.024,
        )
        colorbar.set_label("dBFS", color=self.ui["muted"], fontsize=8)
        colorbar.ax.tick_params(colors=self.ui["muted"], labelsize=7)
        colorbar.outline.set_edgecolor(self.ui["border"])

        # Grad-CAM uses a perceptually ordered map instead of rainbow/jet.
        self.gradcam_image = None
        if self.ax_gradcam is not None:
            self.gradcam_image = self.ax_gradcam.imshow(
                self.gradcam_matrix,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=[-self.history_seconds, 0.0, 0.0, self.fs / 2.0],
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            self.ax_gradcam.set_ylim(self.min_frequency, self.max_frequency)
            self._style_plot_axis(self.ax_gradcam, "Model attention")
            self.ax_gradcam.set_xlabel("History (s)")
            self.ax_gradcam.set_ylabel("Frequency (Hz)")
            cam_bar = self.fig.colorbar(
                self.gradcam_image,
                ax=self.ax_gradcam,
                pad=0.012,
                fraction=0.024,
            )
            cam_bar.set_label("Attention", color=self.ui["muted"], fontsize=8)
            cam_bar.ax.tick_params(colors=self.ui["muted"], labelsize=7)
            cam_bar.outline.set_edgecolor(self.ui["border"])

        # Right information panel
        self.ax_info.set_facecolor(self.ui["panel"])
        self.ax_info.set_xlim(0.0, 1.0)
        self.ax_info.set_ylim(0.0, 1.0)
        self.ax_info.set_xticks([])
        self.ax_info.set_yticks([])
        for spine in self.ax_info.spines.values():
            spine.set_color(self.ui["border"])
            spine.set_linewidth(0.8)

        self.result_name = self.ax_info.text(
            0.05,
            0.88,
            "Waiting for data",
            color=self.ui["text"],
            fontsize=12,
            fontweight="medium",
            ha="left",
            va="top",
            wrap=True,
        )
        self.result_confidence = self.ax_info.text(
            0.05,
            0.55,
            "—",
            color=self.ui["muted"],
            fontsize=16,
            fontweight="medium",
            ha="left",
            va="top",
        )
        self.result_badge = self.ax_info.text(
            0.05,
            0.20,
            "NO TELEMETRY",
            color=self.ui["muted"],
            fontsize=7.5,
            fontweight="medium",
            ha="left",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.3,rounding_size=0.8",
                "facecolor": self.ui["panel_alt"],
                "edgecolor": self.ui["border"],
                "linewidth": 0.8,
            },
        )
        self.result_badge.set_visible(False)

        self.wave_info = self.ax_info.text(
            0.55,
            0.88,
            "DEVICE\nWaiting…",
            color=self.ui["text"],
            fontsize=7.5,
            linespacing=1.2,
            ha="left",
            va="top",
        )
        self.host_info = self.ax_info.text(
            0.55,
            0.35,
            (
                "HOST MODEL\nNot enabled"
                if self.host_analyzer is None
                else "HOST MODEL\nLoaded"
                if getattr(self.host_analyzer, "model_loaded", False)
                else "HOST MODEL\nError"
            ),
            color=self.ui["muted"],
            fontsize=7.5,
            linespacing=1.2,
            ha="left",
            va="top",
        )
        self.host_badge = self.ax_info.text(
            0.55,
            0.08,
            (
                "HOST OFF"
                if self.host_analyzer is None
                else "MODEL LOADED"
                if getattr(self.host_analyzer, "model_loaded", False)
                else "MODEL ERROR"
                if getattr(self.host_analyzer, "model_path", None)
                else "ANALYZER READY"
            ),
            color=self.ui["muted"],
            fontsize=7.0,
            fontweight="medium",
            ha="left",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.25,rounding_size=0.8",
                "facecolor": self.ui["panel_alt"],
                "edgecolor": self.ui["border"],
                "linewidth": 0.8,
            },
        )

        # Class probabilities horizontal bar chart subplot
        self._style_plot_axis(self.ax_probs, "Class probabilities")
        self.ax_probs.set_xlim(0, 115)
        self.ax_probs.set_ylim(-0.7, 10.7)
        y_pos = np.arange(10, -1, -1)
        self.ax_probs.set_yticks(y_pos)
        self.ax_probs.set_yticklabels(
            [self._short_class_label(CLASS_NAMES[i]) for i in range(11)],
            color=self.ui["text"],
            fontsize=8.5,
        )
        self.ax_probs.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x)}%")
        )

        bar_height = 0.35
        self.mcu_bars = self.ax_probs.barh(
            y_pos + bar_height / 2,
            np.zeros(11),
            height=bar_height,
            color=self.ui["accent"],
            label="MCU",
            alpha=0.88,
        )
        self.host_bars = self.ax_probs.barh(
            y_pos - bar_height / 2,
            np.zeros(11),
            height=bar_height,
            color=self.ui["success"],
            label="Host",
            alpha=0.88,
        )
        for bar in self.host_bars:
            bar.set_visible(False)

        self.mcu_bar_texts = [
            self.ax_probs.text(
                1.5,
                y_pos[i] + bar_height / 2,
                "",
                va="center",
                ha="left",
                fontsize=7.5,
                color=self.ui["accent"],
                fontweight="bold",
            )
            for i in range(11)
        ]
        self.host_bar_texts = [
            self.ax_probs.text(
                1.5,
                y_pos[i] - bar_height / 2,
                "",
                va="center",
                ha="left",
                fontsize=7.5,
                color=self.ui["success"],
                fontweight="bold",
            )
            for i in range(11)
        ]
        for t in self.host_bar_texts:
            t.set_visible(False)

        self.ax_probs.legend(
            loc="lower right",
            facecolor=self.ui["panel_alt"],
            edgecolor=self.ui["border"],
            fontsize=7.5,
            labelcolor=self.ui["text"],
            framealpha=0.85,
        )

        # Dense embedding panel
        self.emb_line = None
        if self.ax_emb is not None:
            (self.emb_line,) = self.ax_emb.plot(
                np.arange(256),
                np.zeros(256),
                color=self.ui["accent"],
                linewidth=1.0,
            )
            self.ax_emb.set_xlim(0, 255)
            self.ax_emb.set_ylim(-2.0, 2.0)
            self._style_plot_axis(self.ax_emb, "Dense embedding")
            self.ax_emb.set_xlabel("Neuron")
            self.ax_emb.set_ylabel("Activation")

        # Quiet footer: transport health only. Detailed values stay in the side panel.
        self.fig.add_artist(
            patches.Rectangle(
                (0.055, 0.071),
                0.91,
                0.001,
                transform=self.fig.transFigure,
                facecolor=self.ui["border"],
                edgecolor="none",
            )
        )
        self.status_text = self.fig.text(
            0.055,
            0.043,
            "Connecting…",
            color=self.ui["muted"],
            ha="left",
            va="center",
            fontsize=8.5,
        )

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

    def _append_spectrogram(self, audio: np.ndarray) -> np.ndarray:
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
            return new_columns

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
        return newest

    def _append_gradcam(
        self, heatmap: np.ndarray, stft_cols: Optional[np.ndarray] = None
    ) -> None:
        if not self.enable_gradcam or self.gradcam_image is None:
            return

        cols_per_packet = max(1, int(round(self.packet_hop_samples / self.hop_length)))
        if heatmap.ndim == 1:
            h_interp = np.interp(
                np.linspace(0, len(heatmap) - 1, cols_per_packet),
                np.arange(len(heatmap)),
                heatmap,
            )
            if stft_cols is not None and stft_cols.shape[1] >= cols_per_packet:
                # STFT frequency spectral magnitude * temporal Grad-CAM attention (PSD fashion)
                stft_new = stft_cols[:, -cols_per_packet:]
                norm_stft = np.clip(
                    (stft_new - self.floor_db) / (self.ceiling_db - self.floor_db),
                    0.0,
                    1.0,
                )
                slice_2d = norm_stft * h_interp[np.newaxis, :]
            else:
                slice_2d = np.tile(h_interp, (len(self.freqs), 1))
        elif heatmap.ndim == 2:
            slice_2d = heatmap
        else:
            return

        if cols_per_packet >= self.spec_columns:
            self.gradcam_matrix[:, :] = slice_2d[:, -self.spec_columns :]
        else:
            self.gradcam_matrix[:, :-cols_per_packet] = self.gradcam_matrix[
                :, cols_per_packet:
            ]
            self.gradcam_matrix[:, -cols_per_packet:] = slice_2d

        self.gradcam_image.set_data(self.gradcam_matrix)

    def _update_waveform_axes(self) -> None:
        """Update waveform limits from the valid portion of the rolling data."""
        valid_count = min(
            self.valid_live_wave_samples,
            self.live_wave_sample_count,
        )
        if valid_count <= 0:
            return

        if self.auto_wave_x:
            visible_seconds = min(
                self.live_wave_seconds,
                valid_count / self.fs,
            )
            # Avoid a zero-width axis if an unusually short packet arrives.
            visible_seconds = max(visible_seconds, 2.0 / self.fs)
            self.ax_wave.set_xlim(-visible_seconds, 0.0)

        if self.auto_wave_y:
            current = self.live_wave_buffer[-valid_count:]
            finite = current[np.isfinite(current)]
            if finite.size:
                absolute = np.abs(finite)
                robust_peak = float(np.percentile(absolute, self.wave_y_percentile))
                target_limit = robust_peak * self.wave_y_headroom
                target_limit = float(
                    np.clip(target_limit, self.wave_y_min, self.wave_y_max)
                )

                # Fast attack: never clip a newly stronger signal because of
                # smoothing. Slow release: quiet periods shrink the plot
                # gradually instead of making the Y axis jump every packet.
                if target_limit >= self.current_wave_y_limit:
                    self.current_wave_y_limit = target_limit
                else:
                    self.current_wave_y_limit = (
                        self.wave_y_release * self.current_wave_y_limit
                        + (1.0 - self.wave_y_release) * target_limit
                    )

                self.current_wave_y_limit = float(
                    np.clip(
                        self.current_wave_y_limit,
                        self.wave_y_min,
                        self.wave_y_max,
                    )
                )
                self.ax_wave.set_ylim(
                    -self.current_wave_y_limit,
                    self.current_wave_y_limit,
                )

    def _append_live_waveform(self, audio: np.ndarray) -> None:
        # Preserve the complete first inference window. Once initialized,
        # append only the new hop so overlapping windows are not duplicated.
        if self.valid_live_wave_samples == 0:
            newest = audio[-self.live_wave_sample_count :]
        else:
            newest = audio[-self.packet_hop_samples :]

        count = min(len(newest), self.live_wave_sample_count)
        if count <= 0:
            return

        if count >= self.live_wave_sample_count:
            self.live_wave_buffer[:] = newest[-self.live_wave_sample_count :]
        else:
            self.live_wave_buffer[:-count] = self.live_wave_buffer[count:]
            self.live_wave_buffer[-count:] = newest[-count:]

        self.valid_live_wave_samples = min(
            self.live_wave_sample_count,
            self.valid_live_wave_samples + count,
        )
        self.wave_line.set_ydata(self.live_wave_buffer)
        self._update_waveform_axes()

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
            return "No Mos"

        parts = class_name.split("_")
        if len(parts) >= 3:
            genus = parts[0][:2]
            species = parts[1]
            sex = "♀" if parts[2].startswith("F") else "♂"
            return f"{genus}. {species} {sex}"
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
        """Draw a clear, bold bounding box outline over the spectrogram segment."""
        low_f, high_f = self._detection_band(class_name, peak_frequency)
        box_width = max(self.window_seconds, self.packet_advance_seconds * 2.0)
        x0 = -box_width

        rect = patches.Rectangle(
            (x0, low_f),
            box_width,
            high_f - low_f,
            linewidth=2.2,
            edgecolor=color,
            facecolor=color,
            alpha=0.15,
            zorder=10,
        )
        self.ax_spec.add_patch(rect)

        label = f" {self._short_class_label(class_name)} · {confidence * 100:.0f}% "
        text = self.ax_spec.text(
            x0 + 0.01,
            high_f - 15.0,
            label,
            color="#FFFFFF",
            fontsize=8.0,
            fontweight="bold",
            ha="left",
            va="top",
            zorder=11,
            bbox={
                "boxstyle": "round,pad=0.25,rounding_size=0.3",
                "facecolor": color,
                "alpha": 0.90,
                "edgecolor": "#FFFFFF",
                "linewidth": 0.8,
            },
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
        display_name = class_name.replace("_", " ")
        if class_name == "No_Mos":
            display_name = "No mosquito"

        class_tint = class_color(class_name)
        audio = packet.audio_i16.astype(np.float32) / 32768.0

        self._age_spec_annotations()

        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio)))
        peak_frequency = self._peak_frequency(audio)

        is_detection = (
            class_name not in ("No_Mos", "Unknown")
            and packet.confidence >= self.detection_threshold
        )

        if is_detection:
            state_label = "DETECTED"
            state_color = class_tint
            state_fill = "#1B2731"
        elif class_name == "No_Mos":
            state_label = "NO DETECTION"
            state_color = self.ui["muted"]
            state_fill = self.ui["panel_alt"]
        else:
            state_label = "LOW CONFIDENCE"
            state_color = self.ui["warning"]
            state_fill = "#2A2418"

        self._append_live_waveform(audio)
        self.wave_line.set_color(state_color)
        self.ax_wave.spines["left"].set_color(state_color)
        self.ax_wave.spines["left"].set_linewidth(1.8)

        self.header_status.set_text(state_label)
        self.header_status.set_color(state_color)
        self.header_status.get_bbox_patch().set_facecolor(state_fill)
        self.header_status.get_bbox_patch().set_edgecolor(state_color)

        self.result_name.set_text(display_name)
        self.result_name.set_color(state_color if is_detection else self.ui["text"])
        self.result_confidence.set_text(f"{packet.confidence * 100.0:.1f}%")
        self.result_confidence.set_color(state_color)
        self.result_badge.set_text(state_label)
        self.result_badge.set_color(state_color)
        self.result_badge.get_bbox_patch().set_facecolor(state_fill)
        self.result_badge.get_bbox_patch().set_edgecolor(state_color)

        infer_ms = packet.inference_time_us / 1000.0
        class_age = (
            "Unavailable"
            if packet.class_age_ms == 0xFFFFFFFF
            else f"{packet.class_age_ms} ms"
        )
        self.wave_info.set_text(
            "DEVICE\n"
            f"Inference       {infer_ms:.2f} ms\n"
            f"Prediction age  {class_age}\n"
            f"RMS / peak      {rms:.4f} / {peak:.4f}\n"
            f"Dominant freq.  {peak_frequency:.1f} Hz\n"
            f"Packet          {packet.seq} · classifier {packet.classifier_seq}"
        )
        self.wave_info.set_color(self.ui["text"])

        if (
            hasattr(packet, "class_probability")
            and packet.class_probability
            and len(packet.class_probability) == len(CLASS_NAMES)
        ):
            mcu_p = np.array(packet.class_probability) * 100.0
            host_p = getattr(self, "latest_host_probabilities", None)
            has_host = (
                host_p is not None
                and len(host_p) == len(CLASS_NAMES)
            )

            if has_host:
                host_p_pct = np.array(host_p) * 100.0
            else:
                host_p_pct = np.zeros(11)

            y_pos = np.arange(10, -1, -1)
            bar_height = 0.35 if has_host else 0.55

            for i in range(11):
                m_offset = bar_height / 2 if has_host else 0
                self.mcu_bars[i].set_width(mcu_p[i])
                self.mcu_bars[i].set_y(y_pos[i] + m_offset - bar_height / 2)
                self.mcu_bars[i].set_height(bar_height)

                val_text = f"{mcu_p[i]:.1f}%" if mcu_p[i] >= 0.5 else ""
                self.mcu_bar_texts[i].set_text(val_text)
                self.mcu_bar_texts[i].set_x(mcu_p[i] + 1.5)
                self.mcu_bar_texts[i].set_y(y_pos[i] + m_offset)

                if has_host:
                    hp = host_p_pct[i]
                    self.host_bars[i].set_width(hp)
                    self.host_bars[i].set_y(y_pos[i] - bar_height / 2 - bar_height / 2)
                    self.host_bars[i].set_height(bar_height)
                    self.host_bars[i].set_visible(True)

                    host_val_text = f"{hp:.1f}%" if hp >= 0.5 else ""
                    self.host_bar_texts[i].set_text(host_val_text)
                    self.host_bar_texts[i].set_x(hp + 1.5)
                    self.host_bar_texts[i].set_y(y_pos[i] - bar_height / 2)
                    self.host_bar_texts[i].set_visible(True)
                else:
                    self.host_bars[i].set_visible(False)
                    self.host_bar_texts[i].set_visible(False)

        self.latest_stft = self._append_spectrogram(audio)
        if is_detection:
            annotation_interval = max(0.65, self.packet_advance_seconds * 3.0)
            should_annotate = (
                class_name != self.last_annotation_class
                or packet.received_at - self.last_annotation_at >= annotation_interval
            )
            if should_annotate:
                self._add_spec_box(
                    class_name=class_name,
                    confidence=packet.confidence,
                    color=class_tint,
                    peak_frequency=peak_frequency,
                )
                self.last_annotation_class = class_name
                self.last_annotation_at = packet.received_at

        if self.host_analyzer is not None:
            self.host_analyzer.submit_packet(
                seq=packet.seq,
                mcu_class_id=packet.class_id,
                mcu_confidence=packet.confidence,
                audio_i16=packet.audio_i16,
                received_at=packet.received_at,
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
            f"{self.port}  ·  {self.baud / 1_000_000:.1f} Mbaud  ·  "
            f"{self.fs:,} Hz / {self.window_seconds * 1000.0:.0f} ms  ·  "
            f"{self.packets_per_second:.2f} pkt/s  ·  "
            f"{self.bytes_per_second / 1024.0:.1f} KiB/s  ·  "
            f"packets {stats.valid_packets:,}  ·  dropped {stats.queue_drops}  ·  "
            f"decode errors {stats.cobs_errors + stats.length_errors}  ·  last {age_text}"
        )

        if (
            self.host_analyzer is not None
            and not self.host_analyzer.output_queue.empty()
        ):
            try:
                res = self.host_analyzer.output_queue.get_nowait()
                if res.host_class_probability is not None:
                    self.latest_host_probabilities = res.host_class_probability
                if res.heatmap is not None:
                    self._append_gradcam(
                        res.heatmap,
                        getattr(self, "latest_stft", None),
                    )
                if self.emb_line is not None and res.dense_embedding is not None:
                    emb = res.dense_embedding
                    if len(self.emb_line.get_xdata()) != len(emb):
                        self.emb_line.set_xdata(np.arange(len(emb)))
                        self.ax_emb.set_xlim(0, max(1, len(emb) - 1))
                    self.emb_line.set_ydata(emb)
                    ymin = float(np.min(emb))
                    ymax = float(np.max(emb))
                    if ymax > ymin:
                        margin = max(0.15, 0.08 * (ymax - ymin))
                        self.ax_emb.set_ylim(ymin - margin, ymax + margin)

                if res.host_class_id is not None:
                    h_cls = (
                        CLASS_NAMES[res.host_class_id]
                        if 0 <= res.host_class_id < len(CLASS_NAMES)
                        else str(res.host_class_id)
                    )
                    h_display = h_cls.replace("_", " ")
                    if h_cls == "No_Mos":
                        h_display = "No mosquito"
                    clr = class_color(h_cls)

                    if self.emb_line is not None:
                        self.emb_line.set_color(clr)
                        self.ax_emb.set_title(
                            f"Dense embedding · {h_display}",
                            loc="left",
                            color=self.ui["text"],
                            fontsize=11,
                            fontweight="medium",
                            pad=10,
                        )

                    agreement = "MISMATCH" if res.discrepancy else "MATCH"
                    agreement_color = (
                        self.ui["danger"] if res.discrepancy else self.ui["success"]
                    )
                    gradcam_state = "Active" if self.enable_gradcam else "Disabled"
                    self.host_info.set_text(
                        "HOST MODEL\n"
                        f"Prediction      {h_display}\n"
                        f"Confidence      {res.host_confidence * 100.0:.1f}%\n"
                        f"MCU agreement   {agreement}\n"
                        f"Dominant freq.  {res.f0_hz:.1f} Hz\n"
                        f"Grad-CAM        {gradcam_state}"
                    )
                    self.host_info.set_color(self.ui["text"])
                    self.host_badge.set_text(agreement)
                    self.host_badge.set_color(agreement_color)
                    self.host_badge.get_bbox_patch().set_facecolor(
                        "#2A1B20" if res.discrepancy else "#172720"
                    )
                    self.host_badge.get_bbox_patch().set_edgecolor(agreement_color)
            except queue.Empty:
                pass

        if stats.last_error:
            self.header_status.set_text("SERIAL ERROR")
            self.header_status.set_color(self.ui["danger"])
            self.header_status.get_bbox_patch().set_facecolor("#2A1B20")
            self.header_status.get_bbox_patch().set_edgecolor(self.ui["danger"])
            status += f"  ·  serial error: {stats.last_error}"

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
    parser.add_argument(
        "--port",
        help=(
            "Serial port override, for example COM6, /dev/ttyUSB0, "
            "or /dev/cu.usbserial-110"
        ),
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List detected serial ports with USB metadata and exit.",
    )
    parser.add_argument(
        "--vid",
        type=parse_usb_id,
        default=None,
        help="Optional USB vendor ID filter, for example 0x303A, 10C4, or 1A86.",
    )
    parser.add_argument(
        "--pid",
        type=parse_usb_id,
        default=None,
        help="Optional USB product ID filter.",
    )
    parser.add_argument(
        "--usb-serial",
        default=None,
        help="Optional exact USB serial-number filter.",
    )
    parser.add_argument(
        "--port-match",
        default=None,
        help=(
            "Case-insensitive substring matched against port path, product, "
            "manufacturer, hardware ID, serial number, and location."
        ),
    )
    parser.add_argument(
        "--probe-timeout",
        type=float,
        default=3.0,
        help="Seconds to wait for a valid telemetry frame from each candidate.",
    )
    parser.add_argument(
        "--probe-all-ports",
        action="store_true",
        help="Also probe ports identified as Bluetooth, infrared, or modem devices.",
    )
    parser.add_argument(
        "--strict-probe",
        action="store_true",
        help=(
            "Require a valid telemetry frame during discovery. By default, a "
            "single or clearly ranked USB serial device is allowed as a fallback."
        ),
    )
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--fs", type=int, default=8_000)
    parser.add_argument("--samples", type=int, default=2_400)
    parser.add_argument("--history", type=float, default=5.0)
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
            "half the window for the current firmware's 50%% overlap."
        ),
    )
    parser.add_argument(
        "--live-wave-seconds",
        type=float,
        default=2.0,
        help="Maximum seconds of rolling waveform to show in the live view.",
    )
    parser.add_argument(
        "--fixed-wave-x",
        action="store_true",
        help=(
            "Keep the waveform X axis fixed at --live-wave-seconds instead of "
            "growing with the amount of received audio during startup."
        ),
    )
    parser.add_argument(
        "--fixed-wave-y",
        action="store_true",
        help="Disable automatic waveform Y-axis scaling.",
    )
    parser.add_argument(
        "--wave-y-min",
        type=float,
        default=0.005,
        help="Minimum automatic symmetric waveform Y limit.",
    )
    parser.add_argument(
        "--wave-y-max",
        type=float,
        default=1.0,
        help="Maximum automatic symmetric waveform Y limit.",
    )
    parser.add_argument(
        "--wave-y-percentile",
        type=float,
        default=99.8,
        help=(
            "Absolute-amplitude percentile used for automatic Y scaling. "
            "Use 100 to include the exact maximum sample."
        ),
    )
    parser.add_argument(
        "--wave-y-headroom",
        type=float,
        default=1.15,
        help="Headroom multiplier applied to the automatic Y limit.",
    )
    parser.add_argument(
        "--wave-y-release",
        type=float,
        default=0.90,
        help=(
            "Y-axis contraction smoothing in [0, 1). Higher values contract "
            "more slowly and reduce visual flicker."
        ),
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
    parser.add_argument("--refresh-ms", type=int, default=50)
    parser.add_argument(
        "--local-model",
        type=str,
        default=None,
        help="Path to local Keras/TF model for host-side model analysis.",
    )
    parser.add_argument(
        "--enable-gradcam",
        action="store_true",
        help="Enable live Grad-CAM explainability heatmap computation.",
    )
    parser.add_argument(
        "--enable-dense",
        action="store_true",
        help="Enable live 256-dim dense embedding feature visualization panel.",
    )
    parser.add_argument(
        "--export-anomalies",
        action="store_true",
        help="Automatically export misclassifications and low-confidence frames.",
    )
    parser.add_argument(
        "--anomaly-dir",
        type=str,
        default="output/misclassifications",
        help="Output directory for saved anomaly frames.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_ports:
        ports = list_serial_ports()
        if not ports:
            print("No serial ports found.")
            return 1

        print("Detected serial ports:")
        for detected_port in sorted(
            ports,
            key=lambda port: (-score_port(port), port.device.casefold()),
        ):
            print(f"  {format_port(detected_port)}")
        return 0

    if args.baud <= 0:
        raise ValueError("--baud must be positive")
    if args.fs <= 0 or args.samples <= 0:
        raise ValueError("--fs and --samples must be positive")
    if args.probe_timeout <= 0:
        raise ValueError("--probe-timeout must be positive")
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
    if args.wave_y_min <= 0:
        raise ValueError("--wave-y-min must be positive")
    if args.wave_y_max <= args.wave_y_min:
        raise ValueError("--wave-y-max must be greater than --wave-y-min")
    if not 0.0 < args.wave_y_percentile <= 100.0:
        raise ValueError("--wave-y-percentile must be in (0, 100]")
    if args.wave_y_headroom <= 1.0:
        raise ValueError("--wave-y-headroom must be greater than 1")
    if not 0.0 <= args.wave_y_release < 1.0:
        raise ValueError("--wave-y-release must be in [0, 1)")
    if not 0.0 <= args.min_frequency < args.max_frequency <= args.fs / 2:
        raise ValueError("frequency limits must satisfy 0 <= min < max <= Nyquist")

    packet_hop_samples = (
        args.samples // 2
        if args.packet_hop_samples is None
        else args.packet_hop_samples
    )
    if not 0 < packet_hop_samples <= args.samples:
        raise ValueError("--packet-hop-samples must be between 1 and --samples")

    port, serial_port = discover_serial_connection(
        explicit_port=args.port,
        baud=args.baud,
        sample_count=args.samples,
        probe_timeout=args.probe_timeout,
        target_vid=args.vid,
        target_pid=args.pid,
        target_serial=args.usb_serial,
        port_match=args.port_match,
        probe_all_ports=args.probe_all_ports,
        strict_probe=args.strict_probe,
    )
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
        auto_wave_x=not args.fixed_wave_x,
        auto_wave_y=not args.fixed_wave_y,
        wave_y_min=args.wave_y_min,
        wave_y_max=args.wave_y_max,
        wave_y_percentile=args.wave_y_percentile,
        wave_y_headroom=args.wave_y_headroom,
        wave_y_release=args.wave_y_release,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        floor_db=args.floor_db,
        ceiling_db=args.ceiling_db,
        refresh_ms=args.refresh_ms,
        local_model=args.local_model,
        enable_gradcam=args.enable_gradcam,
        enable_dense=args.enable_dense,
        export_anomalies=args.export_anomalies,
        anomaly_dir=args.anomaly_dir,
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
