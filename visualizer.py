from __future__ import annotations

import argparse
import queue
import re
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
        (
            _seq,
            _audio_timestamp_us,
            class_id,
            confidence,
            _inference_time_us,
            _class_age_ms,
            _classifier_seq,
        ) = struct.unpack_from(HEADER_FORMAT, payload, 0)
    except struct.error:
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
        print(f"Probing {candidate.device}: {candidate.description or 'unknown device'}")
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
        f"  {format_port(port)}\n      probe: {reason}"
        for port, reason in failures
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
        self.wave_time = (
            np.arange(self.live_wave_sample_count, dtype=np.float32)
            - self.live_wave_sample_count
            + 1
        ) / self.fs
        (self.wave_line,) = self.ax_wave.plot(
            self.wave_time,
            self.live_wave_buffer,
            linewidth=0.9,
        )
        self.ax_wave.set_xlim(self.wave_time[0], 0.0)
        self.ax_wave.set_ylim(
            -self.current_wave_y_limit,
            self.current_wave_y_limit,
        )
        axis_mode = []

        mode_suffix = f" ({', '.join(axis_mode)})" if axis_mode else ""
        self.ax_wave.set_title(f"Live rolling waveform{mode_suffix}")
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
                robust_peak = float(
                    np.percentile(absolute, self.wave_y_percentile)
                )
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
            f"dominant≈{peak_frequency:.1f} Hz\n"
            f"wave_view={min(self.valid_live_wave_samples / self.fs, self.live_wave_seconds):.2f}s   "
            f"Y=±{self.current_wave_y_limit:.4f}"
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
    parser.add_argument("--refresh-ms", type=int, default=20)
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
        args.samples // 2 if args.packet_hop_samples is None else args.packet_hop_samples
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
