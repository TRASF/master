import collections
import struct
import sys
import threading

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import serial

# =============================================================================
# CONFIGURATION
# =============================================================================
PORTS = ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0"]
BAUD = 2000000
FS = 8000
CHUNK_SIZE = 2400

# Packet Framing Settings
MAGIC_HEADER = b"\xaa\xbb\xcc\xdd"
HEADER_SIZE = 4
ML_PAYLOAD_SIZE = 5  # 1 byte uint8 (class) + 4 byte float32 (confidence)
AUDIO_BYTES = CHUNK_SIZE * 3  # 24-bit audio = 3 bytes per sample
TOTAL_PACKET_SIZE = HEADER_SIZE + ML_PAYLOAD_SIZE + AUDIO_BYTES

MAX_INT_VAL = 2**23
SPEC_HISTORY = 60  # Past 60 classification windows

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

# Unique colors assigned to different species families for fast tracking
CLASS_COLORS = {
    "Ae": "#00ffcc",  # Cyan
    "An": "#ff3366",  # Pink/Red
    "Cx": "#ffcc00",  # Yellow
    "No": "#555555",  # Gray for background
}


def get_color(name):
    prefix = name[:2]
    return CLASS_COLORS.get(prefix, "#ffffff")


# =============================================================================
# SERIAL INITIALIZATION
# =============================================================================
SELECTED_PORT = None
ser = None

for port in PORTS:
    try:
        ser = serial.Serial(port, BAUD, timeout=0.1)
        ser.flushInput()
        if ser.is_open:
            SELECTED_PORT = port
            break
    except Exception:
        ser = None

if SELECTED_PORT is None or ser is None:
    print("FATAL ERROR: Could not connect to any serial ports.")
    sys.exit(1)

print(f"SUCCESS: Connected to {SELECTED_PORT} at {BAUD} baud.")

# =============================================================================
# UI & DATA BUFFERS
# =============================================================================
raw_byte_pool = bytearray()
pool_lock = threading.Lock()

audio_buffer = collections.deque(np.zeros(CHUNK_SIZE * 2), maxlen=CHUNK_SIZE * 2)
spec_matrix = np.zeros((SPEC_HISTORY, CHUNK_SIZE // 2 + 1))
active_spec_boxes = []  # Keeps track of scrolling historical boxes

fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(12, 9))
fig.patch.set_facecolor("#1e1e1e")
fig.suptitle(
    "Awaiting Edge ML Stream...", color="#00ffcc", fontsize=16, fontweight="bold"
)

# --- Waveform Setup ---
(line,) = ax_wave.plot(np.zeros(CHUNK_SIZE), color="#555555", linewidth=1)
ax_wave.set_ylim(-MAX_INT_VAL, MAX_INT_VAL)
ax_wave.set_title("Time Domain Waveform (Active Inference Window)", color="white")
ax_wave.set_facecolor("#111111")
ax_wave.tick_params(colors="white")
ax_wave.grid(True, color="#222222")

# Waveform Bounding Box (highlights the whole window when a mosquito is present)
wave_box = patches.Rectangle(
    (0, -MAX_INT_VAL),
    CHUNK_SIZE,
    MAX_INT_VAL * 2,
    linewidth=2,
    edgecolor="none",
    facecolor="none",
    alpha=0.15,
)
ax_wave.add_patch(wave_box)
wave_label = ax_wave.text(
    50,
    MAX_INT_VAL * 0.75,
    "",
    color="white",
    fontweight="bold",
    bbox=dict(facecolor="black", alpha=0.8, edgecolor="none"),
)

# --- Spectrogram Setup ---
im = ax_spec.imshow(
    spec_matrix.T,
    aspect="auto",
    origin="lower",
    extent=[0, SPEC_HISTORY, 0, FS / 2],
    cmap="inferno",
    vmin=-100,
    vmax=0,
)
ax_spec.set_title(
    "Historical Scrolling Spectrogram (With ML Object Bounding Boxes)", color="white"
)
ax_spec.set_ylabel("Frequency (Hz)", color="white")
ax_spec.set_ylim(0, 2500)  # Focus visual window on mosquito wingbeat zones
ax_spec.tick_params(colors="white")


# =============================================================================
# DATA PROCESSING
# =============================================================================
def serial_reader_thread():
    global raw_byte_pool
    while True:
        try:
            data = ser.read(max(1, ser.in_waiting))
            if data:
                with pool_lock:
                    raw_byte_pool.extend(data)
        except Exception:
            break


t = threading.Thread(target=serial_reader_thread, daemon=True)
t.start()


def decode_24bit_pcm(data_bytes):
    a = np.frombuffer(data_bytes, dtype=np.uint8).reshape(-1, 3)
    padded = np.zeros((a.shape[0], 4), dtype=np.uint8)
    padded[:, :3] = a
    samples = padded.view(np.int32).flatten()
    return (samples << 8) >> 8


def update(frame):
    global raw_byte_pool, spec_matrix, active_spec_boxes

    with pool_lock:
        idx = raw_byte_pool.find(MAGIC_HEADER)
        if idx == -1:
            if len(raw_byte_pool) > TOTAL_PACKET_SIZE * 4:
                del raw_byte_pool[: -TOTAL_PACKET_SIZE * 2]
            return line, im

        if len(raw_byte_pool) >= idx + TOTAL_PACKET_SIZE:
            packet = raw_byte_pool[idx : idx + TOTAL_PACKET_SIZE]
            del raw_byte_pool[: idx + TOTAL_PACKET_SIZE]
        else:
            return line, im

    # 1. Unpack Telemetry Payload
    class_id = packet[4]
    confidence = struct.unpack("<f", packet[5:9])[0]
    class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "Unknown"
    color = get_color(class_name)

    # 2. Shift Existing Spectrogram Bounding Boxes Left
    for rect, text in list(active_spec_boxes):
        new_x = rect.get_x() - 1
        if new_x < -1:
            rect.remove()
            text.remove()
            active_spec_boxes.remove((rect, text))
        else:
            rect.set_x(new_x)
            text.set_x(new_x + 0.1)

    # 3. Handle Active Waveform Highlights
    if class_name != "No_Mos":
        fig.suptitle(
            f"DETECTED: {class_name} | Confidence: {confidence * 100:.1f}%",
            color=color,
            fontsize=18,
            fontweight="bold",
        )
        line.set_color(color)
        wave_box.set_edgecolor(color)
        wave_box.set_facecolor(color)
        wave_label.set_text(f"{class_name} ({confidence * 100:.1f}%)")
        wave_label.set_visible(True)

        # Draw a targeted bounding box around the mosquito signature on the rightmost slot of the spectrogram
        # Box dimensions target the wingbeat band: Y-axis spans 150 Hz to 1100 Hz
        box_y = 0
        box_h = 4000

        rect = patches.Rectangle(
            (SPEC_HISTORY - 1, box_y),
            1,
            box_h,
            linewidth=1.5,
            edgecolor=color,
            facecolor="none",
        )
        text = ax_spec.text(
            SPEC_HISTORY - 0.9,
            box_y + 50,
            class_name.split("_")[1],
            color=color,
            fontsize=8,
            fontweight="bold",
        )

        ax_spec.add_patch(rect)
        active_spec_boxes.append((rect, text))
    else:
        fig.suptitle(
            "Environment: Monitoring Room Noise (No Mosquito)",
            color="#555555",
            fontsize=16,
            fontweight="normal",
        )
        line.set_color("#555555")
        wave_box.set_edgecolor("none")
        wave_box.set_facecolor("none")
        wave_label.set_visible(False)

    # 4. Unpack and Plot Audio Waveform
    audio_bytes = packet[9:]
    new_samples = decode_24bit_pcm(audio_bytes).astype(np.float32)
    audio_buffer.extend(new_samples)
    y_data = np.array(audio_buffer)[-CHUNK_SIZE:]
    line.set_ydata(y_data)

    # 5. Compute Spectrogram Column
    normalized_y = y_data / MAX_INT_VAL
    windowed_data = normalized_y * np.hanning(len(normalized_y))
    fft_complex = np.fft.rfft(windowed_data)
    fft_mag = np.abs(fft_complex) / (CHUNK_SIZE / 2)
    fft_db = 20 * np.log10(fft_mag + 1e-9)

    spec_matrix[:-1] = spec_matrix[1:]
    spec_matrix[-1] = fft_db
    im.set_array(spec_matrix.T)

    return line, im


# Note: blit must be False so historical box updates render properly over the scrolling background image
ani = animation.FuncAnimation(
    fig, update, interval=30, blit=False, cache_frame_data=False
)

plt.tight_layout()
plt.show()

ser.close()
