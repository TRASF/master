"""Lossless WAV review-copy exporter with Ocenaudio RIFF regions."""

from __future__ import annotations

import os
import re
import shutil
import struct
import wave
from pathlib import Path


def _chunk(chunk_id: bytes, payload: bytes) -> bytes:
    return chunk_id + struct.pack("<I", len(payload)) + payload + (b"\0" if len(payload) % 2 else b"")


def _region_chunks(events: list[dict], sample_rate: int) -> bytes:
    cue_points = []
    adtl = []
    for cue_id, event in enumerate(events, start=1):
        start_sample = max(0, round(float(event["start_s"]) * sample_rate))
        length_samples = max(1, round((float(event["end_s"]) - float(event["start_s"])) * sample_rate))
        confidence = float(event.get("confidence", 1.0))
        label = str(event.get("label", "mosquito"))
        text = f"DET-{cue_id:06d} | conf={confidence:.4f} | {label}".encode("utf-8") + b"\0"

        cue_points.append(struct.pack("<II4sIII", cue_id, start_sample, b"data", 0, 0, start_sample))
        adtl.append(_chunk(b"ltxt", struct.pack("<II4sHHHH", cue_id, length_samples, b"rgn ", 0, 0, 0, 0)))
        adtl.append(_chunk(b"labl", struct.pack("<I", cue_id) + text))

    cue = _chunk(b"cue ", struct.pack("<I", len(cue_points)) + b"".join(cue_points))
    return cue + _chunk(b"LIST", b"adtl" + b"".join(adtl))


def _raw_pcm_format(path: Path) -> tuple[int, int, int]:
    sample_rate, bits, channels = 44_100, 24, 1
    if match := re.search(r"(?:^|_)sr(\d+)(?=_|\.|$)", path.name):
        sample_rate = int(match.group(1))
    if match := re.search(r"(?:^|_)b(\d+)(?=_|\.|$)", path.name):
        bits = int(match.group(1))
    if match := re.search(r"(?:^|_)c(\d+)(?=_|\.|$)", path.name):
        channels = int(match.group(1))
    return sample_rate, bits, channels


def export_review_wav(
    source_audio: str | Path,
    output_wav: str | Path,
    events: list[dict],
) -> Path:
    """Copy PCM audio unchanged and embed linked cue/ltxt/labl regions."""
    source = Path(source_audio)
    output = Path(output_wav)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")

    is_riff_wave = False
    if source.suffix.lower() == ".wav":
        with source.open("rb") as source_file:
            is_riff_wave = source_file.read(12)[:4] == b"RIFF"

    if is_riff_wave:
        with wave.open(str(source), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
        with source.open("rb") as source_file, temporary.open("wb") as output_file:
            shutil.copyfileobj(source_file, output_file, 1024 * 1024)
            if output_file.tell() % 2:
                output_file.write(b"\0")
            output_file.write(_region_chunks(events, sample_rate))
            file_size = output_file.tell()
            if file_size - 8 > 0xFFFFFFFF:
                raise ValueError(f"RIFF output exceeds 4 GiB: {output}")
            output_file.seek(4)
            output_file.write(struct.pack("<I", file_size - 8))
    else:
        sample_rate, bits, channels = _raw_pcm_format(source)
        block_align = channels * (bits // 8)
        byte_rate = sample_rate * block_align
        data_size = source.stat().st_size
        metadata = _region_chunks(events, sample_rate)
        riff_size = 4 + 24 + 8 + data_size + (data_size % 2) + len(metadata)
        if riff_size > 0xFFFFFFFF:
            raise ValueError(f"RIFF output exceeds 4 GiB: {output}")
        with source.open("rb") as source_file, temporary.open("wb") as output_file:
            output_file.write(b"RIFF" + struct.pack("<I", riff_size) + b"WAVE")
            output_file.write(_chunk(b"fmt ", struct.pack("<HHIIHH", 1, channels, sample_rate, byte_rate, block_align, bits)))
            output_file.write(b"data" + struct.pack("<I", data_size))
            shutil.copyfileobj(source_file, output_file, 1024 * 1024)
            if data_size % 2:
                output_file.write(b"\0")
            output_file.write(metadata)

    os.replace(temporary, output)
    return output
