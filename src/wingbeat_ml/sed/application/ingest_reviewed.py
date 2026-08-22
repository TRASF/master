"""Convert human-edited RIFF cues into a versioned verified SED dataset."""

from __future__ import annotations

import hashlib
import struct
import wave
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def _chunks(data: bytes, start: int, end: int):
    position = start
    while position + 8 <= end:
        chunk_id = data[position : position + 4]
        size = struct.unpack_from("<I", data, position + 4)[0]
        payload_start = position + 8
        payload_end = payload_start + size
        if payload_end > end:
            break
        yield chunk_id, data[payload_start:payload_end]
        position = payload_end + (size & 1)


def read_riff_regions(path: str | Path) -> list[dict]:
    """Read linked cue/ltxt/labl regions from one RIFF/WAVE file."""
    path = Path(path)
    data = path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"Not a RIFF/WAVE file: {path}")

    cues: dict[int, int] = {}
    lengths: dict[int, int] = {}
    labels: dict[int, str] = {}
    for chunk_id, payload in _chunks(data, 12, len(data)):
        if chunk_id == b"cue " and len(payload) >= 4:
            count = struct.unpack_from("<I", payload)[0]
            for index in range(count):
                offset = 4 + index * 24
                if offset + 24 <= len(payload):
                    cue_id, _, _, _, _, sample = struct.unpack_from("<II4sIII", payload, offset)
                    cues[cue_id] = sample
        elif chunk_id == b"LIST" and payload[:4] == b"adtl":
            for sub_id, sub_payload in _chunks(payload, 4, len(payload)):
                if sub_id == b"ltxt" and len(sub_payload) >= 8:
                    cue_id, sample_length = struct.unpack_from("<II", sub_payload)
                    lengths[cue_id] = sample_length
                elif sub_id == b"labl" and len(sub_payload) >= 4:
                    cue_id = struct.unpack_from("<I", sub_payload)[0]
                    labels[cue_id] = sub_payload[4:].rstrip(b"\0").decode("utf-8", errors="replace")

    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
    return [
        {
            "cue_id": cue_id,
            "start_s": sample / sample_rate,
            "end_s": (sample + lengths.get(cue_id, 1)) / sample_rate,
            "label": labels.get(cue_id, "mosquito"),
        }
        for cue_id, sample in sorted(cues.items(), key=lambda item: item[1])
    ]


def _iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union > 0 else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_for_review(review_wav: Path, reviewed_root: Path, candidates: pd.DataFrame) -> Path:
    relative = review_wav.relative_to(reviewed_root)
    stem = review_wav.name.removesuffix(".mosquito.wav")
    matches = candidates[candidates["file_path"].map(lambda value: Path(value).stem == stem)]
    if relative.parent != Path("."):
        suffix = relative.parent.parts
        narrowed = matches[
            matches["file_path"].map(
                lambda value: Path(value).parent.parts[-len(suffix) :] == suffix
            )
        ]
        if not narrowed.empty:
            matches = narrowed
    paths = sorted(set(matches["file_path"].astype(str)))
    if len(paths) != 1:
        raise ValueError(f"Expected one source match for {relative}, found {len(paths)}")
    return Path(paths[0])


def ingest_reviewed_wavs(
    reviewed_dir: str | Path,
    candidates_csv: str | Path,
    output_dir: str | Path,
    exhaustive: bool = False,
    match_iou: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Regenerate verified manifests from reviewed WAVs and immutable candidates."""
    reviewed_root = Path(reviewed_dir)
    candidate_path = Path(candidates_csv)
    if not reviewed_root.is_dir():
        raise FileNotFoundError(f"Reviewed WAV directory not found: {reviewed_root}")
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Candidate table not found: {candidate_path}")

    candidates = pd.read_csv(candidate_path)
    required = {"file_path", "start_s", "end_s"}
    if missing := required - set(candidates.columns):
        raise ValueError(f"Candidate table missing columns: {sorted(missing)}")

    recording_rows = []
    event_rows = []
    negative_rows = []
    provenance_rows = []

    review_files = sorted(reviewed_root.rglob("*.mosquito.wav"))
    for review_wav in review_files:
        source = _source_for_review(review_wav, reviewed_root, candidates)
        source_candidates = candidates[candidates["file_path"].astype(str) == str(source)].reset_index(drop=True)
        regions = read_riff_regions(review_wav)
        recording_id = hashlib.sha256(str(source).encode()).hexdigest()[:16]
        recording_rows.append({
            "recording_id": recording_id,
            "audio_path": str(source),
            "sha256": _sha256(source),
            "split_group": str(source.parent),
            "review_wav_path": str(review_wav),
            "review_scope": "exhaustive" if exhaustive else "candidate_only",
            "annotation_version": Path(output_dir).name,
        })

        scores = np.zeros((len(source_candidates), len(regions)), dtype=np.float32)
        for candidate_index, candidate in source_candidates.iterrows():
            for region_index, region in enumerate(regions):
                scores[candidate_index, region_index] = _iou(
                    (float(candidate.start_s), float(candidate.end_s)),
                    (region["start_s"], region["end_s"]),
                )
        matched_candidates: set[int] = set()
        matched_regions: set[int] = set()
        if scores.size:
            candidate_indices, region_indices = linear_sum_assignment(-scores)
            for candidate_index, region_index in zip(candidate_indices, region_indices):
                if scores[candidate_index, region_index] < match_iou:
                    continue
                matched_candidates.add(int(candidate_index))
                matched_regions.add(int(region_index))
                candidate = source_candidates.iloc[candidate_index]
                region = regions[region_index]
                corrected = (
                    abs(float(candidate.start_s) - region["start_s"]) > 0.04
                    or abs(float(candidate.end_s) - region["end_s"]) > 0.04
                )
                decision = "BOUNDARY_CORRECTED" if corrected else "ACCEPTED"
                event_rows.append({
                    "recording_id": recording_id,
                    "start_s": region["start_s"],
                    "end_s": region["end_s"],
                    "label": "mosquito",
                    "verification": "BOUNDARY_CORRECTED" if corrected else "VERIFIED_POSITIVE",
                    "source": "corrected_candidate" if corrected else "retained_candidate",
                })
                provenance_rows.append({
                    "recording_id": recording_id,
                    "original_start_s": candidate.start_s,
                    "original_end_s": candidate.end_s,
                    "reviewed_start_s": region["start_s"],
                    "reviewed_end_s": region["end_s"],
                    "decision": decision,
                    "model_hash": candidate.get("model_hash", ""),
                    "decoder_policy_hash": candidate.get("decoder_policy_hash", ""),
                })

        for candidate_index, candidate in source_candidates.iterrows():
            if candidate_index in matched_candidates:
                continue
            negative_rows.append({
                "recording_id": recording_id,
                "start_s": candidate.start_s,
                "end_s": candidate.end_s,
                "verification": "VERIFIED_FALSE_POSITIVE",
                "original_confidence": candidate.get("confidence", ""),
                "model_hash": candidate.get("model_hash", ""),
            })
            provenance_rows.append({
                "recording_id": recording_id,
                "original_start_s": candidate.start_s,
                "original_end_s": candidate.end_s,
                "reviewed_start_s": "",
                "reviewed_end_s": "",
                "decision": "REJECTED",
                "model_hash": candidate.get("model_hash", ""),
                "decoder_policy_hash": candidate.get("decoder_policy_hash", ""),
            })

        for region_index, region in enumerate(regions):
            if region_index in matched_regions:
                continue
            event_rows.append({
                "recording_id": recording_id,
                "start_s": region["start_s"],
                "end_s": region["end_s"],
                "label": "mosquito",
                "verification": "HUMAN_ADDED_POSITIVE",
                "source": "manual",
            })
            provenance_rows.append({
                "recording_id": recording_id,
                "original_start_s": "",
                "original_end_s": "",
                "reviewed_start_s": region["start_s"],
                "reviewed_end_s": region["end_s"],
                "decision": "HUMAN_ADDED",
                "model_hash": "",
                "decoder_policy_hash": "",
            })

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = (
        pd.DataFrame(recording_rows, columns=["recording_id", "audio_path", "sha256", "split_group", "review_wav_path", "review_scope", "annotation_version"]),
        pd.DataFrame(event_rows, columns=["recording_id", "start_s", "end_s", "label", "verification", "source"]),
        pd.DataFrame(negative_rows, columns=["recording_id", "start_s", "end_s", "verification", "original_confidence", "model_hash"]),
        pd.DataFrame(provenance_rows, columns=["recording_id", "original_start_s", "original_end_s", "reviewed_start_s", "reviewed_end_s", "decision", "model_hash", "decoder_policy_hash"]),
    )
    for table, name in zip(tables, ("recordings.csv", "events.csv", "hard_negatives.csv", "provenance.csv")):
        temporary = output / f"{name}.tmp"
        table.to_csv(temporary, index=False)
        temporary.replace(output / name)

    print(f"Imported {len(review_files)} reviewed WAVs")
    print(f"Verified positives: {len(tables[1])}")
    print(f"Verified false positives: {len(tables[2])}")
    print(f"Dataset: {output}")
    return tables
