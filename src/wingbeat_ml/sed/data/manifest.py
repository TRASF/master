"""Sound Event Detection (SED) Manifest Generator.

Generates recordings.csv, events.csv, and clips.csv manifests with complete
provenance tracking, SHA-256 audio hashing, metadata, and split_group tags.
"""

from __future__ import annotations

import hashlib
import os
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

from wingbeat_ml.data.synthetic import decode_pcm_bytes


@dataclass
class RecordingRecord:
    file_id: str
    path: str
    sha256: str
    size_bytes: int
    source_id: str
    parent_recording_id: str
    dataset: str
    supervision_type: str  # strong, positive_clip, weak_positive, negative, unlabeled
    sample_rate_hz: int
    channels: int
    duration_s: float
    device: str
    session: str
    environment: str
    location: str
    species: str
    sex: str
    individual_id: str
    annotation_source: str
    split_group: str
    channel_index: int = 0


@dataclass
class EventRecord:
    event_id: str
    recording_file_id: str
    path: str
    start_s: float
    end_s: float
    duration_s: float
    label: str
    confidence: float
    provenance: str
    split_group: str


@dataclass
class ClipRecord:
    clip_id: str
    recording_file_id: str
    path: str
    sha256: str
    duration_s: float
    label: str
    supervision_type: str
    species: str
    sex: str
    split_group: str


def compute_sha256(path: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_audio_info(path: str | Path) -> tuple[int, int, float, int]:
    """Get sample rate, channels, duration, and byte size for audio file."""
    p = Path(path)
    size_bytes = p.stat().st_size
    try:
        with wave.open(str(p), "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            frames = wf.getnframes()
            dur = frames / float(sr) if sr > 0 else 0.0
            return sr, ch, dur, size_bytes
    except Exception:
        # Fallback estimation for raw / unparsed audio if needed
        return 44100, 1, 0.0, size_bytes


def get_active_channel_indices(path: str | Path, threshold: float = 1e-4) -> list[int]:
    """Return 1-based physical microphone channels, excluding dead channels."""
    with wave.open(str(path), "rb") as wf:
        n_ch = wf.getnchannels()
        signal = decode_pcm_bytes(wf.readframes(wf.getnframes()), wf.getsampwidth())
    if n_ch == 1:
        return [1]
    signal = signal[: len(signal) - (len(signal) % n_ch)].reshape(-1, n_ch)
    rms = np.sqrt(np.mean(signal**2, axis=0))
    active = np.flatnonzero(rms >= threshold)
    if not len(active):
        active = np.array([int(np.argmax(rms))])
    return (active + 1).tolist()


def generate_sed_manifests(
    msb_dir: str | Path,
    philip_dir: str | Path,
    recordings_dir: str | Path,
    output_dir: str | Path,
    trang_dir: str | Path | None = "/home/miru4090s/clones/Trang/datasets/Trang",
    active_channel_rms_threshold: float = 1e-4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scan dataset directories and export recordings.csv, events.csv, clips.csv."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    recordings: list[RecordingRecord] = []
    events: list[EventRecord] = []
    clips: list[ClipRecord] = []

    # 1. Process MSB dataset
    msb_p = Path(msb_dir)
    if msb_p.exists():
        for file_p in msb_p.rglob("*.wav"):
            rel_path = str(file_p)
            sr, ch, dur, size = get_audio_info(file_p)
            sha = compute_sha256(file_p)

            # Metadata parsing
            parts = file_p.parts
            env = "indoor" if "Indoor" in parts else ("outdoor" if "Outdoor" in parts else "environmental")
            label = "mosquito"
            supervision = "positive_clip"
            species = "unknown"
            sex = "unknown"

            for p in parts:
                if "Ae_aegypti" in p or "Ae. aegypti" in p:
                    species = "Ae_aegypti"
                elif "Ae_albopictus" in p:
                    species = "Ae_albopictus"
                elif "An_minimus" in p or "An_Minimus" in p:
                    species = "An_minimus"
                elif "Cx_quin" in p:
                    species = "Cx_quin"

                if "Female" in p or "_F" in p:
                    sex = "female"
                elif "Male" in p or "_M" in p:
                    sex = "male"

                if "No.Mos" in p or "Nomos" in p or "noises" in p:
                    label = "background"
                    supervision = "negative"

            parent_id = f"msb_{file_p.parent.name}"
            file_id = f"msb_{sha[:12]}"

            rec = RecordingRecord(
                file_id=file_id,
                path=rel_path,
                sha256=sha,
                size_bytes=size,
                source_id=f"MSB/{file_p.name}",
                parent_recording_id=parent_id,
                dataset="MSB",
                supervision_type=supervision,
                sample_rate_hz=sr,
                channels=ch,
                duration_s=dur,
                device="INMP441",
                session="MSB_session",
                environment=env,
                location="lab",
                species=species,
                sex=sex,
                individual_id="unknown",
                annotation_source="dataset_folder",
                split_group=parent_id,
            )
            recordings.append(rec)

            clip = ClipRecord(
                clip_id=f"clip_{file_id}",
                recording_file_id=file_id,
                path=rel_path,
                sha256=sha,
                duration_s=dur,
                label=label,
                supervision_type=supervision,
                species=species,
                sex=sex,
                split_group=parent_id,
            )
            clips.append(clip)

    # 2. Process Philip dataset
    philip_p = Path(philip_dir)
    if philip_p.exists():
        for file_p in philip_p.rglob("*.wav"):
            rel_path = str(file_p)
            sr, ch, dur, size = get_audio_info(file_p)
            sha = compute_sha256(file_p)
            parts = file_p.parts

            label = "mosquito" if "Lab" in parts else "background"
            supervision = "positive_clip" if label == "mosquito" else "negative"

            parent_id = f"philip_{file_p.parent.name}"
            file_id = f"philip_{sha[:12]}"

            rec = RecordingRecord(
                file_id=file_id,
                path=rel_path,
                sha256=sha,
                size_bytes=size,
                source_id=f"Philip/{file_p.name}",
                parent_recording_id=parent_id,
                dataset="Philip",
                supervision_type="unlabeled" if label != "mosquito" else "positive_clip",
                sample_rate_hz=sr,
                channels=ch,
                duration_s=dur,
                device="recorder_philip",
                session="Philip_session",
                environment="lab" if "Lab" in parts else "outdoor",
                location="lab",
                species="unknown",
                sex="unknown",
                individual_id="unknown",
                annotation_source="cage_unlabeled",
                split_group=parent_id,
            )
            recordings.append(rec)

            clip = ClipRecord(
                clip_id=f"clip_{file_id}",
                recording_file_id=file_id,
                path=rel_path,
                sha256=sha,
                duration_s=dur,
                label=label,
                supervision_type=supervision,
                species="unknown",
                sex="unknown",
                split_group=parent_id,
            )
            clips.append(clip)

    # 3. Process Long Recordings directory & existing manifests
    recs_p = Path(recordings_dir)
    if recs_p.exists():
        labeled_manifest = recs_p / "by_species_mos_labeled" / "manifest.csv"
        if labeled_manifest.exists():
            df_lbl = pd.read_csv(labeled_manifest)
            for idx, row in df_lbl.iterrows():
                out_wav = str(row["output_wav"])
                wav_path = Path(out_wav)
                if not wav_path.exists():
                    continue

                sr, ch, recording_duration, size = get_audio_info(wav_path)
                event_duration = float(row.get("duration_sec", 0.0))
                start_s = float(row.get("start_sec", 0.0))
                end_s = float(row.get("end_sec", start_s + event_duration))

                sha = compute_sha256(wav_path)
                parent_id = f"rec_{wav_path.stem.split('__')[0]}"
                file_id = f"long_{sha[:12]}"

                rec = RecordingRecord(
                    file_id=file_id,
                    path=out_wav,
                    sha256=sha,
                    size_bytes=size,
                    source_id=str(row.get("source_path", out_wav)),
                    parent_recording_id=parent_id,
                    dataset="LongRecordings",
                    supervision_type="strong",
                    sample_rate_hz=sr,
                    channels=ch,
                    duration_s=recording_duration,
                    device="ESP32_recorder",
                    session=parent_id,
                    environment="field",
                    location="field",
                    species=str(row.get("species", "unknown")),
                    sex="unknown",
                    individual_id="unknown",
                    annotation_source="detector_v0",
                    split_group=parent_id,
                )
                recordings.append(rec)

                evt = EventRecord(
                    event_id=f"evt_{file_id}_{idx}",
                    recording_file_id=file_id,
                    path=out_wav,
                    start_s=start_s,
                    end_s=end_s,
                    duration_s=end_s - start_s,
                    label="mosquito",
                    confidence=float(row.get("confidence", 1.0)),
                    provenance="detector_prelabel",
                    split_group=parent_id,
                )
                events.append(evt)

    # 4. Process Trang dataset
    if trang_dir is not None:
        trang_p = Path(trang_dir)
        meta_csv = trang_p / "recording_metadata.csv"
        if meta_csv.exists():
            df_trang = pd.read_csv(meta_csv)
            for idx, row in df_trang.iterrows():
                rel_folder = str(row["relative_folder"])
                wav_name = str(row["wav_file"])
                lbl_name = str(row["label_file"])

                folder_p = trang_p / rel_folder
                wav_p = folder_p / wav_name
                lbl_p = folder_p / lbl_name

                if not wav_p.exists():
                    continue

                sr, ch, dur, size = get_audio_info(wav_p)
                sha = compute_sha256(wav_p)
                parent_id = f"trang_{rel_folder.replace('/', '_')}"

                species = str(row.get("species_corrected", row.get("species", "unknown")))
                sex = str(row.get("sex", "unknown"))
                labelled_intervals = []
                if lbl_p.exists():
                    with open(lbl_p, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                try:
                                    labelled_intervals.append((float(parts[0]), float(parts[1])))
                                except ValueError:
                                    continue

                for channel_index in get_active_channel_indices(wav_p, active_channel_rms_threshold):
                    file_id = f"trang_{sha[:12]}_ch{channel_index}"
                    rec = RecordingRecord(
                        file_id=file_id,
                        path=str(wav_p),
                        sha256=sha,
                        size_bytes=size,
                        source_id=f"Trang/{wav_name}#ch{channel_index}",
                        parent_recording_id=parent_id,
                        dataset="Trang",
                        supervision_type="strong",
                        sample_rate_hz=sr,
                        channels=1,
                        duration_s=dur,
                        device=f"{row.get('microphones', 'Clippy')}:ch{channel_index}",
                        session=parent_id,
                        environment="lab",
                        location="lab",
                        species=species,
                        sex=sex,
                        individual_id=str(row.get("record_id", "unknown")),
                        annotation_source="trang_labels_txt",
                        split_group=parent_id,
                        channel_index=channel_index,
                    )
                    recordings.append(rec)

                    for e_idx, (st_s, en_s) in enumerate(labelled_intervals):
                        events.append(EventRecord(
                            event_id=f"evt_trang_{file_id}_{e_idx}",
                            recording_file_id=file_id,
                            path=str(wav_p),
                            start_s=st_s,
                            end_s=en_s,
                            duration_s=en_s - st_s,
                            label="mosquito",
                            confidence=1.0,
                            provenance="trang_ground_truth",
                            split_group=parent_id,
                        ))

    rec_df = pd.DataFrame([asdict(r) for r in recordings]).drop_duplicates("file_id")
    evt_df = pd.DataFrame([asdict(e) for e in events])
    clip_df = pd.DataFrame([asdict(c) for c in clips])

    rec_df.to_csv(output_path / "recordings.csv", index=False)
    evt_df.to_csv(output_path / "events.csv", index=False)
    clip_df.to_csv(output_path / "clips.csv", index=False)

    return rec_df, evt_df, clip_df


if __name__ == "__main__":
    generate_sed_manifests(
        msb_dir="/home/miru4090s/clones/Master Thesises/MosSongPlus/dataset/MSB",
        philip_dir="/home/miru4090s/clones/Master Thesises/MosSongPlus/dataset/Philip",
        recordings_dir="/media/miru4090s/New Volume2/recordings",
        output_dir="/home/miru4090s/clones/Master Thesises/MosSongPlus/metadata",
    )
