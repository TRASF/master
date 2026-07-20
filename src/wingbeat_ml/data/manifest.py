"""Dataset manifest: identity, file hashes, and metadata.

The manifest provides a deterministic, portable record of a dataset version.
It does not store machine-specific absolute paths or volatile timestamps.

Schema version: 1

Manifest header fields:
  schema_version    int     Always 1 for this format.
  dataset_name      str     Human-readable dataset identifier.
  dataset_version   str     Semantic version string (e.g. "1.0.0").
  label_map_version str     Version of the label mapping.
  sample_rate       int     Target audio sample rate (e.g. 8000).
  segment_length    int     Target segment length in samples (e.g. 2400).
  split_seed        int     Seed used for reproducible splitting.
  files             list    Per-file records (see FileRecord below).

Per-file record fields:
  path              str     Relative path from dataset_root (no absolute paths).
  sha256            str     Hex SHA-256 of the raw file bytes.
  size_bytes        int     File size in bytes.
  label             str     Human-readable class name.
  label_index       int     Integer class index.
  source_recording  str     Stable source-recording identifier.
  split             str     "train" | "validation" | "test".
  sample_rate       int     Recorded sample rate (from file metadata).

Identity hash:
  manifest_sha256() hashes the canonical JSON of header + sorted file records,
  excluding: created_at, any filesystem-modification times, and the manifest
  identity hash itself.

Sorting key for determinism: (split, label, path)
"""

from __future__ import annotations

import hashlib
import json
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FileRecord:
    path: str
    sha256: str
    size_bytes: int
    label: str
    label_index: int
    source_recording: str
    split: str
    sample_rate: int


@dataclass
class DatasetManifest:
    schema_version: int = 1
    dataset_name: str = ""
    dataset_version: str = "0.0.0"
    label_map_version: str = "1.0"
    sample_rate: int = 8000
    segment_length: int = 2400
    split_seed: int = 42
    files: list[FileRecord] = field(default_factory=list)
    # created_at is excluded from the identity hash
    created_at: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: str | Path) -> str:
    """Stream-hash a file with SHA-256."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_sample_rate(path: str | Path) -> int:
    """Return the sample rate stored in a WAV header; 0 for unknown."""
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getframerate()
    except Exception:
        return 0


def _source_recording_id(path: str | Path, dataset_root: str | Path) -> str:
    """Derive a stable source-recording ID from the relative path parent.

    Uses the parent directory name relative to the class directory as the
    source identifier.  Falls back to the file stem when the structure is flat.
    """
    rel = Path(path).relative_to(Path(dataset_root))
    parts = rel.parts

    # Nested layout: <class>/<recording>/<file>.
    if len(parts) >= 3:
        return "/".join(parts[:2])

    # Flat layout: <class>/<file>.
    return rel.stem


def _sort_key(rec: FileRecord) -> tuple[str, str, str]:
    return (rec.split, rec.label, rec.path)


def _manifest_identity_dict(manifest: DatasetManifest) -> dict:
    """Return the subset of manifest data included in the identity hash.

    Excludes: created_at, manifest identity hash itself.
    """
    files_dicts = []
    for r in sorted(manifest.files, key=_sort_key):
        files_dicts.append({
            "path": r.path,
            "sha256": r.sha256,
            "size_bytes": r.size_bytes,
            "label": r.label,
            "label_index": r.label_index,
            "source_recording": r.source_recording,
            "split": r.split,
            "sample_rate": r.sample_rate,
        })
    return {
        "schema_version": manifest.schema_version,
        "dataset_name": manifest.dataset_name,
        "dataset_version": manifest.dataset_version,
        "label_map_version": manifest.label_map_version,
        "sample_rate": manifest.sample_rate,
        "segment_length": manifest.segment_length,
        "split_seed": manifest.split_seed,
        "files": files_dicts,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_manifest(
    dataset_root: str | Path,
    file_paths: Sequence[str],
    labels: Sequence[str],
    label_indices: Sequence[int],
    splits: Sequence[str],
    *,
    dataset_name: str = "",
    dataset_version: str = "0.0.0",
    label_map_version: str = "1.0",
    sample_rate: int = 8000,
    segment_length: int = 2400,
    split_seed: int = 42,
    full_hash: bool = True,
) -> DatasetManifest:
    """Generate a DatasetManifest from file lists.

    Args:
        dataset_root: Root directory; used only to compute relative paths.
        file_paths: Absolute or relative-to-cwd file paths.
        labels: Class name string for each file.
        label_indices: Integer class index for each file.
        splits: Split assignment ("train"/"validation"/"test") per file.
        dataset_name: Human-readable name.
        dataset_version: Semantic version string.
        label_map_version: Version of the label mapping.
        sample_rate: Target sample rate.
        segment_length: Target segment length in samples.
        split_seed: Seed used for reproducible splitting.
        full_hash: If True, SHA-256 hash each file (may be slow).
                   If False, uses empty string for sha256.

    Returns:
        DatasetManifest with sorted file records.
    """
    column_lengths = {
        len(file_paths),
        len(labels),
        len(label_indices),
        len(splits),
    }
    if len(column_lengths) != 1:
        raise ValueError(
            "Manifest columns must contain the same number of values."
        )

    allowed_splits = {"train", "validation", "test"}
    invalid_splits = sorted(set(splits) - allowed_splits)
    if invalid_splits:
        raise ValueError(f"Invalid manifest splits: {invalid_splits}")

    root = Path(dataset_root).resolve()
    records: list[FileRecord] = []

    for fp, label, idx, split in zip(file_paths, labels, label_indices, splits):
        abs_path = Path(fp).resolve()

        try:
            rel_path = str(abs_path.relative_to(root))
        except ValueError as error:
            raise ValueError(
                f"Manifest file is outside dataset root: {abs_path}"
            ) from error

        if not abs_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {abs_path}")

        size_bytes = abs_path.stat().st_size
        sha256 = _sha256_file(abs_path) if full_hash else ""
        src_rec = _source_recording_id(abs_path, root)
        file_sr = _file_sample_rate(abs_path) if abs_path.suffix.casefold() == ".wav" else 0

        records.append(FileRecord(
            path=rel_path,
            sha256=sha256,
            size_bytes=size_bytes,
            label=label,
            label_index=int(idx),
            source_recording=src_rec,
            split=split,
            sample_rate=file_sr,
        ))

    records.sort(key=_sort_key)

    return DatasetManifest(
        schema_version=1,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        label_map_version=label_map_version,
        sample_rate=sample_rate,
        segment_length=segment_length,
        split_seed=split_seed,
        files=records,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def manifest_sha256(manifest: DatasetManifest) -> str:
    """Compute the identity SHA-256 of a manifest.

    Hashes canonical JSON of the identity-relevant fields only.
    Excluded: created_at, any filesystem timestamps, absolute paths.

    Returns:
        64-character lowercase hex string.
    """
    identity = _manifest_identity_dict(manifest)
    canonical = json.dumps(identity, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_manifest(manifest: DatasetManifest, path: str | Path) -> None:
    """Serialise a manifest to a JSON file at *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _manifest_identity_dict(manifest)
    data["created_at"] = manifest.created_at
    path.write_text(
        json.dumps(data, indent=2, sort_keys=False, ensure_ascii=True),
        encoding="utf-8",
    )


def load_manifest(path: str | Path) -> DatasetManifest:
    """Deserialise a manifest from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    files = [FileRecord(**r) for r in data.get("files", [])]
    return DatasetManifest(
        schema_version=int(data.get("schema_version", 1)),
        dataset_name=str(data.get("dataset_name", "")),
        dataset_version=str(data.get("dataset_version", "0.0.0")),
        label_map_version=str(data.get("label_map_version", "1.0")),
        sample_rate=int(data.get("sample_rate", 8000)),
        segment_length=int(data.get("segment_length", 2400)),
        split_seed=int(data.get("split_seed", 42)),
        files=files,
        created_at=str(data.get("created_at", "")),
    )


def verify_manifest(
    manifest: DatasetManifest,
    dataset_root: str | Path,
    *,
    full_hash: bool = False,
) -> dict[str, list[str]]:
    """Verify that files in the manifest match expected metadata.

    Args:
        manifest: The manifest to verify.
        dataset_root: Root path used to resolve relative file paths.
        full_hash: If True, also verify SHA-256 hashes (slow on large datasets).
                   If False, only verifies file existence and size.

    Returns:
        Dict with keys "missing", "size_mismatch", "hash_mismatch".
        Empty lists indicate no issues.
    """
    root = Path(dataset_root)
    issues: dict[str, list[str]] = {
        "missing": [],
        "size_mismatch": [],
        "hash_mismatch": [],
    }

    for rec in manifest.files:
        abs_path = root / rec.path
        if not abs_path.is_file():
            issues["missing"].append(rec.path)
            continue
        actual_size = abs_path.stat().st_size
        if actual_size != rec.size_bytes:
            issues["size_mismatch"].append(
                f"{rec.path} (expected {rec.size_bytes}, got {actual_size})"
            )
        if full_hash and rec.sha256:
            actual_hash = _sha256_file(abs_path)
            if actual_hash != rec.sha256:
                issues["hash_mismatch"].append(rec.path)

    return issues
