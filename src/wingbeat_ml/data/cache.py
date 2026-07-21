"""Stable, single-writer TensorFlow cache publication."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time


CACHE_SCHEMA_VERSION = 1
_CACHE_EVENTS = []


def stable_cache_key(
    file_paths,
    preprocessing,
    *,
    manifest_sha256="",
    schema_version=CACHE_SCHEMA_VERSION,
):
    """Hash deterministic cache inputs without filesystem timestamps."""
    identity = {
        "schema_version": int(schema_version),
        "manifest_sha256": str(manifest_sha256 or ""),
        "paths": sorted(str(path) for path in file_paths),
        "preprocessing": preprocessing,
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def exclusive_cache_lock(path):
    """Hold an inter-process lock for one cache prefix."""
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def materialize_tensorflow_cache(dataset, prefix):
    """Build a cache once and atomically publish its TensorFlow files."""
    prefix = Path(prefix)
    started = time.perf_counter()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    complete = Path(f"{prefix}.complete")
    if complete.is_file():
        _CACHE_EVENTS.append({
            "key": prefix.name,
            "reused": True,
            "duration_seconds": time.perf_counter() - started,
        })
        return dataset.cache(str(prefix))

    with exclusive_cache_lock(f"{prefix}.lock"):
        if complete.is_file():
            _CACHE_EVENTS.append({
                "key": prefix.name,
                "reused": True,
                "duration_seconds": time.perf_counter() - started,
            })
            return dataset.cache(str(prefix))

        for stale in prefix.parent.glob(f"{prefix.name}.tmp-*"):
            if stale.is_file():
                stale.unlink()
        temporary = Path(f"{prefix}.tmp-{os.getpid()}")
        cached = dataset.cache(str(temporary))
        for _ in cached:
            pass

        temporary_files = sorted(
            temporary.parent.glob(f"{temporary.name}.*")
        )
        if not temporary_files:
            raise RuntimeError(f"TensorFlow did not create cache files: {temporary}")
        for source in temporary_files:
            suffix = source.name[len(temporary.name):]
            os.replace(source, Path(f"{prefix}{suffix}"))

        complete_tmp = Path(f"{complete}.tmp-{os.getpid()}")
        complete_tmp.write_text("complete\n", encoding="utf-8")
        os.replace(complete_tmp, complete)

    _CACHE_EVENTS.append({
        "key": prefix.name,
        "reused": False,
        "duration_seconds": time.perf_counter() - started,
    })

    return dataset.cache(str(prefix))


def consume_cache_events():
    """Return cache timing events accumulated during dataset construction."""
    events = list(_CACHE_EVENTS)
    _CACHE_EVENTS.clear()
    return events


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "consume_cache_events",
    "exclusive_cache_lock",
    "materialize_tensorflow_cache",
    "stable_cache_key",
]
