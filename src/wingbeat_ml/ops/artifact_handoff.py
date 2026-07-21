"""Finalize, verify, and centralize completed training-run directories."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time


MANIFEST_NAME = ".artifact-manifest.json"
READY_NAME = ".artifact-ready.json"
SYNCED_NAME = ".artifact-synced.json"
VERIFIED_NAME = ".artifact-verified.json"
_CONTROL_NAMES = {
    MANIFEST_NAME,
    READY_NAME,
    SYNCED_NAME,
    VERIFIED_NAME,
}


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path, payload):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def sanitize_run_id(value):
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip(".-")
    if not run_id:
        raise ValueError("run id is empty after sanitization")
    return run_id[:160]


def finalize_run(run_dir, *, exit_code):
    """Write status and checksums, publishing the ready marker last."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = sanitize_run_id(run_dir.name)
    exit_code = int(exit_code)

    _write_json_atomic(
        run_dir / "run-status.json",
        {
            "completed_at": _utc_now(),
            "exit_code": exit_code,
            "run_id": run_id,
            "status": "succeeded" if exit_code == 0 else "failed",
        },
    )

    files = []
    for path in sorted(candidate for candidate in run_dir.rglob("*") if candidate.is_file()):
        relative = path.relative_to(run_dir).as_posix()
        if path.name in _CONTROL_NAMES or path.name.endswith(".tmp"):
            continue
        files.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
        )

    manifest_path = _write_json_atomic(
        run_dir / MANIFEST_NAME,
        {
            "exit_code": exit_code,
            "files": files,
            "generated_at": _utc_now(),
            "run_id": run_id,
            "version": 1,
        },
    )
    return _write_json_atomic(
        run_dir / READY_NAME,
        {
            "exit_code": exit_code,
            "file_count": len(files),
            "manifest_sha256": _sha256(manifest_path),
            "ready_at": _utc_now(),
            "run_id": run_id,
        },
    )


def verify_run(run_dir):
    """Verify a finalized run and return its ready-marker metadata."""
    run_dir = Path(run_dir).resolve()
    ready_path = run_dir / READY_NAME
    manifest_path = run_dir / MANIFEST_NAME
    if not ready_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"artifact handoff is not ready: {run_dir}")

    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if _sha256(manifest_path) != ready["manifest_sha256"]:
        raise RuntimeError("artifact manifest checksum mismatch")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = run_dir / item["path"]
        if not path.is_file():
            raise RuntimeError(f"artifact file is missing: {item['path']}")
        if path.stat().st_size != item["size"] or _sha256(path) != item["sha256"]:
            raise RuntimeError(f"artifact file checksum mismatch: {item['path']}")

    if len(manifest["files"]) != ready["file_count"]:
        raise RuntimeError("artifact manifest file count mismatch")
    return ready


def _remote_verification_script(destination):
    destination_literal = json.dumps(str(destination))
    return f'''from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

root = Path({destination_literal})

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

ready_path = root / {READY_NAME!r}
manifest_path = root / {MANIFEST_NAME!r}
ready = json.loads(ready_path.read_text(encoding="utf-8"))
if sha256(manifest_path) != ready["manifest_sha256"]:
    raise SystemExit("artifact manifest checksum mismatch")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for item in manifest["files"]:
    path = root / item["path"]
    if not path.is_file():
        raise SystemExit(f"artifact file is missing: {{item['path']}}")
    if path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
        raise SystemExit(f"artifact file checksum mismatch: {{item['path']}}")
payload = {{
    "run_id": ready["run_id"],
    "verified_at": datetime.now(timezone.utc).isoformat(),
}}
target = root / {VERIFIED_NAME!r}
temporary = target.with_name(target.name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
temporary.replace(target)
'''


def _remote_prepare_script(destination):
    destination_literal = json.dumps(str(destination))
    return f'''from pathlib import Path
Path({destination_literal}).mkdir(parents=True, exist_ok=True)
'''


def _sync_remote(source, destination, owner, run_command):
    run_command(
        ["ssh", owner, "python3", "-"],
        check=True,
        input=_remote_prepare_script(destination),
        text=True,
    )
    run_command(
        [
            "rsync",
            "--archive",
            "--partial",
            "--delay-updates",
            "--protect-args",
            f"{source}/",
            f"{owner}:{destination}/",
        ],
        check=True,
    )
    run_command(
        ["ssh", owner, "python3", "-"],
        check=True,
        input=_remote_verification_script(destination),
        text=True,
    )


def sync_ready_runs(
    staging_root,
    destination_root,
    *,
    owner=None,
    run_command=subprocess.run,
):
    """Copy ready runs, verify the owner copy, and retain local staging."""
    staging_root = Path(staging_root).expanduser().resolve()
    if owner:
        destination_root = Path(destination_root)
        if not destination_root.is_absolute():
            raise ValueError("remote artifact destination must be absolute")
    else:
        destination_root = Path(destination_root).expanduser().resolve()
    runs_root = staging_root / "runs"
    if not runs_root.is_dir():
        return []

    synced = []
    for source in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        if not (source / READY_NAME).is_file() or (source / SYNCED_NAME).exists():
            continue
        ready = verify_run(source)
        destination = destination_root / "runs" / sanitize_run_id(source.name)
        if owner:
            _sync_remote(source, destination, owner, run_command)
        elif source != destination:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination, dirs_exist_ok=True)
        if not owner:
            verify_run(destination)
            _write_json_atomic(
                destination / VERIFIED_NAME,
                {
                    "run_id": ready["run_id"],
                    "verified_at": _utc_now(),
                },
            )
        _write_json_atomic(
            source / SYNCED_NAME,
            {
                "destination": (
                    f"{owner}:{destination}" if owner else str(destination)
                ),
                "run_id": ready["run_id"],
                "synced_at": _utc_now(),
            },
        )
        synced.append(ready["run_id"])
    return synced


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    sanitize_parser = commands.add_parser("sanitize")
    sanitize_parser.add_argument("run_id")

    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--run-dir", required=True)
    finalize_parser.add_argument("--exit-code", required=True, type=int)

    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--run-dir", required=True)

    sync_parser = commands.add_parser("sync")
    sync_parser.add_argument("--staging-root", required=True)
    sync_parser.add_argument("--destination-root", required=True)
    sync_parser.add_argument("--owner")
    sync_parser.add_argument("--watch", action="store_true")
    sync_parser.add_argument("--poll-seconds", type=float, default=30.0)

    args = parser.parse_args(argv)
    if args.command == "sanitize":
        print(sanitize_run_id(args.run_id))
        return 0
    if args.command == "finalize":
        print(finalize_run(args.run_dir, exit_code=args.exit_code))
        return 0
    if args.command == "verify":
        print(json.dumps(verify_run(args.run_dir), sort_keys=True))
        return 0

    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be greater than zero")
    while True:
        synced = sync_ready_runs(
            args.staging_root,
            args.destination_root,
            owner=args.owner or None,
        )
        for run_id in synced:
            print(f"Artifact verified on owner: {run_id}", flush=True)
        if not args.watch:
            return 0
        time.sleep(args.poll_seconds)


__all__ = [
    "finalize_run",
    "sanitize_run_id",
    "sync_ready_runs",
    "verify_run",
]


if __name__ == "__main__":
    raise SystemExit(main())
