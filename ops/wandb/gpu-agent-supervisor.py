#!/usr/bin/env python3
"""Start one W&B Launch agent per detected GPU."""

import argparse
import json
import os
import shutil
import subprocess
import sys

from wingbeat_ml.ops.gpu_agents import (
    build_agent_specs,
    discover_gpus,
    supervise_agents,
)
from wingbeat_ml.ops.preflight import run_host_preflight


def require_command(name):
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command is not installed: {name}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default=os.environ.get("WINGBEAT_LAUNCH_QUEUE", "wingbeat-training"))
    parser.add_argument("--dataset-dir", default=os.environ.get("WINGBEAT_DATASET_DIR"))
    parser.add_argument("--runtime-root", default=os.environ.get("WINGBEAT_RUNTIME_ROOT"))
    parser.add_argument("--cache-dir", default=os.environ.get("WINGBEAT_CACHE_DIR"))
    parser.add_argument("--manifest", default=os.environ.get("WINGBEAT_DATASET_MANIFEST"))
    parser.add_argument("--manifest-sha256", default=os.environ.get("WINGBEAT_DATASET_MANIFEST_SHA256"))
    parser.add_argument("--container-dataset-dir", default="/data")
    parser.add_argument("--container-runtime-root", default="/runtime")
    parser.add_argument("--container-cache-dir", default="/runtime/dataset/.tf_cache")
    parser.add_argument("--container-manifest", default="/data/dataset-manifest.json")
    parser.add_argument(
        "--gpu-check-image",
        default=os.environ.get(
            "WINGBEAT_GPU_CHECK_IMAGE",
            "nvidia/cuda:12.8.1-base-ubuntu24.04",
        ),
    )
    parser.add_argument("--skip-docker-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    for value, name in (
        (args.dataset_dir, "WINGBEAT_DATASET_DIR"),
        (args.runtime_root, "WINGBEAT_RUNTIME_ROOT"),
        (args.cache_dir, "WINGBEAT_CACHE_DIR"),
    ):
        if not value:
            parser.error(f"{name} is required")
    if not args.manifest or not args.manifest_sha256:
        parser.error(
            "WINGBEAT_DATASET_MANIFEST and "
            "WINGBEAT_DATASET_MANIFEST_SHA256 are required"
        )

    require_command("nvidia-smi")
    require_command("wandb")
    require_command("docker")
    subprocess.run(["wandb", "status"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        ["docker", "info"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    if not args.skip_docker_check:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                args.gpu_check_image,
                "nvidia-smi",
                "-L",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    preflight = run_host_preflight(
        dataset_dir=args.dataset_dir,
        runtime_root=args.runtime_root,
        cache_dir=args.cache_dir,
        manifest_path=args.manifest,
        expected_manifest_sha256=args.manifest_sha256,
    )
    specs = build_agent_specs(
        discover_gpus(),
        queue=args.queue,
        job_environment={
            "WINGBEAT_DATASET_DIR": args.container_dataset_dir,
            "WINGBEAT_RUNTIME_ROOT": args.container_runtime_root,
            "WINGBEAT_CACHE_DIR": args.container_cache_dir,
            "WINGBEAT_DATASET_MANIFEST": args.container_manifest,
            "WINGBEAT_DATASET_MANIFEST_SHA256": args.manifest_sha256,
        },
    )
    if not specs:
        raise RuntimeError("No NVIDIA GPUs were detected")

    summary = {
        **preflight,
        "queue": args.queue,
        "agents": [
            {"gpu_uuid": spec.gpu.uuid, "gpu_name": spec.gpu.name, "max_jobs": 1}
            for spec in specs
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.dry_run:
        supervise_agents(specs)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Agent supervisor failed: {error}", file=sys.stderr)
        raise SystemExit(1)
