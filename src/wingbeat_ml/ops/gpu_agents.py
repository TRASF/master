"""GPU discovery and one-agent-per-GPU process supervision."""

from dataclasses import dataclass
import os
import signal
import subprocess
import time


@dataclass(frozen=True)
class GpuDevice:
    uuid: str
    name: str


@dataclass(frozen=True)
class AgentSpec:
    gpu: GpuDevice
    queue: str
    max_jobs: int
    environment: dict[str, str]

    @property
    def command(self):
        return (
            "wandb",
            "launch-agent",
            "--queue",
            self.queue,
            "--max-jobs",
            str(self.max_jobs),
        )


def parse_gpu_query(output):
    """Parse ``nvidia-smi --query-gpu=uuid,name`` CSV output."""
    devices = []
    for line in str(output).splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2 or not parts[0]:
            raise ValueError(f"Invalid nvidia-smi GPU row: {line!r}")
        devices.append(GpuDevice(uuid=parts[0], name=parts[1]))
    return devices


def discover_gpus(*, run=subprocess.run):
    """Return visible physical GPUs by stable UUID."""
    result = run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_gpu_query(result.stdout)


def build_agent_specs(gpus, *, queue, job_environment=None):
    """Create one exclusive, single-job Launch agent for each GPU."""
    seen = set()
    specs = []
    for gpu in gpus:
        if gpu.uuid in seen:
            raise ValueError(f"Duplicate GPU UUID: {gpu.uuid}")
        seen.add(gpu.uuid)
        environment = dict(os.environ)
        environment.update(job_environment or {})
        environment["CUDA_VISIBLE_DEVICES"] = gpu.uuid
        environment["NVIDIA_VISIBLE_DEVICES"] = gpu.uuid
        environment["WINGBEAT_AGENT_GPU_UUID"] = gpu.uuid
        specs.append(
            AgentSpec(
                gpu=gpu,
                queue=queue,
                max_jobs=1,
                environment=environment,
            )
        )
    return specs


def supervise_agents(specs, *, poll_seconds=5.0, max_backoff=60.0):
    """Run and restart desired agents until SIGINT or SIGTERM."""
    processes = {}
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    previous = {
        sig: signal.signal(sig, request_stop)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    failures = {spec.gpu.uuid: 0 for spec in specs}
    try:
        while not stopping:
            for spec in specs:
                process = processes.get(spec.gpu.uuid)
                if process is not None and process.poll() is None:
                    continue
                if process is not None:
                    failures[spec.gpu.uuid] += 1
                    delay = min(
                        max_backoff,
                        2 ** min(failures[spec.gpu.uuid], 6),
                    )
                    time.sleep(delay)
                processes[spec.gpu.uuid] = subprocess.Popen(
                    spec.command,
                    env=spec.environment,
                )
            time.sleep(poll_seconds)
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


__all__ = [
    "AgentSpec",
    "GpuDevice",
    "build_agent_specs",
    "discover_gpus",
    "parse_gpu_query",
    "supervise_agents",
]
