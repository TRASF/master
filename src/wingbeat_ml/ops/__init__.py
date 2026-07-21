"""Operational helpers for lab GPU workers."""

from wingbeat_ml.ops.gpu_agents import (
    AgentSpec,
    GpuDevice,
    build_agent_specs,
    discover_gpus,
    parse_gpu_query,
)
from wingbeat_ml.ops.preflight import require_manifest_checksum

__all__ = [
    "AgentSpec",
    "GpuDevice",
    "build_agent_specs",
    "discover_gpus",
    "parse_gpu_query",
    "require_manifest_checksum",
]
