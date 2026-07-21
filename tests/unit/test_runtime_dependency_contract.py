"""Runtime dependency declaration regression tests."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dependency_name(requirement: str) -> str:
    value = requirement.strip()

    if value.startswith(("#", "-", "git+", "http://", "https://")):
        return ""

    name = re.split(r"[\s<>=!~;\[]", value, maxsplit=1)[0]
    return name.lower().replace("_", "-")


def _runtime_requirements() -> list[str]:
    pyproject_path = PROJECT_ROOT / "pyproject.toml"

    with pyproject_path.open("rb") as stream:
        data: dict[str, Any] = tomllib.load(stream)

    project = data.get("project", {})
    dependencies = project.get("dependencies")

    if isinstance(dependencies, list):
        return [str(item) for item in dependencies]

    dynamic_names = project.get("dynamic", [])

    if "dependencies" not in dynamic_names:
        return []

    dynamic_spec = (
        data.get("tool", {})
        .get("setuptools", {})
        .get("dynamic", {})
        .get("dependencies", {})
    )

    filenames = dynamic_spec.get("file", [])

    if isinstance(filenames, str):
        filenames = [filenames]

    requirements: list[str] = []

    for filename in filenames:
        path = PROJECT_ROOT / filename

        requirements.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )

    return requirements


def test_resampy_is_declared_as_runtime_dependency() -> None:
    names = {
        _dependency_name(requirement)
        for requirement in _runtime_requirements()
    }

    assert "resampy" in names
