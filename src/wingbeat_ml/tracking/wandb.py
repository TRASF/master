"""Small adapter around the W&B Artifact Registry API."""

import re
from pathlib import Path


def registry_target(registry, collection):
    """Build a W&B Registry collection target path."""
    registry = str(registry).strip()
    collection = str(collection).strip()

    if not registry or "/" in registry:
        raise ValueError("registry must be one non-empty path segment")
    if not collection or "/" in collection:
        raise ValueError("collection must be one non-empty path segment")

    return f"wandb-registry-{registry}/{collection}"


def _artifact_name(model_path):
    name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "-",
        Path(model_path).stem,
    ).strip("-")
    return name or "wingbeat-model"


def promote_artifact(
    *,
    registry,
    collection,
    aliases,
    project,
    entity=None,
    model_path=None,
    artifact_ref=None,
    artifact_name=None,
    metadata=None,
    wandb_module=None,
):
    """Log or reuse an artifact and link it into W&B Registry."""
    if bool(model_path) == bool(artifact_ref):
        raise ValueError(
            "provide exactly one of model_path or artifact_ref"
        )
    if not project:
        raise ValueError("project is required for remote promotion")

    if wandb_module is None:
        try:
            import wandb as wandb_module
        except ImportError as error:
            raise RuntimeError(
                "W&B promotion requires the wandb package"
            ) from error

    target = registry_target(registry, collection)
    aliases = list(aliases or ["candidate"])

    with wandb_module.init(
        entity=entity,
        project=project,
        job_type="model-promotion",
    ) as run:
        if model_path:
            path = Path(model_path)
            if not path.is_file():
                raise FileNotFoundError(
                    f"model file not found: {model_path}"
                )

            artifact = wandb_module.Artifact(
                name=artifact_name or _artifact_name(path),
                type="model",
                metadata=metadata or {},
            )
            artifact.add_file(str(path))
            artifact = run.log_artifact(artifact)

            wait = getattr(artifact, "wait", None)
            if callable(wait):
                wait()
        else:
            artifact = run.use_artifact(artifact_ref)

        linked = run.link_artifact(
            artifact=artifact,
            target_path=target,
            aliases=aliases,
        )

    reference = (
        getattr(linked, "qualified_name", None)
        or getattr(linked, "name", None)
        or artifact_ref
    )

    return {
        "artifact": reference,
        "target": target,
        "aliases": aliases,
    }


__all__ = ["promote_artifact", "registry_target"]
