"""Quality-gated model promotion pipeline."""

from wingbeat_ml.pipelines.validate import validate_metrics
from wingbeat_ml.tracking import (
    build_lineage,
    promote_artifact,
    registry_target,
    write_lineage,
)


def promote_candidate(
    *,
    metrics,
    minimums,
    registry,
    collection,
    model_path=None,
    artifact_ref=None,
    aliases=None,
    artifact_name=None,
    config_sha256=None,
    dataset_sha256=None,
    git_commit=None,
    entity=None,
    project=None,
    quality_output=None,
    lineage_output=None,
    execute=False,
    wandb_module=None,
):
    """Validate, record lineage and optionally promote a model."""
    if bool(model_path) == bool(artifact_ref):
        raise ValueError(
            "provide exactly one of model_path or artifact_ref"
        )

    quality = validate_metrics(
        metrics,
        minimums,
        output_path=quality_output,
    )

    result = {
        "promoted": False,
        "dry_run": not execute,
        "quality": quality,
        "target": registry_target(registry, collection),
        "aliases": list(aliases or ["candidate"]),
    }

    if not quality["passed"]:
        return result

    lineage = build_lineage(
        metrics=metrics,
        model_path=model_path,
        config_sha256=config_sha256,
        dataset_sha256=dataset_sha256,
        git_commit=git_commit,
        source_artifact=artifact_ref,
    )
    result["lineage"] = lineage

    if lineage_output:
        write_lineage(lineage, lineage_output)

    if not execute:
        return result

    remote = promote_artifact(
        registry=registry,
        collection=collection,
        aliases=result["aliases"],
        project=project,
        entity=entity,
        model_path=model_path,
        artifact_ref=artifact_ref,
        artifact_name=artifact_name,
        metadata=lineage,
        wandb_module=wandb_module,
    )

    result.update(remote)
    result["promoted"] = True
    result["dry_run"] = False
    return result


__all__ = ["promote_candidate"]
