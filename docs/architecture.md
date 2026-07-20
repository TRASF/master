# Wingbeat ML architecture

## Canonical package

All production Python code lives under `src/wingbeat_ml/`. The built wheel
contains this package only; research data, repository tools, and compatibility
modules are deliberately excluded from the distribution.

| Area | Responsibility |
|---|---|
| `config/` | Deterministic loading, validation, profiles, and runtime setup |
| `data/`, `augmentations/` | Audio loading, manifests, splits, datasets, transforms |
| `models/`, `registry.py` | Selectable model builders and component lookup |
| `training/`, `pipelines/` | Reusable training primitives and task orchestration |
| `evaluation/`, `quality/` | Metrics, reports, validation gates |
| `tracking/` | W&B integration, lineage, and promotion metadata |
| `export/` | TFLite conversion, verification, and deployment bundles |

The dependency direction is one-way: pipelines may compose canonical
components, but canonical modules never import legacy repository paths.

## MLOps execution flow

1. Resolve base, model, experiment, and profile configuration.
2. Build deterministic data splits and TensorFlow datasets.
3. Select the model through `wingbeat_ml.registry`.
4. Train, evaluate, and apply quality gates.
5. Record lineage and optionally promote an approved model.
6. Export and verify deployment artifacts.

Use `python -m wingbeat_ml` or the installed `wingbeat-ml` command as the
public entrypoint.

## Compatibility boundary

The modules under `src/framework/`, `src/evaluation/`, `src/io/`,
`src/quantization/`, and `configs/mos_config.py` are source-repository
compatibility wrappers. They preserve historical commands and imports while
delegating to `wingbeat_ml`. They are not shipped in the production wheel and
must not contain independent implementation logic.

Remove a compatibility wrapper only after its callers and parity tests have
been migrated in a separately reviewed change.

## Runtime and artifacts

Resolved configurations, logs, checkpoints, W&B files, exports, and TensorFlow
caches belong outside Git under `WINGBEAT_RUNTIME_ROOT`. Local, CI, and W&B
Launch jobs share `ops/wandb/train-pretrain.sh` and select behavior through
configuration profiles and environment overrides.

## Extending the system

To add a model, implement its builder under `wingbeat_ml.models`, register its
stable identifier in `wingbeat_ml.registry`, add a model YAML file, and test the
builder and registry lookup. Add data transforms, metrics, trackers, or
exporters to their corresponding canonical module and compose them from a
pipeline; do not add new implementation to a compatibility wrapper.
