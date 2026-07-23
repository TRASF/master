# Pipeline Helpers and Configuration Ownership

## Objective

Keep canonical pipeline files short and readable. Use `configs/` as the single
place for run customization and `wingbeat_ml.pipelines.helpers` as the single
place for reusable pipeline coordination.

## Structure

```text
src/wingbeat_ml/pipelines/
├── helpers/
│   ├── configuration.py
│   ├── runtime.py
│   ├── components.py
│   └── reporting.py
├── pretrain.py
├── linear_probe.py
├── fine_tune.py
├── evaluate.py
├── export.py
├── promote.py
└── validate.py
```

- `configuration.py` resolves base, model, experiment, profile, environment,
  and command-line layers.
- `runtime.py` prepares run identity, paths, tracking, and runtime state.
- `components.py` assembles existing dataset, model, loss, evaluator, and
  training APIs without containing their algorithms.
- `reporting.py` coordinates existing evaluation and reporting APIs.

Pipeline modules retain only their workflow-specific sequence. Fine-tuning,
for example, retains its two phases, but shared setup and reporting move into
helpers. Domain algorithms remain in their domain packages.

## Configuration ownership

`configs/base.yaml` owns universal defaults. Experiment files own workflow
policy. Profiles own environment-specific policy. A new
`configs/profiles/pilot.yaml` owns the bare pretrain pilot's epochs, batch size,
augmentation settings, and W&B setting.

Machine paths resolve in this order: explicit CLI, environment, profile, then a
portable repository/runtime-relative value. Supported machine overrides are
`WINGBEAT_RUNTIME_ROOT` and `WINGBEAT_DATASET_DIR`. Tracked code and YAML must
not contain `/media/...` or `/app/...` paths.

Required operational keys are validated once after resolution. Python must not
silently substitute epochs, batch sizes, seeds, output paths, tracking policy,
or augmentation probabilities. Model mathematics, tensor dimensions, RNG
stream identifiers, and numerical constants stay in Python.

## Flow and compatibility

Each pipeline reads as: resolve configuration, prepare runtime, build
components, perform workflow-specific preparation, execute, then report.

The bare `python -m wingbeat_ml.pipelines.pretrain` command selects
`pilot.yaml` through the normal resolver. Explicit configuration paths retain
their current behavior. Legacy entrypoints remain thin wrappers.

Configuration errors must occur before TensorFlow initialization and identify
the missing key plus its CLI or environment override.

## Verification and boundaries

Test-first implementation must cover helper responsibilities, removal of
duplicated pipeline setup, pilot YAML ownership, bare and explicit pretrain
contracts, required-key validation, absence of machine paths and inline pilot
policy, legacy delegation, and the canonical-only distribution.

Completion requires focused RED/GREEN tests, the full suite, wheel installation,
installed CLI checks, CPU smoke training, and a bounded bare GPU pilot.

The change may modify canonical pipelines, focused tests, configuration files,
and the new helpers package. It must not modify datasets, runtime artifacts,
`AGENTS.md`, or `CLAUDE.md`, and must not commit generated artifacts.
