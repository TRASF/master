# Phase 10: CI and W&B Launch Design

## Objective

Complete the MLOps foundation without changing model, dataset, training,
evaluation, or export behavior.

## Deliverables

- One Python 3.12 GitHub Actions workflow that runs the full tests, builds
  and installs the wheel, checks the CLI, and performs CPU smoke training.
- One shared `ops/wandb/train-pretrain.sh` entrypoint for local, CI, and
  W&B Launch execution.
- One bounded Docker Launch-agent configuration with a single worker.
- Root and operations documentation with no committed credentials.

## Runtime boundary

All resolved configurations, logs, models, W&B files, and TensorFlow
caches live beneath `WINGBEAT_RUNTIME_ROOT`. The approved local default is
`/media/miru4090s/New Volume2/wingbeat_ml`; CI overrides it with temporary
runner storage.

## Verification

Phase 10 passes only when YAML and shell validation, the complete test
suite, wheel installation, CLI checks, and CPU fixture smoke training all
succeed.

## Non-goals

This phase does not refactor internal training/configuration code, remove
compatibility wrappers, alter model behavior, or touch real datasets and
runtime artifacts.
