# Wingbeat ML

`wingbeat_ml` is an installable training and MLOps package for mosquito
wingbeat classification. MosSongPlus is its first selectable model.

## Install

```bash
python -m pip install -e '.[dev]'
python -m wingbeat_ml --version
```

## Resolve configuration

```bash
wingbeat-ml config resolve \
  --base configs/base.yaml \
  --model configs/models/mossong_plus.yaml \
  --experiment configs/experiments/pretrain.yaml \
  --profile configs/profiles/ci.yaml \
  --output /tmp/wingbeat-resolved.yaml
```

Merge precedence is `base -> model -> experiment -> profile -> --set`.

## Train

Keep runtime outputs outside the repository:

```bash
export WINGBEAT_RUNTIME_ROOT='/media/miru4090s/New Volume2/wingbeat_ml'
export WINGBEAT_DATASET_DIR="$PWD/dataset/MSB/Indoor"
export WINGBEAT_PROFILE="$PWD/configs/profiles/local.yaml"
export WINGBEAT_ENABLE_WANDB=false
bash ops/wandb/train-pretrain.sh
```

## Quality, promotion, and export

```bash
wingbeat-ml quality validate --metrics metrics.json --minimum macro_f1=0.80
wingbeat-ml promote --help
wingbeat-ml export --help
```

## Verify

```bash
PYTHONPATH="$PWD:$PWD/src" pytest -q --tb=short
python -m build --wheel
```

See `ops/wandb/README.md` for W&B Launch setup. Datasets, model files,
logs, caches, credentials, and W&B run data are not committed.

## Architecture

Production code is packaged exclusively from `src/wingbeat_ml`. Historical
repository entrypoints remain as thin compatibility wrappers and are not part
of the wheel. See [`docs/architecture.md`](docs/architecture.md) for component
ownership, dependency rules, the MLOps execution flow, and extension guidance.
