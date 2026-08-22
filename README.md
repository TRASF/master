# Wingbeat ML

`wingbeat_ml` contains two mosquito-audio products: multiclass wingbeat classification and binary mosquito sound-event detection (SED). Shared deployment, analysis, and visualization code supports both.

## Install

```bash
python -m pip install -e '.[dev]'
python -m wingbeat_ml --version
```

## Resolve configuration

```bash
wingbeat-ml config resolve \
  --base configs/base.yaml \
  --model configs/classification/models/mossong_plus.yaml \
  --experiment configs/classification/experiments/pretrain.yaml \
  --profile configs/classification/profiles/ci.yaml \
  --output /tmp/wingbeat-resolved.yaml
```

Merge precedence is `base -> model -> experiment -> profile -> --set`.

## Train

Keep runtime outputs outside the repository:

```bash
export WINGBEAT_RUNTIME_ROOT='/media/miru4090s/New Volume2/wingbeat_ml'
export WINGBEAT_DATASET_DIR="$PWD/dataset/MSB/Indoor"
export WINGBEAT_PROFILE="$PWD/configs/classification/profiles/local.yaml"
export WINGBEAT_ENABLE_WANDB=false
bash ops/wandb/train-pretrain.sh
python -m wingbeat_ml.classification.pipelines.pretrain --defaults_path configs/classification/defaults.yaml --model_cfg_path configs/classification/models/mossong_plus.yaml
```

## Mosquito SED

```bash
./sed.sh train
./sed.sh evaluate
./sed.sh label
./sed.sh detect /path/to/recording.wav
```

`label` and `detect` load the existing trained checkpoint without rebuilding data or training. `run_sed_pipeline.sh` remains the backward-compatible full-cycle command. SED configuration and operating documentation live under `configs/sed/` and `docs/sed/`.

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
ownership, dependency rules, the MLOps execution flow, and extension guidance. See [`docs/REPOSITORY_LAYOUT.md`](docs/REPOSITORY_LAYOUT.md) for product boundaries and compatibility paths.
