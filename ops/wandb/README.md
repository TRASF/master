# W&B Launch operations

`train-pretrain.sh` is the shared local, CI, and Launch training entrypoint.
It resolves canonical configuration before changing into the runtime root,
so generated models, logs, W&B files, and TensorFlow caches remain outside
the Git checkout.

## Local smoke or training run

```bash
export WINGBEAT_RUNTIME_ROOT='/media/miru4090s/New Volume2/wingbeat_ml'
export WINGBEAT_DATASET_DIR="$PWD/dataset/MSB/Indoor"
export WINGBEAT_PROFILE="$PWD/configs/profiles/local.yaml"
export WINGBEAT_ENABLE_WANDB=false
bash ops/wandb/train-pretrain.sh
```

The selected profile must define `dataset.train_dir` and `wandb.enabled`
because the entrypoint safely overrides those existing fields.

## Launch agent

1. Authenticate without committing the API key:

   ```bash
   wandb login
   ```

2. Create a Docker queue named `wingbeat-training` in W&B. Configure its
   Docker resource with the dataset mounted read-only at
   `/app/dataset` and runtime storage mounted at `/runtime`. Pass these
   environment values through the queue or job configuration:

   ```text
   WINGBEAT_DATASET_DIR=/app/dataset/MSB/Indoor
   WINGBEAT_RUNTIME_ROOT=/runtime
   WINGBEAT_ENABLE_WANDB=true
   ```

3. Install the bounded agent configuration and start the agent:

   ```bash
   mkdir -p ~/.config/wandb
   cp ops/wandb/launch-config.yaml ~/.config/wandb/launch-config.yaml
   wandb launch-agent --queue wingbeat-training --max-jobs 1
   ```

4. Submit the current Git repository as a Launch job:

   ```bash
   wandb launch \
     --uri "$(git remote get-url origin)" \
     --queue wingbeat-training \
     --project MosSongPlus \
     --job-name wingbeat-pretrain \
     --dockerfile Dockerfile \
     --entry-point "bash ops/wandb/train-pretrain.sh"
   ```

`WANDB_API_KEY` belongs in the agent environment or secret manager, never
in this repository.
