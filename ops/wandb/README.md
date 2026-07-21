# W&B Launch operations

The default topology is one independent Launch agent and one job per detected
GPU. Restarting the supervisor after installing another GPU adds the new slot;
no repository or queue change is required.

`train-pretrain.sh` is the shared local, CI, and Launch training entrypoint. It
resolves canonical configuration before changing into the runtime root, so
models, JSONL metrics, W&B files, and TensorFlow caches remain outside Git.

Normal runs do not pipe stdout through `tee`. Set
`WINGBEAT_CAPTURE_CONSOLE_LOG=true` only when a duplicate terminal log is
needed.

## Local smoke or training run

```bash
export WINGBEAT_RUNTIME_ROOT='/media/miru4090s/New Volume2/wingbeat_ml'
export WINGBEAT_DATASET_DIR="$PWD/dataset/MSB/Indoor"
export WINGBEAT_PROFILE="$PWD/configs/profiles/local.yaml"
export WINGBEAT_ENABLE_WANDB=false
export WINGBEAT_CACHE_DIR="$WINGBEAT_RUNTIME_ROOT/dataset/.tf_cache"
bash ops/wandb/train-pretrain.sh
```

## Shared Docker queue

Use `ops/wandb/docker-queue-config.json` as the Docker queue configuration in
the W&B App. It forwards the GPU UUID and job environment from each agent into
its container.

Docker bind mounts cannot vary by agent in one shared queue, so both computers
must expose their synchronized local copies at these host paths:

- `/srv/wingbeat/dataset`
- `/srv/wingbeat/runtime`

The dataset mount must contain `dataset-manifest.json`. The runtime mount must
be writable. These paths may be bind mounts backed by different physical disks
on the two computers.

Each job writes to an isolated `runs/<run-id>` directory. On exit, including a
failed training exit, it writes checksums and publishes `.artifact-ready.json`
last. The shared cache remains outside run directories and is not archived.

## Per-GPU Launch agents

1. Authenticate once without committing the API key:

   ```bash
   wandb login
   ```

2. Copy and edit the host environment template:

   ```bash
   mkdir -p ~/.config/wingbeat ~/.config/wandb
   cp ops/wandb/agents.env.example ~/.config/wingbeat/agents.env
   cp ops/wandb/launch-config.yaml ~/.config/wandb/launch-config.yaml
   ```

3. Verify discovery, Docker GPU access, W&B authentication, local paths, and
   the approved dataset checksum without starting agents:

   ```bash
   set -a
   source ~/.config/wingbeat/agents.env
   set +a
   bash ops/wandb/start-gpu-agents.sh --dry-run
   ```

4. Start all current GPU agents:

   ```bash
   bash ops/wandb/start-gpu-agents.sh
   ```

   For persistent startup, copy `wingbeat-agents.service` to
   `~/.config/systemd/user/`, then run:

   ```bash
   systemctl --user daemon-reload
   systemctl --user enable --now wingbeat-agents.service
   ```

5. Publish or submit the repository as a W&B Launch Job:

   ```bash
   wandb launch \
     --uri "$(git remote get-url origin)" \
     --queue wingbeat-training \
     --project MosSongPlus \
     --job-name wingbeat-pretrain \
     --dockerfile Dockerfile \
     --entry-point "bash ops/wandb/train-pretrain.sh"
   ```

The queue configuration names environment variables without assigning values.
W&B therefore takes each value from the GPU-pinned agent and passes it into the
job container. `CUDA_VISIBLE_DEVICES` and `NVIDIA_VISIBLE_DEVICES` restrict
TensorFlow to that agent's one GPU.

`WANDB_API_KEY` belongs in the machine credential store or secret manager,
never in Git, the Docker image, or YAML.

## Central artifact owner

miru4090s is the permanent artifact owner. Compute-only machines use local
staging during training, then transfer only finalized run directories. The
source receives `.artifact-synced.json` only after miru4090s independently
verifies every file checksum.

On miru4090s, bind `/srv/wingbeat/runtime` directly to the permanent experiment
disk, set `WINGBEAT_ARTIFACT_DESTINATION=/srv/wingbeat/runtime`, and leave
`WINGBEAT_ARTIFACT_OWNER` blank. On every compute-only machine, bind the same
standard path to fast local scratch and set:

```bash
WINGBEAT_ARTIFACT_OWNER=your-ssh-user@miru4090s
WINGBEAT_ARTIFACT_DESTINATION=/media/miru4090s/New Volume1/MosSongPlus_experiments
```

Passwordless SSH and `rsync` are required on compute-only machines. Verify the
connection before enabling the service:

```bash
ssh "$WINGBEAT_ARTIFACT_OWNER" hostname
rsync --version
```

Install the non-destructive synchronization service on both machines:

```bash
cp ops/wandb/wingbeat-artifact-sync.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now wingbeat-artifact-sync.service
```

The service never deletes staging data. After a run has both the source
`.artifact-synced.json` marker and the owner `.artifact-verified.json` marker,
cleanup remains an explicit operator action.
