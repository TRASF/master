# Training Throughput and W&B GPU-Agent Design

## Objective

Increase research throughput and improve sample-level model quality without
coupling independent lab computers into one synchronous training job.

The system must:

- run one independent experiment per GPU;
- dynamically adapt as GPUs are added or removed from either lab computer;
- allow experiments and sweeps to be created from W&B;
- keep local datasets and caches fast and reproducible;
- reduce terminal noise without hiding actionable warnings;
- measure startup, input, training, validation, and logging costs separately;
- correct class-weight resolution before judging model quality; and
- preserve file-level evaluation as an explicitly disabled compatibility path.

## Current Evidence

The pilot has approximately 55 optimizer steps per epoch. The trainer already
executes up to 20 batches inside one compiled TensorFlow call, so replacing
epoch terminology with step terminology would not itself improve speed.

The first epoch takes roughly 14 seconds while later epochs take approximately
0.9 seconds. The difference is consistent with graph tracing, GPU kernel
initialization, and initial dataset-cache population. Performance reporting
must separate this cold-start cost from steady-state throughput.

The current model has roughly 350,000 parameters. Synchronizing such a small
model across ordinary lab Ethernet is unlikely to outperform independent jobs
on each GPU. Independent experiment parallelism is therefore the default.

The current class-weight configuration uses class names, but the resolver looks
up numeric keys and silently substitutes `1.0`. The observed log containing
eleven `1.0` weights confirms that intended imbalance correction is inactive.

## Deployment Topology

Two lab computers share a network and internet access:

- Machine A currently has one GPU and will soon have two.
- Machine B has two GPUs.
- Both machines keep synchronized local copies of the dataset.
- Both machines run Docker with NVIDIA Container Toolkit.
- Both authenticate to the same W&B entity and consume the same Launch queue.

The target capacity is three concurrent experiments now and four after the
second GPU is installed in Machine A.

## W&B Control Plane

The canonical training container is published as a versioned W&B Job. Normal
experiments and sweeps are created from the W&B App and submitted to one shared
Launch queue.

Each lab computer runs a host supervisor. At startup it:

1. queries visible GPUs with `nvidia-smi`;
2. validates Docker GPU access;
3. validates W&B authentication and queue access;
4. verifies the local dataset manifest checksum;
5. verifies runtime and cache directories are writable; and
6. starts one Launch agent per visible GPU.

Each agent has `max_jobs: 1`, listens to the same queue, and receives one
exclusive `CUDA_VISIBLE_DEVICES` value. The Docker queue passes that existing
environment value, plus `NVIDIA_VISIBLE_DEVICES`, into each job. The queue uses
`--gpus all` to enable the NVIDIA runtime, but TensorFlow sees only the GPU UUID
assigned to that agent.

A shared Docker queue has one static bind-mount configuration. Both computers
therefore expose their different local disks through the same host mount
points: `/srv/wingbeat/dataset` and `/srv/wingbeat/runtime`. Containers receive
those mounts as `/data` and `/runtime`; agent-provided environment variables
use the container paths. This preserves local-disk performance while keeping
one queue usable by both machines.

The host supervisor reconciles desired and running agents. Restarting it after
installing another GPU creates the additional worker slot without source-code
or queue changes. It never starts two agents for the same physical GPU.

The default mode is one experiment per GPU. Local `MirroredStrategy` remains an
explicit future/diagnostic option, not the scheduling default. Multi-worker
training across computers is outside the critical path; the training core may
retain a future-compatible strategy boundary without deploying synchronous
multi-worker execution in this phase.

## Dataset and Cache Boundary

Each computer trains from a local dataset copy. A canonical manifest records
relative paths, sizes, and content hashes. Every run logs the manifest checksum
and refuses to start when the local copy does not match the approved manifest.

The physical dataset path is host configuration. The shared Docker queue binds
its standardized host view at `/data`, which is supplied to jobs through
`WINGBEAT_DATASET_DIR`; physical disk paths are not written into source code,
Docker images, or experiment YAML.

The cache root is host configuration. Cache keys are stable and derived from:

- dataset manifest checksum;
- sample rate and segment duration;
- deterministic preprocessing configuration; and
- a cache schema/version identifier.

Expensive deterministic loading and preprocessing happen before the reusable
cache boundary. Stochastic segmentation and augmentation happen after it.
Cache creation uses a single-writer lock and atomic publication so concurrent
jobs cannot consume a partial cache. Jobs reuse completed immutable caches.

## Training Semantics

Epochs remain the outer unit because they represent dataset coverage and are a
natural boundary for validation, checkpointing, early stopping, and reporting.

The trainer also records:

- `global_step`: total optimizer updates;
- `steps_per_epoch`: batches actually consumed in the epoch;
- `steps_per_call`: batches executed by one compiled TensorFlow call; and
- global examples processed.

`steps_per_call` moves from a hard-coded value of 20 to validated YAML. It is a
performance knob, not a change to optimization semantics.

The dedicated fine-tuning warmup subsystem is removed. This includes
`warmup_epochs`, `warmup_augment_p`, the alternate warmup dataset, warmup loop,
and warmup-only log fields. Fine-tuning uses the canonical training runner. A
future gradual-unfreezing schedule must be explicit and independently tested.

Runtime graph/cache initialization is reported as startup cost and is not
represented as a training warmup phase.

## Compute Policy

Compute behavior is configuration-owned:

```yaml
performance:
  precision: auto
  steps_per_call: 20
  jit_compile: false
  profiler:
    enabled: false
    start_step: 10
    num_steps: 10
```

`precision: auto` selects mixed FP16 only on a supported GPU and otherwise uses
FP32. Numerically sensitive reductions, losses, metrics, and model outputs stay
FP32. The custom loop applies loss scaling correctly.

Mixed precision is accepted only after fixed-seed metric-parity and throughput
benchmarks. XLA remains opt-in because compilation cost may exceed its benefit
for this small model.

Reproducible and maximum-throughput profiles remain distinct. Reproducibility,
including deterministic data and operations, remains the default. Sweeps may
explicitly select the throughput profile, and every run records the choice.

## Performance Measurement

Timing separates:

- process and TensorFlow startup;
- graph tracing and kernel initialization;
- cold or reused cache preparation;
- training device time;
- validation time;
- checkpoint time;
- W&B/local logging time; and
- total wall time.

Reported throughput distinguishes cold-start and steady-state examples per
second. Profiler capture is opt-in, covers a short configured step window, and
starts after initialization. The dataset loader is benchmarkable independently
with model computation removed, and synthetic-data training is available to
identify input-bound execution.

## Logging Policy

Logging configuration is explicit:

```yaml
logging:
  console: normal
  epoch_interval: 1
  model_summary: false
  classification_report: file
  jsonl: true
```

Console modes are:

- `quiet`: errors and final outcome;
- `normal`: compact startup, one line per configured epoch interval,
  checkpoint changes, and final outcome; and
- `verbose`: model summary and detailed diagnostics.

There is no per-batch terminal output. TensorFlow INFO diagnostics are hidden
by default, while warnings and errors remain visible. Detailed timing,
per-class metrics, resolved configuration, global step, and artifact paths go
to append-only JSONL and W&B at epoch boundaries. Because logging occurs once
per epoch, an asynchronous logging subsystem is unnecessary.

## Sample-Level Evaluation

Sample/window evaluation is the canonical quality boundary:

```yaml
evaluation:
  sample_level:
    enabled: true
  file_level:
    enabled: false
```

Training batches report loss and accuracy for diagnostics. Validation samples
are evaluated after each epoch. Checkpointing, early stopping, learning-rate
scheduling, and W&B sweeps maximize validation sample macro-F1. The test split
is evaluated exactly once after training using the best checkpoint and never
influences sweep selection.

File-level evaluation remains a compatibility feature but is disabled by
default. When disabled it does not load source files, run inference, create
reports, print metrics, or log W&B tables. It is excluded from normal quality
gates and sweeps.

## Class-Weight Policy

Class weighting becomes explicit:

```yaml
class_weights:
  mode: auto
```

Supported modes are:

- `auto`: use weights computed from the training split;
- `manual`: resolve configured class names through the canonical label map;
  unknown, duplicated, or misspelled names are configuration errors; and
- `off`: train without sample weights.

There is no silent `1.0` fallback. Resolved class counts and weights are logged
to JSONL and W&B and embedded in the resolved run configuration.

## Sweep Workflow

The recommended sequence is:

1. run the corrected baseline across three or four seeds;
2. quantify variance in validation sample macro-F1;
3. create a bounded Bayesian sweep in the W&B App using the versioned Job and
   shared Launch queue;
4. set scheduler concurrency to the number of healthy GPU agents;
5. retest the best configuration across fresh seeds; and
6. promote only models meeting sample-level quality gates.

The queue and agents control placement. Sweep configuration controls model and
training hyperparameters. Host paths, GPU indices, credentials, and secrets are
never sweep parameters.

## Failure Handling

Agent preflight failures prevent that GPU slot from consuming jobs and report a
clear host-local diagnostic. A failed training job affects only its assigned
GPU; other agents continue. The supervisor restarts failed idle agents with
bounded backoff.

Termination handling marks the W&B run appropriately, flushes JSONL and W&B
epoch metrics, and preserves the latest recoverable checkpoint. Only the job
that owns a GPU writes to its unique runtime directory. Credentials are kept in
host environment/credential storage and are never committed.

## Verification

Unit and contract tests cover:

- class-name and automatic weight resolution;
- rejection of unknown manual weight names;
- disabled file-level evaluation performing no work;
- logging modes and epoch intervals;
- stable cache keys and concurrent cache publication;
- precision selection, loss scaling, and FP32 outputs;
- global-step and `steps_per_call` accounting;
- GPU discovery and unique per-GPU agent generation; and
- dataset-manifest mismatch rejection.

Integration checks cover:

- full existing tests and wheel installation;
- CPU smoke training;
- single-GPU FP32 and mixed-precision smoke training;
- two concurrent jobs pinned to distinct GPUs;
- W&B offline job logging before live queue tests;
- cold-cache, warm-cache, and synthetic-data benchmarks;
- fixed-seed FP32/mixed-precision metric parity; and
- a live three-agent sweep, followed by automatic four-agent discovery after
  Machine A receives its second GPU.

## Acceptance Criteria

The phase passes when:

- one independent job can run on every detected GPU without assignment
  overlap;
- adding a GPU requires only restarting/reconciling the host supervisor;
- experiments and Launch sweeps can be submitted from W&B;
- both computers reject mismatched datasets before training;
- disabled file-level evaluation performs no I/O or inference;
- resolved class weights are correct and never silently defaulted;
- sample-level validation macro-F1 does not regress against the corrected FP32
  baseline;
- mixed precision is enabled automatically only after parity validation;
- steady-state throughput is preserved or improved;
- normal console logging has no per-batch output and negligible measured
  overhead; and
- unrelated repository and runtime files remain untouched.

## Non-Goals

This phase does not:

- synchronize gradients across the two computers;
- deploy Kubernetes, Slurm, or a multi-worker TensorFlow cluster;
- make file-level evaluation part of model selection;
- optimize on test metrics;
- introduce a custom hyperparameter scheduler when the standard W&B scheduler
  is sufficient; or
- change the model architecture without a separate quality experiment.
