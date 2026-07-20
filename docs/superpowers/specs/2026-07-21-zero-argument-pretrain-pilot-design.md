# Zero-argument pretraining pilot

## Objective

Make `python -m wingbeat_ml.pipelines.pretrain` run a safe local pilot while
preserving all explicit-path behavior.

## Behavior

With no configuration arguments, the entrypoint finds the canonical project
configuration, resolves base, MosSong+, pretrain, and local layers, and applies
five pilot overrides: the repository Indoor dataset, five epochs, batch size
256, disabled noise overlay, and disabled W&B.

The runtime root comes from `WINGBEAT_RUNTIME_ROOT` when set and otherwise uses
the approved local path `/media/miru4090s/New Volume2/wingbeat_ml`. Each command
creates and enters a timestamped directory beneath `pilots/`, so relative
checkpoints and reports remain outside Git and never overwrite an earlier run.

Supplying either configuration argument retains the historical defaults for
the other argument and does not activate pilot resolution.

## Verification

Unit tests cover configuration resolution, bare dispatch, and explicit-path
compatibility. The full suite, wheel installation, CPU smoke training, and one
real five-epoch bare-command GPU run must pass.
