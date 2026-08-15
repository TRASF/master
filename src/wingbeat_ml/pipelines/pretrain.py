from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from wingbeat_ml.config import validate_config
from wingbeat_ml.pipelines.helpers import (
    build_supervised_components,
    evaluate_training_run,
    load_pipeline_configuration,
    make_epoch_printer,
    prepare_default_pilot,
    prepare_training_run,
)
from wingbeat_ml.pipelines.train import run_training
from wingbeat_ml.training.adabn import AdaBN


def _resolved_dataset_root(path: Optional[str]) -> Optional[Path]:
    """Return a stable comparable dataset root."""
    if path is None:
        return None

    value = os.path.expandvars(os.path.expanduser(str(path)))
    return Path(value).resolve(strict=False)


def _uses_external_validation_domain(
    train_dir: str,
    val_dir: Optional[str],
) -> bool:
    """Return True when validation is configured from another dataset root."""
    train_root = _resolved_dataset_root(train_dir)
    val_root = _resolved_dataset_root(val_dir)

    return (
        train_root is not None
        and val_root is not None
        and train_root != val_root
    )


def _target_adabn_checkpoint_path(
    source_checkpoint: Union[str, os.PathLike],
) -> Path:
    """Return a separate Keras-compatible AdaBN checkpoint path."""
    source = Path(source_checkpoint)
    suffix = ".weights.h5"

    if source.name.endswith(suffix):
        base = source.name[: -len(suffix)]
        filename = f"{base}.adabn-target{suffix}"
    else:
        filename = f"{source.stem}.adabn-target.weights.h5"

    return source.with_name(filename)


def _metric_value(metrics: Any, *names: str) -> float:
    """Extract one scalar metric from common evaluator return forms."""
    if isinstance(metrics, Mapping):
        for name in names:
            if name in metrics:
                value = metrics[name]
                if hasattr(value, "numpy"):
                    value = value.numpy()
                return float(value)

    for name in names:
        if hasattr(metrics, name):
            value = getattr(metrics, name)
            if hasattr(value, "numpy"):
                value = value.numpy()
            return float(value)

    available = (
        list(metrics.keys())
        if isinstance(metrics, Mapping)
        else type(metrics).__name__
    )
    raise KeyError(
        f"Could not find any of {names!r} in evaluator result. "
        f"Available: {available!r}"
    )


def _evaluate_validation(evaluator, validation_dataset):
    """Evaluate the current in-memory model on validation data."""
    return evaluator.evaluate_epoch(validation_dataset)


def _select_adabn_checkpoint(
    *,
    model,
    evaluator,
    validation_dataset,
    source_checkpoint: Union[str, os.PathLike],
    console: str,
    min_f1_gain: float = 0.0,
) -> Path:
    """Use target AdaBN only when target validation macro-F1 improves.

    This function never uses test data.

    Procedure:
      1. Load the selected/best source checkpoint.
      2. Evaluate target validation with source BN statistics.
      3. Calibrate target BN statistics from target validation inputs.
      4. Evaluate the same validation set with target BN statistics.
      5. Persist/use target BN only if validation macro-F1 improves.
         Otherwise retain the original source checkpoint.

    The validation labels are used only for model selection, which is the
    intended role of a validation set. Test data remains untouched.
    """
    source_checkpoint = Path(source_checkpoint)

    if not source_checkpoint.exists():
        raise FileNotFoundError(
            "Selected source checkpoint does not exist: "
            f"{source_checkpoint}"
        )

    # Always start from the actual best source checkpoint.
    model.load_weights(source_checkpoint)

    if console != "quiet":
        print(
            "\n>>> External validation domain detected."
            "\n>>> Running guarded AdaBN selection..."
        )

    # ---------------------------------------------------------
    # Baseline: source BN statistics on the target validation set
    # ---------------------------------------------------------
    source_val = _evaluate_validation(
        evaluator,
        validation_dataset,
    )
    source_f1 = _metric_value(
        source_val,
        "macro_f1",
        "val_macro_f1",
        "f1",
    )

    if console != "quiet":
        print(
            f">>> Target validation macro-F1 with SOURCE BN: "
            f"{source_f1:.6f}"
        )

    # ---------------------------------------------------------
    # Candidate: exact target-domain BN statistics
    # ---------------------------------------------------------
    adabn = AdaBN(model)
    adabn.calibrate(validation_dataset)

    with adabn.domain("target"):
        target_val = _evaluate_validation(
            evaluator,
            validation_dataset,
        )
        target_f1 = _metric_value(
            target_val,
            "macro_f1",
            "val_macro_f1",
            "f1",
        )

    if console != "quiet":
        print(
            f">>> Target validation macro-F1 with TARGET BN: "
            f"{target_f1:.6f}"
        )

    required_f1 = source_f1 + float(min_f1_gain)

    # Prefer the source model on a tie. AdaBN must earn its use.
    if target_f1 <= required_f1:
        if console != "quiet":
            print(
                ">>> AdaBN rejected: target BN did not improve "
                "target validation macro-F1."
                f"\n>>> Required > {required_f1:.6f}; "
                f"observed {target_f1:.6f}."
                "\n>>> Final target evaluation will use SOURCE BN."
            )

        # calibrate() and domain() both restore the source bank.
        model.load_weights(source_checkpoint)
        return source_checkpoint

    # ---------------------------------------------------------
    # AdaBN passed validation. Save it separately.
    # ---------------------------------------------------------
    target_checkpoint = _target_adabn_checkpoint_path(
        source_checkpoint
    )

    with adabn.domain("target") as target_model:
        target_model.save_weights(target_checkpoint)

    if console != "quiet":
        print(
            ">>> AdaBN accepted."
            f"\n>>> Validation macro-F1 gain: "
            f"{target_f1 - source_f1:+.6f}"
            f"\n>>> Target checkpoint: {target_checkpoint}"
            f"\n>>> Source checkpoint remains unchanged: "
            f"{source_checkpoint}"
        )

    return target_checkpoint


def train_supervised(
    defaults_path: Union[str, os.PathLike] = "configs/defaults.yaml",
    model_cfg_path: Union[str, os.PathLike] = "configs/model.yaml",
    save_path: Optional[str] = None,
    results_dir: Optional[str] = None,
):
    """Run canonical supervised pretraining."""
    config, model_config = load_pipeline_configuration(
        defaults_path,
        model_cfg_path,
    )

    run = prepare_training_run(
        config,
        mode="Pretrain",
        save_path=save_path,
        results_dir=results_dir,
    )

    components = build_supervised_components(
        config,
        model_config,
        show_class_counts=True,
    )

    app_cfg = validate_config(config)
    epochs = app_cfg.train.epochs
    print(f"\nModel Summary:\n{components.model.summary()}")
    print(f"Output activation: {app_cfg.model.output_activation}")
    print(f"\nStarting training for {epochs} epochs...")

    # ============================================================
    # 1. SOURCE TRAINING
    # ============================================================
    run_training(
        components.model,
        components.train_dataset,
        config,
        evaluate_epoch=lambda: (
            components.evaluator.evaluate_epoch(
                components.validation_dataset
            )
        ),
        on_epoch_end=make_epoch_printer(
            config,
            detailed=True,
        ),
        class_weights=components.class_weights,
        save_path=run.save_path,
    )

    source_checkpoint = Path(run.save_path)
    target_evaluation_checkpoint = source_checkpoint

    # ============================================================
    # 2. OPTIONAL, GUARDED TARGET AdaBN
    # ============================================================
    external_validation_domain = (
        _uses_external_validation_domain(
            app_cfg.dataset.train_dir,
            app_cfg.dataset.val_dir,
        )
    )

    if app_cfg.adabn.enabled:
        if external_validation_domain:
            target_evaluation_checkpoint = (
                _select_adabn_checkpoint(
                    model=components.model,
                    evaluator=components.evaluator,
                    validation_dataset=(
                        components.validation_dataset
                    ),
                    source_checkpoint=source_checkpoint,
                    console=app_cfg.logging.console,
                    min_f1_gain=0.0,
                )
            )
        elif app_cfg.logging.console != "quiet":
            print(
                "\n>>> AdaBN enabled, but train_dir and val_dir "
                "resolve to the same dataset root."
                "\n>>> AdaBN skipped."
            )

    # ============================================================
    # 3. FINAL EVALUATION
    # ============================================================
    #
    # validation/test: target domain -> selected target checkpoint
    # training:        source domain -> original source checkpoint
    #
    # IMPORTANT:
    # evaluate_training_run() needs the small patch documented in
    # evaluate_training_run_patch.txt so that file-level training
    # evaluation reloads source_checkpoint before evaluating train.
    evaluate_training_run(
        model=components.model,
        evaluator=components.evaluator,
        dataset_builder=components.dataset_builder,
        config=config,
        checkpoint_path=target_evaluation_checkpoint,
        training_checkpoint_path=source_checkpoint,
        results_dir=run.results_dir,
        artifact_name="mossongplus-pretrained",
        validation_dataset=components.validation_dataset,
        test_dataset=components.test_dataset,
    )


def main(args=None):
    """Run pretraining, selecting the pilot profile when no paths are given."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--defaults_path", type=str)
    parser.add_argument("--model_cfg_path", type=str)
    parsed_args, _ = parser.parse_known_args(args)

    if (
        parsed_args.defaults_path is None
        and parsed_args.model_cfg_path is None
    ):
        defaults_path, model_cfg_path, runtime_root = (
            prepare_default_pilot()
        )
        os.environ["WINGBEAT_RUNTIME_ROOT"] = str(runtime_root)
        os.chdir(runtime_root)
    else:
        defaults_path = (
            parsed_args.defaults_path or "configs/defaults.yaml"
        )
        model_cfg_path = (
            parsed_args.model_cfg_path or "configs/model.yaml"
        )

    train_supervised(
        defaults_path=defaults_path,
        model_cfg_path=model_cfg_path,
    )


if __name__ == "__main__":
    main()
