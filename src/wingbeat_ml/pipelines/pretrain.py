"""Canonical pretraining pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Union

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
from wingbeat_ml.training.domain_adaptation import AdaBN


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

    print(f"Output activation: {app_cfg.model.output_activation}")
    print(f"\nStarting training for {epochs} epochs...")

    # ---------------------------------------------------------
    # 1. SOURCE TRAINING
    # ---------------------------------------------------------

    run_training(
        components.model,
        components.train_dataset,
        config,
        evaluate_epoch=lambda: components.evaluator.evaluate_epoch(
            components.validation_dataset
        ),
        on_epoch_end=make_epoch_printer(
            config,
            detailed=True,
        ),
        class_weights=components.class_weights,
        save_path=run.save_path,
    )

    # ---------------------------------------------------------
    # 2. EXPLICITLY RESTORE BEST SOURCE CHECKPOINT
    # ---------------------------------------------------------

    components.model.load_weights(run.save_path)

    print(
        f"\nRestored best source checkpoint: "
        f"{run.save_path}"
    )

    # ---------------------------------------------------------
    # 3. FINAL SOURCE EVALUATION
    #
    # This happens BEFORE target adaptation and uses the
    # untouched source BN statistics.
    # ---------------------------------------------------------

    evaluate_training_run(
        model=components.model,
        evaluator=components.evaluator,
        dataset_builder=components.dataset_builder,
        config=config,
        checkpoint_path=run.save_path,
        results_dir=run.results_dir,
        artifact_name="mossongplus-pretrained-source",
        validation_dataset=components.validation_dataset,
        test_dataset=components.test_dataset,
    )

    # ---------------------------------------------------------
    # 4. OPTIONAL ADABN TARGET CALIBRATION
    # ---------------------------------------------------------

    if not app_cfg.adabn.enabled:
        return

    print("\n>>> Preparing AdaBN target calibration...")

    # IMPORTANT:
    # This dataset must contain TARGET-DOMAIN calibration data,
    # processed with deterministic inference preprocessing.
    #
    # NO:
    #   MixUp
    #   random gain
    #   noise overlay
    #   random time shift
    #   random overlap
    #   stochastic augmentation
    #
    # It must also NOT contain the final target test samples.
    target_calibration_dataset = build_target_calibration_dataset(
        target_dir=app_cfg.adabn.target_dir,
        config=config,
    )

    # ---------------------------------------------------------
    # 5. CAPTURE SOURCE BN BANK
    #
    # AdaBN MUST be constructed after load_weights().
    # ---------------------------------------------------------

    adabn = AdaBN(
        components.model,
    )

    # ---------------------------------------------------------
    # 6. CALIBRATE TARGET BN BANK
    # ---------------------------------------------------------

    adabn.calibrate(
        target_calibration_dataset,
    )

    print(">>> AdaBN target calibration complete.")

    # The model has automatically returned to SOURCE stats here.
    assert adabn.active_domain == "source"
    assert adabn.target_ready

    # ---------------------------------------------------------
    # 7. OPTIONAL: SAVE TARGET-BN CHECKPOINT
    #
    # Never overwrite run.save_path.
    # ---------------------------------------------------------

    source_checkpoint = Path(run.save_path)

    target_checkpoint = source_checkpoint.with_name(
        f"{source_checkpoint.stem}.target_adabn"
        f"{source_checkpoint.suffix}"
    )

    with adabn.domain("target") as target_model:
        target_model.save_weights(
            str(target_checkpoint)
        )

    print(
        f">>> Saved target AdaBN checkpoint: "
        f"{target_checkpoint}"
    )

    # At this point source stats are active again.
    assert adabn.active_domain == "source"


def main(args=None):
    """Run pretraining, selecting pilot profile when needed."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--defaults_path",
        type=str,
    )

    parser.add_argument(
        "--model_cfg_path",
        type=str,
    )

    parsed_args, _ = parser.parse_known_args(args)

    if (
        parsed_args.defaults_path is None
        and parsed_args.model_cfg_path is None
    ):
        defaults_path, model_cfg_path, runtime_root = (
            prepare_default_pilot()
        )

        os.environ["WINGBEAT_RUNTIME_ROOT"] = str(
            runtime_root
        )

        os.chdir(runtime_root)

    else:
        defaults_path = (
            parsed_args.defaults_path
            or "configs/defaults.yaml"
        )

        model_cfg_path = (
            parsed_args.model_cfg_path
            or "configs/model.yaml"
        )

    train_supervised(
        defaults_path=defaults_path,
        model_cfg_path=model_cfg_path,
    )


if __name__ == "__main__":
    main()
