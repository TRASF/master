"""Canonical pretraining pipeline."""

import argparse
import os

from wingbeat_ml.pipelines.helpers import (
    build_supervised_components,
    evaluate_training_run,
    load_pipeline_configuration,
    make_epoch_printer,
    prepare_default_pilot,
    prepare_training_run,
)
from wingbeat_ml.pipelines.train import run_training


def train_supervised(
    defaults_path="configs/defaults.yaml",
    model_cfg_path="configs/model.yaml",
    save_path=None,
    results_dir=None,
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

    epochs = config["train"]["epochs"]
    print(f"Output activation: {config['model']['output_activation']}")
    print(f"\nStarting training for {epochs} epochs...")
    run_training(
        components.model,
        components.train_dataset,
        config,
        evaluate_epoch=lambda: components.evaluator.evaluate_epoch(
            components.validation_dataset
        ),
        on_epoch_end=make_epoch_printer(config, detailed=True),
        class_weights=components.class_weights,
        save_path=run.save_path,
    )

    evaluate_training_run(
        model=components.model,
        evaluator=components.evaluator,
        dataset_builder=components.dataset_builder,
        config=config,
        checkpoint_path=run.save_path,
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
