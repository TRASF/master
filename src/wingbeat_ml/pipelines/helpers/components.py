"""Assembly of existing domain components for canonical pipelines."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SupervisedComponents:
    dataset_builder: object
    train_dataset: object
    validation_dataset: object
    test_dataset: object
    model: object
    loss_fn: object
    evaluator: object
    class_weights: object


def build_dataset_bundle(config, *, return_builder=False):
    """Build configured train, validation, and test datasets."""
    from wingbeat_ml.data.dataset import build_datasets

    dataset = config["dataset"]
    return build_datasets(
        dataset["train_dir"],
        config,
        val_dir=dataset["val_dir"],
        test_dir=dataset["test_dir"],
        return_builder=return_builder,
    )


def build_model_component(config, model_config, *, batch_size=None):
    """Build the configured model through the canonical registry."""
    from wingbeat_ml.registry import build_model

    arguments = {}
    if batch_size is not None:
        arguments["batch_size"] = batch_size
    return build_model(config, model_config, **arguments)


def _synchronize_loss_activation(config):
    activation = config["model"]["output_activation"]
    config["loss"]["from_logits"] = activation is None


def build_supervised_components(
    config,
    model_config,
    *,
    show_class_counts=False,
):
    """Build the common dataset, model, loss, and evaluation stack."""
    from wingbeat_ml.evaluation import ModelEvaluator
    from wingbeat_ml.pipelines.train import resolve_training_class_weights
    from wingbeat_ml.training import LossFactory

    print("Setting up datasets...")
    builder, train, validation, test = build_dataset_bundle(
        config,
        return_builder=True,
    )

    print("Building model...")
    model = build_model_component(config, model_config)
    model.summary()

    class_weights = resolve_training_class_weights(
        config,
        builder,
        show_counts=show_class_counts,
    )
    _synchronize_loss_activation(config)
    loss_fn = LossFactory.get_loss(config)
    evaluator = ModelEvaluator(model, config["classes"], loss_fn)

    return SupervisedComponents(
        dataset_builder=builder,
        train_dataset=train,
        validation_dataset=validation,
        test_dataset=test,
        model=model,
        loss_fn=loss_fn,
        evaluator=evaluator,
        class_weights=class_weights,
    )


__all__ = [
    "SupervisedComponents",
    "build_dataset_bundle",
    "build_model_component",
    "build_supervised_components",
]
