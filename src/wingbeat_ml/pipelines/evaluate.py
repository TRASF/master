"""High-level model evaluation pipeline."""

from wingbeat_ml.evaluation import ModelEvaluator


def evaluate_model(
    model,
    dataset,
    classes,
    *,
    loss_fn=None,
    save_dir=None,
    return_predictions=False,
):
    """Evaluate a model and return the canonical result mapping."""
    evaluator = ModelEvaluator(model, list(classes), loss_fn)
    return evaluator.evaluate_final_test(
        dataset,
        save_dir=save_dir,
        return_predictions=return_predictions,
    )


__all__ = ["evaluate_model"]
