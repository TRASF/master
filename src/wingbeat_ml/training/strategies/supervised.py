"""Supervised training strategy: GradientTape loop owned by strategy."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional
import tensorflow as tf

from wingbeat_ml.training.strategies.base import TrainingStrategy


class SupervisedStrategy(TrainingStrategy):
    """Supervised training strategy owning the supervised forward/gradient step."""

    required_datasets = {"train", "validation"}

    def __init__(
        self,
        model: Any = None,
        optimizer: Any = None,
        loss_fn: Any = None,
        config: Any = None,
        *,
        train_dataset: Any = None,
        class_weights: Optional[Any] = None,
        evaluate_fn: Optional[Callable[[], Dict[str, float]]] = None,
        profiler_logdir: Optional[Any] = None,
        _trainer: Optional[Any] = None,
    ):
        if _trainer is not None:
            self.model = _trainer.model
            self.optimizer = _trainer.optimizer
            self.loss_fn = _trainer.loss_fn
            self.class_weights = getattr(_trainer, "class_weights", None)
            self._trainer = _trainer
        else:
            self.model = model
            self.optimizer = optimizer
            self.loss_fn = loss_fn
            self.class_weights = class_weights
            self._trainer = None

        self.train_dataset = train_dataset
        self._evaluate_fn = evaluate_fn
        self.global_step = 0
        loss_name = getattr(self.loss_fn, "name", "") if self.loss_fn else ""
        if not isinstance(loss_name, str):
            loss_name = str(loss_name)
        self.is_contrastive = "contrastive" in loss_name.lower()
        self._compiled_step = None

    def _get_sample_weights(self, y: tf.Tensor) -> Optional[tf.Tensor]:
        if self.class_weights is None:
            return None
        weights = tf.cast(self.class_weights, y.dtype)
        return tf.reduce_sum(y * weights, axis=-1)

    def train_step(self, x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        sample_weight = self._get_sample_weights(y)

        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.loss_fn(y, predictions, sample_weight=sample_weight)

            if len(loss.shape) > 0:
                loss = tf.reduce_mean(loss)

            if hasattr(self.optimizer, "get_scaled_loss"):
                scaled_loss = self.optimizer.get_scaled_loss(loss)
            elif hasattr(self.optimizer, "scale_loss"):
                scaled_loss = self.optimizer.scale_loss(loss)
            else:
                scaled_loss = loss

        scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        if hasattr(self.optimizer, "get_unscaled_gradients"):
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        elif hasattr(self.optimizer, "unscale_gradients"):
            gradients = self.optimizer.unscale_gradients(scaled_gradients)
        else:
            gradients = scaled_gradients

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        correct = (
            tf.reduce_sum(
                tf.cast(
                    tf.equal(tf.argmax(y, axis=-1), tf.argmax(predictions, axis=-1)),
                    tf.float32,
                )
            )
            if not self.is_contrastive
            else tf.constant(0.0, dtype=tf.float32)
        )

        return loss, correct

    def train_epoch(self, datasets: Any = None, *, epoch: int = 0) -> Dict[str, Any]:
        if self._trainer is not None and hasattr(self._trainer, "train_epoch"):
            res = self._trainer.train_epoch()
            self.global_step = self._trainer.global_step
            return res

        ds = datasets or self.train_dataset
        if ds is None:
            return {"loss": 0.0, "accuracy": 0.0, "batches": 0, "examples": 0, "global_step": self.global_step}

        if self._compiled_step is None:
            self._compiled_step = tf.function(self.train_step, reduce_retracing=True)

        batches = 0
        examples = 0
        total_loss_sum = 0.0
        total_correct_sum = 0.0

        for x, y in ds:
            loss, correct = self._compiled_step(x, y)
            batch_size_i = int(tf.shape(x)[0])
            batches += 1
            examples += batch_size_i
            total_loss_sum += float(loss) * batch_size_i
            total_correct_sum += float(correct)

        self.global_step += batches
        avg_loss = (total_loss_sum / examples) if examples > 0 else 0.0
        avg_acc = (total_correct_sum / examples) if (examples > 0 and not self.is_contrastive) else 0.0

        return {
            "loss": avg_loss,
            "accuracy": avg_acc,
            "batches": batches,
            "examples": examples,
            "global_step": self.global_step,
        }

    def validate_epoch(self, dataset: Any = None, *, epoch: int = 0) -> Dict[str, float]:
        if self._evaluate_fn is not None:
            return self._evaluate_fn()
        return {}

    def checkpoint_objects(self) -> Dict[str, Any]:
        return {}


__all__ = ["SupervisedStrategy"]
