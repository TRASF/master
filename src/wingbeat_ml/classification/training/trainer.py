import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import tensorflow as tf

from wingbeat_ml.config.schema import TrainingResult
from wingbeat_ml.integrations.wandb import NoOpTracker


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_ds,
        class_weights=None,
        *,
        steps_per_call=20,
        jit_compile=False,
        profiler=None,
        profiler_logdir=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_ds = train_ds
        self.steps_per_call = int(steps_per_call)
        if self.steps_per_call <= 0:
            raise ValueError("steps_per_call must be greater than zero")
        self.global_step = 0
        self.profiler = profiler or {}
        self.profiler_logdir = profiler_logdir
        self._profiler_active = False
        self._profiler_finished = False

        self._compiled_train_step = tf.function(
            self.train_step,
            reduce_retracing=True,
            jit_compile=bool(jit_compile),
        )

        if class_weights is not None:
            if isinstance(class_weights, dict):
                class_weights = [class_weights[k] for k in sorted(class_weights.keys())]

            self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        else:
            self.class_weights = None

        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")

    def set_class_weights(self, class_weights):
        """
        Allows dynamic class-weight updates between epochs.
        """
        if class_weights is None:
            self.class_weights = None
            return

        if isinstance(class_weights, dict):
            class_weights = [class_weights[k] for k in sorted(class_weights.keys())]

        self.class_weights = tf.constant(class_weights, dtype=tf.float32)

    def _get_sample_weights(self, y):
        """
        y is expected to be one-hot, shape: (batch, num_classes).
        """
        if self.class_weights is None:
            return None

        weights = tf.cast(self.class_weights, y.dtype)
        return tf.reduce_sum(y * weights, axis=-1)

    def train_step(self, x, y):
        """Legacy compatibility wrapper for SupervisedStrategy step."""
        from wingbeat_ml.classification.training.strategies.supervised import SupervisedStrategy
        strat = SupervisedStrategy(
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            class_weights=self.class_weights,
        )
        return strat.train_step(x, y)

    def train_epoch(self):
        batches = 0
        examples = 0
        total_loss_sum = 0.0
        total_correct_sum = 0.0

        for x, y in self.train_ds:
            current_step = self.global_step + batches
            if self.profiler.get("enabled") and not self._profiler_finished:
                start_step = int(self.profiler.get("start_step", 10))
                end_step = start_step + int(self.profiler.get("num_steps", 10))
                if not self._profiler_active and current_step >= start_step:
                    if not self.profiler_logdir:
                        raise ValueError(
                            "profiler_logdir is required when profiler is enabled"
                        )
                    tf.profiler.experimental.start(str(self.profiler_logdir))
                    self._profiler_active = True

            loss, correct = self._compiled_train_step(x, y)
            batch_size_i = int(tf.shape(x)[0])
            batches += 1
            examples += batch_size_i
            total_loss_sum += float(loss) * batch_size_i
            total_correct_sum += float(correct)

            current_step = self.global_step + batches
            if self._profiler_active:
                start_step = int(self.profiler.get("start_step", 10))
                end_step = start_step + int(self.profiler.get("num_steps", 10))
                if current_step >= end_step:
                    tf.profiler.experimental.stop()
                    self._profiler_active = False
                    self._profiler_finished = True

        if self._profiler_active:
            tf.profiler.experimental.stop()
            self._profiler_active = False
            self._profiler_finished = True

        self.global_step += batches

        avg_loss = (total_loss_sum / examples) if examples > 0 else 0.0
        avg_acc = (total_correct_sum / examples) if examples > 0 else 0.0

        return {
            "loss": avg_loss,
            "accuracy": avg_acc,
            "batches": batches,
            "examples": examples,
            "global_step": self.global_step,
        }

    def fit(
        self,
        *,
        model=None,
        train_dataset=None,
        epochs=1,
        strategy=None,
        callbacks=None,
        tracker=None,
        evaluate_epoch=None,
        on_epoch_end=None,
        save_path=None,
        config=None,
    ) -> TrainingResult:
        model = model or self.model
        train_ds = train_dataset if train_dataset is not None else self.train_ds
        callbacks = callbacks or {}
        tracker = tracker or NoOpTracker()

        history: List[Dict[str, float]] = []
        console = getattr(config.logging, "console", "normal") if config and hasattr(config, "logging") else "normal"
        jsonl_logger = None
        if config and hasattr(config, "logging") and config.logging.jsonl and save_path:
            from wingbeat_ml.classification.pipelines.helpers.reporting import JsonlMetricLogger
            jsonl_logger = JsonlMetricLogger(Path(save_path).parent / "metrics.jsonl")

        def _get_lr():
            lr = getattr(self.optimizer, "learning_rate", None)
            if lr is not None:
                return float(tf.keras.backend.get_value(lr))
            inner = getattr(self.optimizer, "inner_optimizer", None)
            if inner is not None:
                return float(tf.keras.backend.get_value(inner.learning_rate))
            return 0.001

        best_metric = 0.0
        best_epoch = 0

        for epoch in range(epochs):
            started = time.perf_counter()
            if strategy is not None and hasattr(strategy, "train_epoch"):
                train_metrics = strategy.train_epoch(train_ds, epoch=epoch)
            else:
                train_metrics = self.train_epoch()
            train_duration = time.perf_counter() - started

            logs: Dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_metrics.get("loss", 0.0),
                "train_accuracy": train_metrics.get("accuracy", 0.0),
                "learning_rate": _get_lr(),
                "epoch_duration_seconds": train_duration,
                "train_duration_seconds": train_duration,
                "global_step": train_metrics.get("global_step", self.global_step),
                "steps_per_epoch": train_metrics.get("batches", 0),
                "steps_per_call": self.steps_per_call,
            }

            for k, v in train_metrics.items():
                logs.setdefault(f"train_{k}", v)

            val_metrics = None
            if evaluate_epoch is not None:
                val_started = time.perf_counter()
                val_metrics = evaluate_epoch()
                logs["validation_duration_seconds"] = time.perf_counter() - val_started
                for k, v in val_metrics.items():
                    name = k if k.startswith("val_") else f"val_{k}"
                    logs[name] = v
            elif strategy is not None and hasattr(strategy, "validate_epoch"):
                val_started = time.perf_counter()
                val_metrics = strategy.validate_epoch(train_ds, epoch=epoch)
                if val_metrics:
                    logs["validation_duration_seconds"] = time.perf_counter() - val_started
                    for k, v in val_metrics.items():
                        name = k if k.startswith("val_") else f"val_{k}"
                        logs[name] = v

            cb_started = time.perf_counter()
            ckpt_started = time.perf_counter()
            checkpoint_cb = callbacks.get("model_checkpoint")
            if checkpoint_cb is not None:
                saved = checkpoint_cb.save(model, logs)
                if saved and save_path:
                    monitor = getattr(checkpoint_cb, "monitor", None)
                    monitor_name = getattr(monitor, "monitor", "val_score")
                    monitor_val = float(logs.get(monitor_name, 0.0))
                    if monitor_val > best_metric:
                        best_metric = monitor_val
                        best_epoch = epoch
                    if console != "quiet":
                        print(f"  --> Saved best weights to {save_path} ({monitor_name}={monitor_val:.4f})")
            logs["checkpoint_duration_seconds"] = time.perf_counter() - ckpt_started

            reduce_lr = callbacks.get("reduce_lr_on_plateau")
            if reduce_lr is not None:
                reduce_lr.on_epoch_end(logs)

            cosine = callbacks.get("cosine_annealing")
            if cosine is not None:
                cosine.on_epoch_end(logs)

            log_started = time.perf_counter()
            tracker.on_epoch_end(epoch, logs)
            wandb_cb = callbacks.get("wandb_logger")
            if wandb_cb is not None:
                wandb_cb.on_epoch_end(logs)
            logs["tracking_duration_seconds"] = time.perf_counter() - log_started
            logs["callback_duration_seconds"] = time.perf_counter() - cb_started
            logs["epoch_total_duration_seconds"] = time.perf_counter() - started

            if jsonl_logger is not None:
                jsonl_logger.log(logs)
            history.append(dict(logs))

            if on_epoch_end is not None:
                on_epoch_end(epoch, logs)

            early_stopping = callbacks.get("early_stopping")
            if early_stopping is not None and early_stopping.check(logs, epoch=epoch):
                if console != "quiet":
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                break

        return TrainingResult(
            best_checkpoint=Path(save_path) if save_path else None,
            final_epoch=len(history) - 1 if history else 0,
            best_metric=best_metric,
            history={k: [float(h[k]) for h in history if k in h] for k in history[0].keys()} if history else {},
        )

# Compatibility name retained for older callers.
Train = Trainer

__all__ = ["Train", "Trainer"]
