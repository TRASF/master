import tensorflow as tf


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
        self.is_contrastive = "contrastive" in getattr(loss_fn, "name", "").lower()
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

        scaled_gradients = tape.gradient(
            scaled_loss, self.model.trainable_variables
        )
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
        avg_acc = (total_correct_sum / examples) if (examples > 0 and not self.is_contrastive) else 0.0

        return {
            "loss": avg_loss,
            "accuracy": avg_acc,
            "batches": batches,
            "examples": examples,
            "global_step": self.global_step,
        }

# Compatibility name retained for older callers.
Train = Trainer

__all__ = ["Train", "Trainer"]
