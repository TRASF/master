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
        self._compiled_train_steps = tf.function(
            self._train_steps,
            jit_compile=bool(jit_compile),
        )

        if class_weights is not None:
            if isinstance(class_weights, dict):
                class_weights = [class_weights[k] for k in sorted(class_weights.keys())]

            self.class_weights = tf.Variable(
                class_weights,
                dtype=tf.float32,
                trainable=False,
                name="class_weights",
            )
        else:
            self.class_weights = None

        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")

    def set_class_weights(self, class_weights):
        """
        Allows dynamic class-weight updates between epochs.
        """
        if isinstance(class_weights, dict):
            class_weights = [class_weights[k] for k in sorted(class_weights.keys())]

        class_weights = tf.constant(class_weights, dtype=tf.float32)

        if self.class_weights is None:
            self.class_weights = tf.Variable(
                class_weights,
                dtype=tf.float32,
                trainable=False,
                name="class_weights",
            )
        else:
            self.class_weights.assign(class_weights)

    def _get_sample_weights(self, y):
        """
        y is expected to be one-hot, shape: (batch, num_classes).
        """
        if self.class_weights is None:
            return None

        return tf.reduce_sum(y * self.class_weights, axis=-1)

    def train_step(self, x, y):
        sample_weight = self._get_sample_weights(y)

        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)

            loss = self.loss_fn(
                y,
                predictions,
                sample_weight=sample_weight,
            )

            if len(loss.shape) > 0:
                loss = tf.reduce_mean(loss)

            scaled_loss = loss
            if hasattr(self.optimizer, "scale_loss"):
                scaled_loss = self.optimizer.scale_loss(loss)
            elif hasattr(self.optimizer, "get_scaled_loss"):
                scaled_loss = self.optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        if hasattr(self.optimizer, "get_unscaled_gradients"):
            gradients = self.optimizer.get_unscaled_gradients(gradients)

        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

        self.train_loss_metric.update_state(loss)

        if not self.is_contrastive:
            self.train_acc_metric.update_state(y, predictions)

        return loss

    def _train_steps(self, iterator, num_steps):
        batches = tf.constant(0, dtype=tf.int32)
        examples = tf.constant(0, dtype=tf.int32)

        for _ in tf.range(num_steps):
            optional_element = iterator.get_next_as_optional()
            if not optional_element.has_value():
                break
            x, y = optional_element.get_value()
            self.train_step(x, y)
            batches += 1
            examples += tf.shape(x)[0]

        return batches, examples

    def train_epoch(self):
        self.train_loss_metric.reset_state()
        self.train_acc_metric.reset_state()

        batches = 0
        examples = 0
        iterator = iter(self.train_ds)
        while True:
            call_steps = self.steps_per_call
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
                if current_step < start_step:
                    call_steps = min(call_steps, start_step - current_step)
                elif self._profiler_active:
                    call_steps = min(call_steps, end_step - current_step)

            b, e = self._compiled_train_steps(
                iterator,
                tf.constant(call_steps, dtype=tf.int32),
            )
            if b == 0:
                break
            batches += int(b)
            examples += int(e)

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

        return {
            "loss": float(self.train_loss_metric.result()),
            "accuracy": float(self.train_acc_metric.result()),
            "batches": batches,
            "examples": examples,
            "global_step": self.global_step,
        }

# Compatibility name retained for older callers.
Train = Trainer

__all__ = ["Train", "Trainer"]
