import tensorflow as tf


class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_ds, class_weights=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_ds = train_ds
        self.is_contrastive = "contrastive" in getattr(loss_fn, "name", "").lower()

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

        gradients = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

        self.train_loss_metric.update_state(loss)

        if not self.is_contrastive:
            self.train_acc_metric.update_state(y, predictions)

        return loss

    @tf.function
    def _train_steps_tf(self, iterator, num_steps):
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
        steps_per_call = tf.constant(20, dtype=tf.int32)

        while True:
            b, e = self._train_steps_tf(iterator, steps_per_call)
            if b == 0:
                break
            batches += int(b)
            examples += int(e)

        return {
            "loss": float(self.train_loss_metric.result()),
            "accuracy": float(self.train_acc_metric.result()),
            "batches": batches,
            "examples": examples,
        }

# Compatibility name retained for older callers.
Train = Trainer

__all__ = ["Train", "Trainer"]
