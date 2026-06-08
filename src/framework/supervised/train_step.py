import tensorflow as tf


class Train:
    def __init__(self, model, optimizer, loss_fn, train_ds, class_weights=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_ds = train_ds

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

    @tf.function
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
        self.train_acc_metric.update_state(y, predictions)

        return loss

    def train_epoch(self):
        self.train_loss_metric.reset_state()
        self.train_acc_metric.reset_state()

        for x, y in self.train_ds:
            self.train_step(x, y)

        return {
            "loss": float(self.train_loss_metric.result()),
            "accuracy": float(self.train_acc_metric.result()),
        }
