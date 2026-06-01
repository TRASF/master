import tensorflow as tf

class Train:
    def __init__(self, model, optimizer, loss_fn, train_ds):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_ds = train_ds

        # 1. Initialize Metrics
        self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.loss_fn(y, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

        self.train_loss_metric.update_state(loss)
        self.train_acc_metric.update_state(y, predictions)

        return loss

    
    def train_epoch(self):
        """
        Iterates through the training dataset and returns aggregated metrics.
        """
        # 3. Reset metrics at the start of every epoch
        self.train_loss_metric.reset_state()
        self.train_acc_metric.reset_state()

        for x, y in self.train_ds:
            self.train_step(x, y)

        # 4. Return the aggregated result
        return {
            "loss": float(self.train_loss_metric.result()),
            "accuracy": float(self.train_acc_metric.result())
        }
    
    