import tensorflow as tf

class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.1, name='supervised_contrastive_loss', **kwargs):
        self.from_logits = kwargs.pop('from_logits', True)
        super().__init__(name=name, **kwargs)
        self.temperature = temperature

    def call(self, y_true, y_pred):
        # Convert one-hot to integer labels if necessary
        if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
            labels = tf.argmax(y_true, axis=-1)
        else:
            labels = tf.squeeze(y_true)

        labels = tf.cast(labels, tf.int32)

        # Normalize embeddings
        y_pred = tf.math.l2_normalize(y_pred, axis=1)

        # Compute dot-product (cosine similarity since normalized)
        logits = tf.matmul(y_pred, y_pred, transpose_b=True) / self.temperature

        batch_size = tf.shape(y_pred)[0]
        masks = tf.cast(tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1)), tf.float32)

        # Mask out self-contrast
        logits_max = tf.reduce_max(logits, axis=1, keepdims=True)
        logits = logits - tf.stop_gradient(logits_max) # Numerically stable

        exp_logits = tf.exp(logits) * (1 - tf.eye(batch_size))
        log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-12)

        # Mask for positive pairs (excluding self)
        mask_pos = masks - tf.eye(batch_size)

        num_positives = tf.reduce_sum(mask_pos, axis=1)
        # Avoid division by zero for classes with only 1 sample in batch
        num_positives = tf.maximum(num_positives, 1.0)

        mean_log_prob_pos = tf.reduce_sum(mask_pos * log_prob, axis=1) / num_positives

        return -mean_log_prob_pos

class LossFactory:
    @staticmethod
    def get_loss(config: dict = None):
        config = config or {}
        loss_cfg = config.get("loss", {}).copy()
        name = loss_cfg.pop("name", "CategoricalCrossentropy")

        aliases = {
            "CategoricalFocalLoss": "CategoricalFocalCrossentropy",
            "FocalLoss": "CategoricalFocalCrossentropy",
        }
        name = aliases.get(name, name)

        if name == "SupervisedContrastiveLoss":
            allowed_keys = {"temperature", "from_logits", "reduction", "name"}
            loss_params = {k: v for k, v in loss_cfg.items() if k in allowed_keys}
            return SupervisedContrastiveLoss(**loss_params)

        try:
            return tf.keras.losses.get({"class_name": name, "config": loss_cfg})
        except Exception as e:
            raise ValueError(f"Loss function '{name}' not found in tf.keras.losses: {e}")


__all__ = ["LossFactory", "SupervisedContrastiveLoss"]
