import tensorflow as tf

class CategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.75, gamma=5.0, from_logits=True, label_smoothing=0.0, reduction=tf.keras.losses.Reduction.NONE, name='categorical_focal_loss'):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], y_true.dtype)
            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)

        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # Clip to avoid log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = self.alpha * tf.pow(1.0 - y_pred, self.gamma) * cross_entropy
        
        return tf.reduce_sum(loss, axis=-1)

class WeightedLossWrapper(tf.keras.losses.Loss):
    def __init__(self, base_loss, class_weights=None, name=None):
        super().__init__(name=name or base_loss.name)
        self.base_loss = base_loss
        self.class_weights = None
        if class_weights is not None:
            if isinstance(class_weights, dict):
                class_weights = [class_weights[k] for k in sorted(class_weights.keys())]
            self.class_weights = tf.constant(class_weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        # We assume base_loss returns per-example loss (shape: (batch_size,))
        per_example_loss = self.base_loss(y_true, y_pred)
        
        if self.class_weights is not None:
            sample_weights = tf.reduce_sum(y_true * self.class_weights, axis=-1)
            loss = tf.math.divide_no_nan(
                tf.reduce_sum(per_example_loss * sample_weights),
                tf.reduce_sum(sample_weights),
            )
        else:
            loss = tf.reduce_mean(per_example_loss)
            
        return loss

class LossFactory:
    @staticmethod
    def get_loss(config: dict = None):
        """
        Retrieves the loss function. Defaults to CategoricalCrossentropy if not specified.
        """
        config = config or {}
        loss_config = config.get('loss', {'name': 'CategoricalCrossentropy'})
        class_weights = config.get('class_weights')
        
        loss_name = loss_config.get('name')
        loss_params = {k: v for k, v in loss_config.items() if k != 'name'}
        
        # Initialize base loss
        if loss_name == 'CategoricalFocalLoss':
            base_loss = CategoricalFocalLoss(**loss_params)
        else:
            try:
                LossClass = getattr(tf.keras.losses, loss_name)
                base_loss = LossClass(**loss_params)
            except AttributeError:
                raise ValueError(f"Loss function '{loss_name}' not found in tf.keras.losses")
                
        # Wrap it with WeightedLossWrapper
        return WeightedLossWrapper(base_loss=base_loss, class_weights=class_weights)
