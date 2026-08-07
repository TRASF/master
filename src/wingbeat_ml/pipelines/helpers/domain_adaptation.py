import enum
from typing import Any, Dict, Optional, Union
import tensorflow as tf
import tensorflow.keras as keras

from wingbeat_ml.config.schema import AppConfig

class Mode(enum.Enum):
    OTF = "otf"      # On-the-Fly (calculates stats live per batch)
    ADHOC = "adhoc"  # Offline sweep before deployment

class AdaBN(keras.Model):
    """
    Pluggable AdaBN Wrapper for Keras Models.

    Wraps an existing model and manages all target domain adaptation logic.
    """
    def __init__(self, base_model: keras.Model, mode: Mode = Mode.ADHOC, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.mode = mode

        # Find all Dropout layers across all nested submodules recursively
        self._dropout_layers = [
            layer for layer in self.base_model.layers
            if isinstance(layer, keras.layers.Dropout)
        ]
        self._original_dropout_rates = {l: l.rate for l in self._dropout_layers}

    def adapt(self, target_loader: keras.utils.Sequence):
        """
        Runs the offline target domain sweep (ADHOC mode).
        Recalculates BN running statistics without modifying model weights.
        """
        print("Executing ADHOC AdaBN sweep over target dataset...")

        # 1. Zero out dropout rates for clean statistics
        for layer in self._dropout_layers:
            layer.rate = 0.0

        # 2. Forward pass over target loader (no gradients recorded)
        for batch in target_loader:
            x_batch = batch[0] if isinstance(batch, (tuple, list)) else batch
            self.base_model(x_batch, training=True)

        # 3. Restore original dropout rates
        for layer, rate in self._original_dropout_rates.items():
            layer.rate = rate

        print("AdaBN sweep complete! Model is adapted and ready for evaluation.")

    def call(self, inputs, training=None):
        if self.mode == Mode.OTF:
            # OTF Mode: Forces BN layers to normalize per mini-batch live at test time
            # while temporarily disabling Dropout so no neurons are dropped.
            for layer in self._dropout_layers:
                layer.rate = 0.0

            outputs = self.base_model(inputs, training=True)

            for layer, rate in self._original_dropout_rates.items():
                layer.rate = rate

            return outputs
        else:
            # ADHOC Mode / Standard Inference: Uses the adapted frozen target stats
            return self.base_model(inputs, training=training)

def compute_coral_loss(source_features: tf.Tensor, target_features: tf.Tensor) -> tf.Tensor:
    """
    Computes the Correlation Alignment (CORAL) loss between source and target feature embeddings.

    source_features: (batch_size, feature_dim)
    target_features: (batch_size, feature_dim)
    """
    d = tf.cast(tf.shape(source_features)[1], tf.float32)
    ns = tf.cast(tf.shape(source_features)[0], tf.float32)
    nt = tf.cast(tf.shape(target_features)[0], tf.float32)

    # 1. Center features (subtract mean across batch)
    source_mean = tf.reduce_mean(source_features, axis=0, keepdims=True)
    target_mean = tf.reduce_mean(target_features, axis=0, keepdims=True)

    source_centered = source_features - source_mean
    target_centered = target_features - target_mean

    # 2. Compute Covariance Matrices: (feature_dim, feature_dim)
    cov_source = tf.matmul(source_centered, source_centered, transpose_a=True) / (ns - 1.0)
    cov_target = tf.matmul(target_centered, target_centered, transpose_a=True) / (nt - 1.0)

    # 3. Frobenius norm squared of the covariance difference
    loss = tf.reduce_sum(tf.square(cov_source - cov_target))
    loss = loss / (4.0 * d * d)

    return loss

class DeepCORAL(keras.Model):
    class SingleModelDeepCORAL(keras.Model):
        """
        Deep CORAL Wrapper for standard single-instance classifier models.
        Automatically extracts features from the penultimate layer for CORAL loss.
        """
        def __init__(
            self,
            base_model: keras.Model,
            feature_layer_name: str = None,
            coral_weight: float = 10.0,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.base_model = base_model
            self.coral_weight = coral_weight

            # 1. If no layer name is given, default to the layer right before the output layer
            if feature_layer_name is None:
                # base_model.layers[-2] is usually GlobalAveragePooling2D or the last Dense layer
                feature_layer_name = base_model.layers[-2].name
                print(f"Auto-selected feature layer for CORAL: '{feature_layer_name}'")

            # 2. Build a multi-output model that returns both [features, final_logits]
            self.feature_extractor = keras.Model(
                inputs=base_model.input,
                outputs=[
                    base_model.get_layer(feature_layer_name).output,
                    base_model.output
                ]
            )

            # Trackers for loss monitoring
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.cls_loss_tracker = keras.metrics.Mean(name="cls_loss")
            self.coral_loss_tracker = keras.metrics.Mean(name="coral_loss")

        @property
        def metrics(self):
            return [self.total_loss_tracker, self.cls_loss_tracker, self.coral_loss_tracker]

        def call(self, inputs, training=False):
            # Standard forward pass during evaluation
            return self.base_model(inputs, training=training)

        def train_step(self, data):
            # Expects zipped dataset: ((x_source, x_target), y_source)
            (x_source, x_target), y_source = data

            with tf.GradientTape() as tape:
                # 1. Forward pass on Source domain -> get (features, logits)
                source_features, source_logits = self.feature_extractor(x_source, training=True)

                # 2. Forward pass on Target domain -> get (features, _)
                target_features, _ = self.feature_extractor(x_target, training=True)

                # 3. Compute Classification Loss + CORAL Covariance Loss
                cls_loss = self.compiled_loss(y_source, source_logits, regularization_losses=self.losses)
                c_loss = compute_coral_loss(source_features, target_features)

                total_loss = cls_loss + (self.coral_weight * c_loss)

            # 4. Apply gradients to the base model
            gradients = tape.gradient(total_loss, self.base_model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))

            # 5. Update metrics
            self.total_loss_tracker.update_state(total_loss)
            self.cls_loss_tracker.update_state(cls_loss)
            self.coral_loss_tracker.update_state(c_loss)

            return {m.name: m.result() for m in self.metrics}


def maybe_apply_adabn(
    model: keras.Model,
    default_target_dataset: Any,
    config: Union[AppConfig, Dict[str, Any]],
    save_path: Optional[str] = None,
) -> None:
    """Run AdaBN adaptation sweep on target domain if enabled in config."""
    from wingbeat_ml.config import validate_config
    config = validate_config(config)
    if not config.adabn.enabled:
        return

    from pathlib import Path

    console = config.logging.console
    if console != "quiet":
        print("\n>>> Running AdaBN sweep on target domain dataset...")

    target_dataset = default_target_dataset
    target_dir = config.adabn.target_dir
    if target_dir:
        from wingbeat_ml.data.dataset import build_datasets

        _, _, target_dataset = build_datasets(
            target_dir,
            config,
        )

    if save_path and Path(save_path).exists():
        model.load_weights(save_path)

    mode_str = config.adabn.mode.lower()
    mode = Mode.OTF if mode_str == "otf" else Mode.ADHOC
    adabn_wrapper = AdaBN(model, mode=mode)
    adabn_wrapper.adapt(target_dataset)

    if save_path:
        model.save_weights(save_path)
        if console != "quiet":
            print(
                f" --> Re-saved best weights with target AdaBN stats to {save_path}"
            )
