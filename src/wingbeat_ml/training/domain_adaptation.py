from __future__ import annotations

import enum
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras


Domain = Literal["source", "target"]
ModeLike = Union["AdaBNMode", str]


class AdaBNMode(str, enum.Enum):
    """Supported AdaBN inference modes."""

    POST = "post"
    OTF = "otf"

    @classmethod
    def parse(cls, value: ModeLike) -> "AdaBNMode":
        if isinstance(value, cls):
            return value

        normalized = str(value).strip().lower()
        aliases = {
            "post": cls.POST,
            "offline": cls.POST,
            "adhoc": cls.POST,
            "ad-hoc": cls.POST,
            "otf": cls.OTF,
            "online": cls.OTF,
            "on_the_fly": cls.OTF,
            "on-the-fly": cls.OTF,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported AdaBN mode {value!r}. Expected one of: "
                "post/offline/adhoc or otf/online."
            ) from exc


@dataclass(frozen=True)
class BatchNormStats:
    """Inference statistics for one BatchNormalization layer."""

    mean: np.ndarray
    variance: np.ndarray


class _OnTheFlyBatchNormalization(keras.layers.BatchNormalization):
    """BatchNorm that always uses current-batch statistics without state updates.

    This layer exists only inside AdaBN's OTF inference clone. All non-BN layers
    remain ordinary inference-mode layers, so Dropout stays disabled while BN
    uses statistics from the current target mini-batch.
    """

    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
        mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        del training

        if mask is not None:
            raise NotImplementedError(
                "AdaBN OTF currently does not support masked BatchNormalization."
            )

        x = tf.cast(inputs, self.compute_dtype)
        shape = x.shape
        rank = getattr(shape, "rank", None)
        if rank is None:
            try:
                rank = len(shape)
            except TypeError:
                rank = None
        if rank is None:
            raise ValueError(
                "AdaBN OTF requires statically known activation rank."
            )
        if rank < 2:
            raise ValueError(
                f"AdaBN OTF BatchNormalization received rank-{rank} input; "
                "expected rank >= 2."
            )

        axis = int(self.axis)
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise ValueError(
                f"Invalid BatchNormalization axis={self.axis} for rank {rank}."
            )

        reduce_axes = [i for i in range(rank) if i != axis]
        mean, variance = tf.nn.moments(x, axes=reduce_axes, keepdims=False)

        # tf.nn.batch_normalization broadcasts naturally only when the feature
        # axis is last. Reshape parameters explicitly so axis=1/channels_first
        # works as well.
        broadcast_shape = [1] * rank
        broadcast_shape[axis] = tf.shape(x)[axis]

        mean = tf.reshape(tf.cast(mean, x.dtype), broadcast_shape)
        variance = tf.reshape(tf.cast(variance, x.dtype), broadcast_shape)
        offset = (
            tf.reshape(tf.cast(self.beta, x.dtype), broadcast_shape)
            if self.center
            else None
        )
        scale = (
            tf.reshape(tf.cast(self.gamma, x.dtype), broadcast_shape)
            if self.scale
            else None
        )

        return tf.nn.batch_normalization(
            x=x,
            mean=mean,
            variance=variance,
            offset=offset,
            scale=scale,
            variance_epsilon=self.epsilon,
        )


class AdaBN:
    """Focused Adaptive Batch Normalization controller for a trained Keras model.

    POST / offline AdaBN
    --------------------
    * Preserves the source-domain BN bank from the trained source checkpoint.
    * Optionally recomputes an exact source bank from deterministic source data.
    * Computes an exact target bank from deterministic target calibration data.
    * Switches source/target BN banks without changing trainable model weights.

    OTF / on-the-fly AdaBN
    ----------------------
    * Builds a lightweight inference clone of the same Functional/Sequential model.
    * Replaces only BatchNormalization layers with current-mini-batch normalization.
    * Does not update any moving statistics.
    * Leaves Dropout and all other stochastic training behavior disabled because the
      clone is still called/evaluated with training=False.

    Intended protocol for clean experiments
    ----------------------------------------
    * source_data: deterministic source TRAIN/calibration view, if source
      recalibration is desired.
    * target_data for POST: target VALIDATION/calibration data.
    * target TEST: evaluate after POST calibration; do not include it in POST
      calibration unless deliberately running a transductive experiment.
    * target TEST for OTF: each test mini-batch itself supplies its BN statistics;
      this is test-time adaptation and should be reported as such.

    Constraints
    -----------
    * Instantiate after loading the trained/best source weights.
    * Model must be built, single-input Functional/Sequential.
    * BN layers must be top-level layers; nested models containing BN are rejected.
    * Offline calibration data must be re-iterable.
    * Calibration data may yield x, (x, y), or (x, y, sample_weight).
    """

    def __init__(self, model: keras.Model) -> None:
        self.model = model
        self._validate_model()

        self._bn_layers = [
            layer
            for layer in self.model.layers
            if isinstance(layer, keras.layers.BatchNormalization)
        ]
        if not self._bn_layers:
            raise ValueError("AdaBN requires at least one BatchNormalization layer.")

        names = [layer.name for layer in self._bn_layers]
        if len(names) != len(set(names)):
            raise ValueError("BatchNormalization layer names must be unique.")

        self._probes: Dict[str, keras.Model] = {
            layer.name: keras.Model(
                inputs=self.model.inputs,
                outputs=layer.input,
                name=f"adabn_probe_{layer.name}",
            )
            for layer in self._bn_layers
        }

        # Source bank is the state of the already-loaded source checkpoint.
        self._source_bank = self._capture_bank()
        self._target_bank: Optional[Dict[str, BatchNormStats]] = None
        self._active_domain: Domain = "source"

        # Built lazily because many experiments only use POST AdaBN.
        self._otf_model: Optional[keras.Model] = None

    @property
    def target_ready(self) -> bool:
        return self._target_bank is not None

    @property
    def active_domain(self) -> Domain:
        return self._active_domain

    @property
    def bn_layer_names(self) -> Tuple[str, ...]:
        return tuple(layer.name for layer in self._bn_layers)

    def capture_source(self) -> "AdaBN":
        """Capture the model's current BN state as the source-domain bank.

        Call this only after the model has the intended source checkpoint active.
        Existing target calibration is invalidated because it was derived relative
        to the previous source model state.
        """
        self._source_bank = self._capture_bank()
        self._target_bank = None
        self._active_domain = "source"
        self._otf_model = None
        return self

    def calibrate(
        self,
        *,
        source_data: Optional[Any] = None,
        target_data: Any,
    ) -> "AdaBN":
        """Prepare source and target BN banks for POST AdaBN.

        Args:
            source_data:
                Optional deterministic source-domain calibration iterable. If
                supplied, the source BN bank is recomputed exactly from this data.
                If None, the BN bank captured from the loaded source checkpoint is
                preserved.
            target_data:
                Deterministic target-domain calibration iterable, normally target
                validation/calibration data rather than the final target test set.
        """
        if source_data is not None:
            self.calibrate_source(source_data)

        self.calibrate_target(target_data)
        return self

    def calibrate_source(self, source_data: Any) -> "AdaBN":
        """Recompute an exact source-domain BN bank from source calibration data."""
        self._validate_reiterable(source_data)

        old_source = self._copy_bank(self._source_bank)
        old_target = (
            None
            if self._target_bank is None
            else self._copy_bank(self._target_bank)
        )

        try:
            source_bank = self._calibrate_bank(
                data=source_data,
                initial_bank=old_source,
            )
        except Exception:
            self._assign_bank(old_source)
            self._source_bank = old_source
            self._target_bank = old_target
            self._active_domain = "source"
            raise

        self._source_bank = source_bank
        # A target bank calibrated against older network normalization is stale.
        self._target_bank = None
        self._assign_bank(self._source_bank)
        self._active_domain = "source"
        self._otf_model = None
        return self

    def calibrate_target(self, target_data: Any) -> "AdaBN":
        """Compute an exact target-domain BN bank for POST AdaBN.

        Layer-wise calibration is used. Earlier BN layers are switched to their
        newly calibrated target statistics before activations entering later BN
        layers are measured. Probes run with training=False, so Dropout is disabled
        and Keras moving-average updates are never used for calibration.

        The base model is always returned to SOURCE statistics afterward.
        """
        self._validate_reiterable(target_data)

        old_target = (
            None
            if self._target_bank is None
            else self._copy_bank(self._target_bank)
        )

        try:
            target_bank = self._calibrate_bank(
                data=target_data,
                initial_bank=self._source_bank,
            )
        except Exception:
            self._target_bank = old_target
            raise
        finally:
            self._assign_bank(self._source_bank)
            self._active_domain = "source"

        self._target_bank = target_bank
        return self

    def set_domain(self, domain: Domain) -> "AdaBN":
        """Activate source or POST-calibrated target BN inference statistics."""
        if domain == "source":
            self._assign_bank(self._source_bank)
        elif domain == "target":
            if self._target_bank is None:
                raise RuntimeError(
                    "Target BN statistics are not available. "
                    "Call calibrate_target() or calibrate() first."
                )
            self._assign_bank(self._target_bank)
        else:
            raise ValueError("domain must be either 'source' or 'target'.")

        self._active_domain = domain
        return self

    @contextmanager
    def domain(self, domain: Domain) -> Iterator[keras.Model]:
        """Temporarily activate a POST AdaBN statistics bank."""
        previous = self._active_domain
        self.set_domain(domain)
        try:
            yield self.model
        finally:
            self.set_domain(previous)

    @contextmanager
    def inference(
        self,
        mode: ModeLike,
        *,
        domain: Domain = "target",
    ) -> Iterator[keras.Model]:
        """Yield the model to use for POST or OTF inference.

        POST:
            Uses the requested persistent source/target bank on the original model.

        OTF:
            Uses a separate inference clone whose BN layers compute statistics from
            every current mini-batch. `domain` is ignored because no persistent BN
            bank is used by the OTF clone.
        """
        parsed = AdaBNMode.parse(mode)

        if parsed is AdaBNMode.POST:
            with self.domain(domain) as post_model:
                yield post_model
            return

        self._sync_otf_model()
        assert self._otf_model is not None
        yield self._otf_model

    def get_otf_model(self, *, sync: bool = True) -> keras.Model:
        """Return the OTF inference clone.

        The returned model should be used with `training=False`, `predict()`, or an
        evaluator that performs inference-mode calls. It is intentionally uncompiled.
        """
        if sync:
            self._sync_otf_model()
        elif self._otf_model is None:
            self._build_otf_model()

        assert self._otf_model is not None
        return self._otf_model

    def save_domain_weights(
        self,
        path: Union[str, Path],
        *,
        domain: Domain,
    ) -> None:
        """Save source or POST-target weights without changing the active domain."""
        with self.domain(domain) as domain_model:
            domain_model.save_weights(str(path))

    def _validate_model(self) -> None:
        inputs = getattr(self.model, "inputs", None)
        if not inputs:
            raise TypeError(
                "AdaBN requires a built Functional/Sequential Keras model with "
                "symbolic inputs."
            )
        if len(inputs) != 1:
            raise NotImplementedError(
                "This focused AdaBN module supports single-input models only."
            )

        for layer in self.model.layers:
            if isinstance(layer, keras.Model):
                nested_bn = [
                    child
                    for child in layer.layers
                    if isinstance(child, keras.layers.BatchNormalization)
                ]
                if nested_bn:
                    raise NotImplementedError(
                        "Nested models containing BatchNormalization are not "
                        "supported by this focused AdaBN implementation."
                    )

    def _validate_reiterable(self, data: Any) -> None:
        iterator = iter(data)
        if iterator is data and len(self._bn_layers) > 1:
            raise TypeError(
                "AdaBN offline calibration performs one pass per BN layer, so the "
                "dataset/Sequence must be re-iterable. One-shot iterators and "
                "generators are not supported."
            )

    def _capture_bank(self) -> Dict[str, BatchNormStats]:
        return {
            layer.name: BatchNormStats(
                mean=np.array(layer.moving_mean.numpy(), copy=True),
                variance=np.array(layer.moving_variance.numpy(), copy=True),
            )
            for layer in self._bn_layers
        }

    @staticmethod
    def _copy_bank(
        bank: Dict[str, BatchNormStats],
    ) -> Dict[str, BatchNormStats]:
        return {
            name: BatchNormStats(
                mean=np.array(stats.mean, copy=True),
                variance=np.array(stats.variance, copy=True),
            )
            for name, stats in bank.items()
        }

    def _assign_bank(self, bank: Dict[str, BatchNormStats]) -> None:
        for layer in self._bn_layers:
            try:
                stats = bank[layer.name]
            except KeyError as exc:
                raise KeyError(
                    f"BN statistics bank is missing layer '{layer.name}'."
                ) from exc

            layer.moving_mean.assign(
                tf.cast(stats.mean, layer.moving_mean.dtype)
            )
            layer.moving_variance.assign(
                tf.cast(stats.variance, layer.moving_variance.dtype)
            )

    def _calibrate_bank(
        self,
        *,
        data: Any,
        initial_bank: Dict[str, BatchNormStats],
    ) -> Dict[str, BatchNormStats]:
        self._assign_bank(initial_bank)
        calibrated: Dict[str, BatchNormStats] = {}

        for bn_layer in self._bn_layers:
            mean, variance = self._calibrate_layer(
                bn_layer=bn_layer,
                data=data,
            )

            stats = BatchNormStats(
                mean=np.array(mean, copy=True),
                variance=np.array(variance, copy=True),
            )
            calibrated[bn_layer.name] = stats

            # Later probes must see already-calibrated statistics from earlier BN
            # layers in the same domain.
            bn_layer.moving_mean.assign(
                tf.cast(stats.mean, bn_layer.moving_mean.dtype)
            )
            bn_layer.moving_variance.assign(
                tf.cast(stats.variance, bn_layer.moving_variance.dtype)
            )

        return calibrated

    def _calibrate_layer(
        self,
        *,
        bn_layer: keras.layers.BatchNormalization,
        data: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        probe = self._probes[bn_layer.name]

        total_count = 0.0
        total_sum: Optional[np.ndarray] = None
        total_sum_sq: Optional[np.ndarray] = None
        batches = 0

        for batch in data:
            x_batch = self._extract_x(batch)
            activations = probe(x_batch, training=False)

            if isinstance(activations, (tuple, list, dict)):
                raise TypeError(
                    f"BN layer '{bn_layer.name}' does not have a single tensor input."
                )

            activations = tf.convert_to_tensor(activations)
            shape = activations.shape
            rank = getattr(shape, "rank", None)
            if rank is None:
                try:
                    rank = len(shape)
                except TypeError:
                    rank = int(tf.rank(activations).numpy())
            if rank < 2:
                raise ValueError(
                    f"BN layer '{bn_layer.name}' received rank-{rank} activations; "
                    "expected rank >= 2."
                )

            axis = int(bn_layer.axis)
            if axis < 0:
                axis += rank
            if axis < 0 or axis >= rank:
                raise ValueError(
                    f"Invalid BatchNormalization axis={bn_layer.axis} for rank {rank}."
                )

            reduce_axes = [i for i in range(rank) if i != axis]

            # Float64 accumulation improves the accuracy of the one-off population
            # statistics. Results are assigned back in each BN variable's dtype.
            x64 = tf.cast(activations, tf.float64)
            count, batch_sum, batch_sum_sq, shift = tf.nn.sufficient_statistics(
                x64,
                axes=reduce_axes,
                shift=None,
                keepdims=False,
            )
            if shift is not None:
                raise RuntimeError("Unexpected shifted sufficient statistics.")

            count_np = float(count.numpy())
            sum_np = np.asarray(batch_sum.numpy(), dtype=np.float64)
            sum_sq_np = np.asarray(batch_sum_sq.numpy(), dtype=np.float64)

            if total_sum is None:
                total_sum = np.zeros_like(sum_np, dtype=np.float64)
                total_sum_sq = np.zeros_like(sum_sq_np, dtype=np.float64)

            total_count += count_np
            total_sum += sum_np
            total_sum_sq += sum_sq_np
            batches += 1

        if batches == 0 or total_sum is None or total_sum_sq is None:
            raise ValueError("AdaBN calibration dataset is empty.")
        if total_count <= 0.0:
            raise ValueError("AdaBN calibration dataset produced zero elements.")

        mean64 = total_sum / total_count
        variance64 = (total_sum_sq / total_count) - np.square(mean64)
        variance64 = np.maximum(variance64, 0.0)

        expected_shape = tuple(int(v) for v in bn_layer.moving_mean.shape)
        if mean64.shape != expected_shape:
            raise ValueError(
                f"Calibrated shape mismatch for BN '{bn_layer.name}': "
                f"got {mean64.shape}, expected {expected_shape}."
            )

        return mean64, variance64

    def _build_otf_model(self) -> None:
        def clone_layer(layer: keras.layers.Layer) -> keras.layers.Layer:
            config = layer.get_config()
            if isinstance(layer, keras.layers.BatchNormalization):
                return _OnTheFlyBatchNormalization.from_config(config)
            return layer.__class__.from_config(config)

        try:
            otf_model = keras.models.clone_model(
                self.model,
                clone_function=clone_layer,
            )
        except Exception as exc:
            raise TypeError(
                "Could not build the AdaBN OTF inference clone. The focused OTF "
                "path requires a serializable Functional/Sequential model whose "
                "top-level layers implement get_config()/from_config()."
            ) from exc

        try:
            otf_model.set_weights(self.model.get_weights())
        except Exception as exc:
            raise RuntimeError(
                "AdaBN OTF clone was built, but source model weights could not be "
                "copied into it."
            ) from exc

        self._otf_model = otf_model

    def _sync_otf_model(self) -> None:
        if self._otf_model is None:
            self._build_otf_model()
            return

        self._otf_model.set_weights(self.model.get_weights())

    @staticmethod
    def _extract_x(batch: Any) -> Any:
        if isinstance(batch, (tuple, list)):
            if not batch:
                raise ValueError("Calibration batch is empty.")
            return batch[0]
        return batch


def compute_coral_loss(
    source_features: tf.Tensor,
    target_features: tf.Tensor,
) -> tf.Tensor:
    """Deep CORAL covariance-alignment loss for rank-2 embeddings."""
    source_features = tf.cast(source_features, tf.float32)
    target_features = tf.cast(target_features, tf.float32)

    tf.debugging.assert_rank(source_features, 2)
    tf.debugging.assert_rank(target_features, 2)
    tf.debugging.assert_equal(
        tf.shape(source_features)[1],
        tf.shape(target_features)[1],
        message="Source and target CORAL feature dimensions must match.",
    )
    tf.debugging.assert_greater(
        tf.shape(source_features)[0],
        1,
        message="CORAL requires source batch size > 1.",
    )
    tf.debugging.assert_greater(
        tf.shape(target_features)[0],
        1,
        message="CORAL requires target batch size > 1.",
    )

    ns = tf.cast(tf.shape(source_features)[0], tf.float32)
    nt = tf.cast(tf.shape(target_features)[0], tf.float32)
    d = tf.cast(tf.shape(source_features)[1], tf.float32)

    source_centered = source_features - tf.reduce_mean(
        source_features,
        axis=0,
        keepdims=True,
    )
    target_centered = target_features - tf.reduce_mean(
        target_features,
        axis=0,
        keepdims=True,
    )

    cov_source = tf.matmul(
        source_centered,
        source_centered,
        transpose_a=True,
    ) / (ns - 1.0)
    cov_target = tf.matmul(
        target_centered,
        target_centered,
        transpose_a=True,
    ) / (nt - 1.0)

    return tf.reduce_sum(tf.square(cov_source - cov_target)) / (4.0 * d * d)


class DeepCORAL(keras.Model):
    """Deep CORAL wrapper for a single-input classifier.

    Expected fit() dataset elements:
        ((x_source, x_target), y_source)
    or:
        ((x_source, x_target), y_source, sample_weight)

    The selected feature layer must output [batch, feature_dim].

    BN policy:
      * Source forward pass uses training mode and keeps its BN moving updates.
      * Target forward pass also uses training-mode target-batch normalization so
        CORAL sees target-normalized features.
      * Target-induced persistent BN moving updates are immediately restored to
        the source-updated state.

    Therefore persistent BN statistics remain source-oriented during Deep CORAL.
    Build target inference statistics afterward with AdaBN.
    """

    def __init__(
        self,
        base_model: keras.Model,
        feature_layer_name: Optional[str] = None,
        coral_weight: float = 10.0,
        classification_metrics: Sequence[keras.metrics.Metric] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if coral_weight < 0.0:
            raise ValueError("coral_weight must be non-negative.")
        if not getattr(base_model, "inputs", None) or len(base_model.inputs) != 1:
            raise TypeError("DeepCORAL requires a built single-input Keras model.")

        if feature_layer_name is None:
            if len(base_model.layers) < 2:
                raise ValueError("Cannot infer a penultimate feature layer.")
            feature_layer_name = base_model.layers[-2].name

        feature_layer = base_model.get_layer(feature_layer_name)
        self.network = keras.Model(
            inputs=base_model.inputs,
            outputs=[feature_layer.output, base_model.output],
            name=f"{base_model.name}_deep_coral_network",
        )

        feature_shape = feature_layer.output.shape
        feature_rank = getattr(feature_shape, "rank", None)
        if feature_rank is None:
            try:
                feature_rank = len(feature_shape)
            except TypeError:
                feature_rank = None
        if feature_rank is not None and feature_rank != 2:
            raise ValueError(
                f"DeepCORAL feature layer '{feature_layer_name}' must output "
                f"rank-2 [batch, feature_dim] embeddings; got rank {feature_rank}."
            )

        self.coral_weight = float(coral_weight)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.cls_loss_tracker = keras.metrics.Mean(name="cls_loss")
        self.coral_loss_tracker = keras.metrics.Mean(name="coral_loss")
        self.classification_metrics = list(classification_metrics)

        self._bn_layers = [
            layer
            for layer in self.network.layers
            if isinstance(layer, keras.layers.BatchNormalization)
        ]

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.cls_loss_tracker,
            self.coral_loss_tracker,
            *self.classification_metrics,
        ]

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        _, logits = self.network(inputs, training=training)
        return logits

    def train_step(self, data: Any) -> Dict[str, tf.Tensor]:
        packed_inputs, y_source, sample_weight = (
            keras.utils.unpack_x_y_sample_weight(data)
        )

        if not isinstance(packed_inputs, (tuple, list)) or len(packed_inputs) != 2:
            raise ValueError("DeepCORAL expects inputs as (x_source, x_target).")
        if y_source is None:
            raise ValueError("DeepCORAL training requires source labels.")

        x_source, x_target = packed_inputs

        with tf.GradientTape() as tape:
            source_features, source_logits = self.network(
                x_source,
                training=True,
            )

            # Compute source classification loss immediately after the source call,
            # before the target forward can replace any call-scoped activity losses.
            cls_loss = self.compute_loss(
                x=x_source,
                y=y_source,
                y_pred=source_logits,
                sample_weight=sample_weight,
            )

            source_bn_state = self._snapshot_bn_state()
            try:
                target_features, _ = self.network(
                    x_target,
                    training=True,
                )
            finally:
                self._restore_bn_state(source_bn_state)

            coral_loss = compute_coral_loss(
                source_features,
                target_features,
            )
            total_loss = cls_loss + (
                tf.cast(self.coral_weight, coral_loss.dtype) * coral_loss
            )

        trainable_variables = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_variables)
        grads_and_vars = [
            (grad, var)
            for grad, var in zip(gradients, trainable_variables)
            if grad is not None
        ]
        if not grads_and_vars:
            raise RuntimeError("No gradients were produced by DeepCORAL train_step().")

        self.optimizer.apply_gradients(grads_and_vars)

        self.total_loss_tracker.update_state(total_loss)
        self.cls_loss_tracker.update_state(cls_loss)
        self.coral_loss_tracker.update_state(coral_loss)

        for metric in self.classification_metrics:
            metric.update_state(
                y_source,
                source_logits,
                sample_weight=sample_weight,
            )

        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data: Any) -> Dict[str, tf.Tensor]:
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        if y is None:
            raise ValueError("DeepCORAL evaluation requires labels.")

        logits = self(x, training=False)
        cls_loss = self.compute_loss(
            x=x,
            y=y,
            y_pred=logits,
            sample_weight=sample_weight,
        )

        self.total_loss_tracker.update_state(cls_loss)
        self.cls_loss_tracker.update_state(cls_loss)
        self.coral_loss_tracker.update_state(
            tf.zeros((), dtype=cls_loss.dtype)
        )

        for metric in self.classification_metrics:
            metric.update_state(y, logits, sample_weight=sample_weight)

        return {metric.name: metric.result() for metric in self.metrics}

    def _snapshot_bn_state(
        self,
    ) -> Tuple[Tuple[keras.layers.BatchNormalization, tf.Tensor, tf.Tensor], ...]:
        return tuple(
            (
                bn,
                tf.identity(bn.moving_mean),
                tf.identity(bn.moving_variance),
            )
            for bn in self._bn_layers
        )

    @staticmethod
    def _restore_bn_state(
        state: Tuple[
            Tuple[keras.layers.BatchNormalization, tf.Tensor, tf.Tensor],
            ...,
        ],
    ) -> None:
        for bn, moving_mean, moving_variance in state:
            bn.moving_mean.assign(moving_mean)
            bn.moving_variance.assign(moving_variance)
