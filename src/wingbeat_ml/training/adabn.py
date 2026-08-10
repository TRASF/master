from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Literal, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


Domain = Literal["source", "target"]


@dataclass(frozen=True)
class BatchNormStats:
    """Inference statistics for one BatchNormalization layer."""

    mean: np.ndarray
    variance: np.ndarray


class AdaBN:
    """Offline AdaBN for a built single-input Keras model.

    Instantiate this class only after the selected/best source checkpoint has
    been loaded into ``model``.

    Calibration accepts a re-iterable dataset/Sequence yielding any of:
      - x
      - (x, y)
      - (x, y, sample_weight)

    Labels and sample weights are ignored.
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
            raise ValueError(
                "AdaBN requires at least one BatchNormalization layer."
            )

        self._source_bank = self._capture_bank()
        self._target_bank: Optional[Dict[str, BatchNormStats]] = None
        self._active_domain: Domain = "source"

    @property
    def target_ready(self) -> bool:
        return self._target_bank is not None

    @property
    def active_domain(self) -> Domain:
        return self._active_domain

    def calibrate(self, target_calibration_data: Any) -> "AdaBN":
        """Calibrate exact target-domain population statistics.

        Calibration is performed layer-by-layer. Earlier BN layers are assigned
        their new target statistics before activations entering deeper BN layers
        are measured.

        The model is always restored to the source bank when calibration exits.
        """
        self._validate_reiterable(target_calibration_data)

        self._assign_bank(self._source_bank)
        self._active_domain = "source"

        target_bank: Dict[str, BatchNormStats] = {}

        try:
            for bn_layer in self._bn_layers:
                mean, variance = self._calibrate_layer(
                    bn_layer,
                    target_calibration_data,
                )

                bn_layer.moving_mean.assign(
                    tf.cast(mean, bn_layer.moving_mean.dtype)
                )
                bn_layer.moving_variance.assign(
                    tf.cast(variance, bn_layer.moving_variance.dtype)
                )

                target_bank[bn_layer.name] = BatchNormStats(
                    mean=np.array(mean, copy=True),
                    variance=np.array(variance, copy=True),
                )

            self._target_bank = target_bank
        finally:
            self._assign_bank(self._source_bank)
            self._active_domain = "source"

        return self

    def set_domain(self, domain: Domain) -> None:
        """Activate one stored BN statistics bank."""
        if domain == "source":
            bank = self._source_bank
        elif domain == "target":
            if self._target_bank is None:
                raise RuntimeError(
                    "Target BN statistics are unavailable. "
                    "Call calibrate() first."
                )
            bank = self._target_bank
        else:
            raise ValueError(
                "domain must be either 'source' or 'target'."
            )

        self._assign_bank(bank)
        self._active_domain = domain

    @contextmanager
    def domain(self, domain: Domain) -> Iterator[keras.Model]:
        """Temporarily activate a domain-specific BN bank."""
        previous = self._active_domain
        self.set_domain(domain)
        try:
            yield self.model
        finally:
            self.set_domain(previous)

    def _validate_model(self) -> None:
        inputs = getattr(self.model, "inputs", None)
        if not inputs:
            raise TypeError(
                "AdaBN requires a built Functional/Sequential model."
            )
        if len(inputs) != 1:
            raise NotImplementedError(
                "This AdaBN implementation supports single-input models only."
            )

        # Keep the module deliberately focused. Probing BN tensors hidden inside
        # nested Models requires architecture-specific graph handling.
        for layer in self.model.layers:
            if isinstance(layer, keras.Model):
                nested_bn = [
                    child
                    for child in layer.layers
                    if isinstance(
                        child,
                        keras.layers.BatchNormalization,
                    )
                ]
                if nested_bn:
                    raise NotImplementedError(
                        "Nested Models containing BatchNormalization are "
                        "not supported by this focused AdaBN module."
                    )

    def _validate_reiterable(self, data: Any) -> None:
        iterator = iter(data)
        if iterator is data and len(self._bn_layers) > 1:
            raise TypeError(
                "AdaBN calibration requires a re-iterable dataset/Sequence. "
                "A one-shot iterator/generator is not supported."
            )

    def _capture_bank(self) -> Dict[str, BatchNormStats]:
        return {
            layer.name: BatchNormStats(
                mean=np.array(
                    layer.moving_mean.numpy(),
                    copy=True,
                ),
                variance=np.array(
                    layer.moving_variance.numpy(),
                    copy=True,
                ),
            )
            for layer in self._bn_layers
        }

    def _assign_bank(
        self,
        bank: Dict[str, BatchNormStats],
    ) -> None:
        for layer in self._bn_layers:
            stats = bank[layer.name]
            layer.moving_mean.assign(
                tf.cast(
                    stats.mean,
                    layer.moving_mean.dtype,
                )
            )
            layer.moving_variance.assign(
                tf.cast(
                    stats.variance,
                    layer.moving_variance.dtype,
                )
            )

    def _calibrate_layer(
        self,
        bn_layer: keras.layers.BatchNormalization,
        data: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Measure the distribution entering one BN layer."""
        probe = keras.Model(
            inputs=self.model.inputs,
            outputs=bn_layer.input,
            name=f"adabn_probe_{bn_layer.name}",
        )

        total_count = 0.0
        total_sum: Optional[np.ndarray] = None
        total_sum_sq: Optional[np.ndarray] = None
        batch_count = 0

        for batch in data:
            x_batch = self._extract_x(batch)

            # Inference mode:
            # - Dropout is disabled.
            # - earlier BN layers use the currently assigned bank.
            # - no Keras BN EMA state is updated.
            activations = probe(
                x_batch,
                training=False,
            )

            if isinstance(
                activations,
                (tuple, list, dict),
            ):
                raise TypeError(
                    f"BN layer '{bn_layer.name}' must have one tensor input."
                )

            activations = tf.convert_to_tensor(activations)
            rank = activations.shape.rank
            if rank is None:
                rank = int(tf.rank(activations).numpy())

            if rank < 2:
                raise ValueError(
                    f"BN layer '{bn_layer.name}' received rank-{rank} "
                    "activations; expected rank >= 2."
                )

            axis = int(bn_layer.axis)
            if axis < 0:
                axis += rank
            if axis < 0 or axis >= rank:
                raise ValueError(
                    f"Invalid BN axis={bn_layer.axis} for rank {rank}."
                )

            reduce_axes = [
                i
                for i in range(rank)
                if i != axis
            ]

            # Use float64 only for this one-off accumulation.
            x64 = tf.cast(
                activations,
                tf.float64,
            )
            count, batch_sum, batch_sum_sq, shift = (
                tf.nn.sufficient_statistics(
                    x64,
                    axes=reduce_axes,
                    shift=None,
                    keepdims=False,
                )
            )
            if shift is not None:
                raise RuntimeError(
                    "Unexpected shifted sufficient statistics."
                )

            count_np = float(count.numpy())
            sum_np = np.asarray(
                batch_sum.numpy(),
                dtype=np.float64,
            )
            sum_sq_np = np.asarray(
                batch_sum_sq.numpy(),
                dtype=np.float64,
            )

            if total_sum is None:
                total_sum = np.zeros_like(
                    sum_np,
                    dtype=np.float64,
                )
                total_sum_sq = np.zeros_like(
                    sum_sq_np,
                    dtype=np.float64,
                )

            total_count += count_np
            total_sum += sum_np
            total_sum_sq += sum_sq_np
            batch_count += 1

        if (
            batch_count == 0
            or total_sum is None
            or total_sum_sq is None
        ):
            raise ValueError(
                "Target calibration dataset is empty."
            )

        if total_count <= 0.0:
            raise ValueError(
                "Target calibration produced zero elements."
            )

        mean = total_sum / total_count
        variance = (
            total_sum_sq / total_count
            - np.square(mean)
        )
        variance = np.maximum(
            variance,
            0.0,
        )

        expected_shape = tuple(
            int(v)
            for v in bn_layer.moving_mean.shape
        )
        if mean.shape != expected_shape:
            raise ValueError(
                f"Calibrated shape mismatch for BN "
                f"'{bn_layer.name}': got {mean.shape}, "
                f"expected {expected_shape}."
            )

        return mean, variance

    @staticmethod
    def _extract_x(batch: Any) -> Any:
        if isinstance(batch, (tuple, list)):
            if not batch:
                raise ValueError(
                    "Calibration batch is empty."
                )
            return batch[0]
        return batch
