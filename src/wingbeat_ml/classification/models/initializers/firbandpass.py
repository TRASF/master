from __future__ import annotations

import math
from typing import Sequence

import tensorflow as tf
import tensorflow.keras as keras


@keras.utils.register_keras_serializable(package="wingbeat_ml")
class FIRBandpassInitializer(keras.initializers.Initializer):
    """Initialize a mono Conv1D kernel as windowed band-pass FIR filters."""

    def __init__(
        self,
        sample_rate: float = 8000.0,
        min_freq: float = 300.0,
        max_freq: float = 3800.0,
        overlap: float = 0.10,
        enforce_odd_kernel: bool = True,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        if not 0.0 <= overlap < 1.0:
            raise ValueError("overlap must satisfy 0 <= overlap < 1.")

        nyquist = sample_rate / 2.0
        if not 0.0 < min_freq < max_freq < nyquist:
            raise ValueError(
                "Require 0 < min_freq < max_freq < sample_rate / 2."
            )

        self.sample_rate = float(sample_rate)
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.overlap = float(overlap)
        self.enforce_odd_kernel = bool(enforce_odd_kernel)

    def __call__(
        self,
        shape: Sequence[int],
        dtype=None,
    ) -> tf.Tensor:
        dtype = tf.as_dtype(dtype or tf.float32)

        if len(shape) != 3:
            raise ValueError(
                "Expected Conv1D kernel shape "
                "[kernel_size, input_channels, filters]."
            )

        kernel_size = int(shape[0])
        input_channels = int(shape[1])
        filters = int(shape[2])

        if input_channels != 1:
            raise ValueError(
                "FIRBandpassInitializer currently supports mono input only."
            )

        if filters < 1:
            raise ValueError("filters must be at least 1.")

        if self.enforce_odd_kernel and kernel_size % 2 == 0:
            raise ValueError(
                "Use an odd kernel_size for a centered Type-I FIR filter."
            )

        n = tf.cast(tf.range(kernel_size), dtype)
        n -= tf.cast(kernel_size - 1, dtype) / 2.0
        n = n[:, None]

        edges = tf.linspace(
            tf.cast(self.min_freq, dtype),
            tf.cast(self.max_freq, dtype),
            filters + 1,
        )

        base_low = edges[:-1]
        base_high = edges[1:]
        bandwidth = base_high - base_low

        expansion = 0.5 * self.overlap * bandwidth

        low = tf.maximum(
            base_low - expansion,
            tf.cast(self.min_freq, dtype),
        )
        high = tf.minimum(
            base_high + expansion,
            tf.cast(self.max_freq, dtype),
        )

        low = low[None, :]
        high = high[None, :]

        sample_rate = tf.cast(self.sample_rate, dtype)
        epsilon = tf.cast(1e-7, dtype)
        pi = tf.cast(math.pi, dtype)

        def normalized_sinc(x: tf.Tensor) -> tf.Tensor:
            pi_x = pi * x

            safe_denominator = tf.where(
                tf.abs(pi_x) < epsilon,
                tf.ones_like(pi_x),
                pi_x,
            )

            value = tf.sin(pi_x) / safe_denominator

            return tf.where(
                tf.abs(pi_x) < epsilon,
                tf.ones_like(value),
                value,
            )

        def lowpass(cutoff: tf.Tensor) -> tf.Tensor:
            normalized_cutoff = 2.0 * cutoff / sample_rate
            return normalized_cutoff * normalized_sinc(
                normalized_cutoff * n
            )

        kernels = lowpass(high) - lowpass(low)

        window = tf.signal.hamming_window(
            kernel_size,
            periodic=False,
            dtype=dtype,
        )
        kernels *= window[:, None]

        # Enforce an exact DC null after finite truncation/windowing.
        kernels -= tf.reduce_mean(kernels, axis=0, keepdims=True)

        energy = tf.sqrt(
            tf.reduce_sum(
                tf.square(kernels),
                axis=0,
                keepdims=True,
            )
            + tf.cast(1e-8, dtype)
        )
        kernels /= energy

        return kernels[:, None, :]

    def get_config(self) -> dict[str, float | bool]:
        return {
            "sample_rate": self.sample_rate,
            "min_freq": self.min_freq,
            "max_freq": self.max_freq,
            "overlap": self.overlap,
            "enforce_odd_kernel": self.enforce_odd_kernel,
        }
