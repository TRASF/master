from __future__ import annotations

import math

import tensorflow as tf
import tensorflow.keras as keras


@keras.utils.register_keras_serializable(
    package="wingbeat_ml"
)
class FIRBandpassInitializer(
    keras.initializers.Initializer
):
    """
    Initialize the first Conv1D as a bank of
    band-pass FIR filters.
'
    Expected Conv1D kernel shape:

        [kernel_size, input_channels, filters]

    Currently intended for mono raw audio.
    """

    def __init__(
        self,
        sample_rate: float = 8000.0,
        min_freq: float = 300.0,
        max_freq: float = 3800.0,
        overlap: float = 0.10,
    ):
        self.sample_rate = float(sample_rate)
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.overlap = float(overlap)

    def __call__(
        self,
        shape,
        dtype=None,
    ):
        # Keras may provide "float32" as a string.
        # TensorFlow signal functions expect tf.DType.
        dtype = tf.as_dtype(
            dtype or tf.float32
        )

        kernel_size = int(shape[0])
        input_channels = int(shape[1])
        filters = int(shape[2])

        if input_channels != 1:
            raise ValueError(
                "FIRBandpassInitializer currently "
                "expects mono input."
            )

        nyquist = self.sample_rate / 2.0

        if not (
            0.0
            < self.min_freq
            < self.max_freq
            < nyquist
        ):
            raise ValueError(
                "Require: "
                "0 < min_freq < max_freq < Nyquist."
            )

        # ----------------------------------------------
        # Time axis centered around zero.
        # Example K=101:
        #
        # -50 ... -1, 0, 1 ... 50
        # ----------------------------------------------

        n = tf.cast(
            tf.range(kernel_size),
            dtype,
        )

        n -= tf.cast(
            kernel_size - 1,
            dtype,
        ) / 2.0

        n = n[:, None]

        # ----------------------------------------------
        # Frequency-band boundaries.
        # ----------------------------------------------

        edges = tf.linspace(
            tf.cast(self.min_freq, dtype),
            tf.cast(self.max_freq, dtype),
            filters + 1,
        )

        low = edges[:-1]
        high = edges[1:]

        bandwidth = high - low

        # Slight overlap between filters.
        low -= (
            bandwidth
            * self.overlap
            * 0.5
        )

        high += (
            bandwidth
            * self.overlap
            * 0.5
        )

        low = tf.maximum(
            low,
            tf.cast(
                self.min_freq,
                dtype,
            ),
        )

        high = tf.minimum(
            high,
            tf.cast(
                self.max_freq,
                dtype,
            ),
        )

        low = low[None, :]
        high = high[None, :]

        fs = tf.cast(
            self.sample_rate,
            dtype,
        )

        # ----------------------------------------------
        # Normalized sinc.
        # ----------------------------------------------

        def sinc(x):
            pi_x = math.pi * x

            return tf.where(
                tf.abs(x) < 1e-7,
                tf.ones_like(x),
                tf.sin(pi_x) / pi_x,
            )

        # ----------------------------------------------
        # Low-pass FIR.
        # ----------------------------------------------

        def lowpass(cutoff):
            normalized = (
                2.0 * cutoff / fs
            )

            return (
                normalized
                * sinc(
                    normalized * n
                )
            )

        # ----------------------------------------------
        # Band-pass:
        #
        # LP(high) - LP(low)
        # ----------------------------------------------

        kernels = (
            lowpass(high)
            - lowpass(low)
        )

        # ----------------------------------------------
        # Window the FIR kernels.
        # ----------------------------------------------

        window = tf.signal.hamming_window(
            kernel_size,
            periodic=False,
            dtype=dtype,
        )

        kernels *= window[:, None]

        # ----------------------------------------------
        # Remove residual DC response.
        # ----------------------------------------------

        kernels -= tf.reduce_mean(
            kernels,
            axis=0,
            keepdims=True,
        )

        # ----------------------------------------------
        # Normalize every filter to similar energy.
        # ----------------------------------------------

        energy = tf.sqrt(
            tf.reduce_sum(
                tf.square(kernels),
                axis=0,
                keepdims=True,
            )
            + tf.cast(1e-8, dtype)
        )

        kernels /= energy

        # Current shape:
        #
        # [kernel_size, filters]
        #
        # Conv1D needs:
        #
        # [kernel_size, input_channels, filters]
        #

        return kernels[:, None, :]

    def get_config(self):
        return {
            "sample_rate": self.sample_rate,
            "min_freq": self.min_freq,
            "max_freq": self.max_freq,
            "overlap": self.overlap,
        }
