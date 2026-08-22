from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


@keras.utils.register_keras_serializable(
    package="wingbeat_ml"
)
class SincConv1D(keras.layers.Layer):
    """
    Numerically stable SincNet-style Conv1D frontend.

    Intended for:
        raw mono waveform
        shape = (batch, samples, 1)

    Learns:
        - lower cutoff frequency
        - bandwidth

    rather than every FIR coefficient independently.

    Important:
        kernel_size must be odd.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        sample_rate: int = 8000,
        strides: int = 1,
        padding: str = "valid",
        dilation_rate: int = 1,
        min_low_hz: float = 20.0,
        min_band_hz: float = 20.0,
        frequency_margin_hz: float = 1.0,
        use_bias: bool = False,
        debug_checks: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if filters <= 0:
            raise ValueError(
                "filters must be > 0."
            )

        if kernel_size <= 1:
            raise ValueError(
                "kernel_size must be > 1."
            )

        # SincNet uses an odd kernel for a true center sample
        # and perfectly symmetric FIR filters.
        if kernel_size % 2 == 0:
            raise ValueError(
                "SincConv1D requires an odd kernel_size. "
                f"Received {kernel_size}. "
                "Use 99 instead of 100."
            )

        if sample_rate <= 0:
            raise ValueError(
                "sample_rate must be > 0."
            )

        if strides <= 0:
            raise ValueError(
                "strides must be > 0."
            )

        if dilation_rate <= 0:
            raise ValueError(
                "dilation_rate must be > 0."
            )

        if (
            strides > 1
            and dilation_rate > 1
        ):
            raise ValueError(
                "strides > 1 cannot be combined "
                "with dilation_rate > 1."
            )

        padding = str(
            padding
        ).lower()

        if padding not in {
            "valid",
            "same",
            "causal",
        }:
            raise ValueError(
                "padding must be one of: "
                "'valid', 'same', 'causal'."
            )

        nyquist = (
            float(sample_rate) / 2.0
        )

        if min_low_hz < 0:
            raise ValueError(
                "min_low_hz must be >= 0."
            )

        if min_band_hz <= 0:
            raise ValueError(
                "min_band_hz must be > 0."
            )

        if (
            min_low_hz
            + min_band_hz
            + frequency_margin_hz
            >= nyquist
        ):
            raise ValueError(
                "Invalid cutoff constraints for "
                "the selected sample_rate."
            )

        self.filters = int(
            filters
        )

        self.kernel_size = int(
            kernel_size
        )

        self.sample_rate = int(
            sample_rate
        )

        self.strides = int(
            strides
        )

        self.padding = padding

        self.dilation_rate = int(
            dilation_rate
        )

        self.min_low_hz = float(
            min_low_hz
        )

        self.min_band_hz = float(
            min_band_hz
        )

        self.frequency_margin_hz = float(
            frequency_margin_hz
        )

        self.use_bias = bool(
            use_bias
        )

        self.debug_checks = bool(
            debug_checks
        )

        # Explicitly constrain input to:
        # (batch, time, 1)
        self.input_spec = (
            keras.layers.InputSpec(
                ndim=3,
                axes={
                    -1: 1,
                },
            )
        )

        self.low_hz_ = None
        self.band_hz_ = None
        self.bias_weight = None

    # ==============================================================
    # Frequency conversion
    # ==============================================================

    @staticmethod
    def _hz_to_mel(
        hz: np.ndarray | float,
    ):
        return (
            2595.0
            * np.log10(
                1.0
                + np.asarray(hz)
                / 700.0
            )
        )

    @staticmethod
    def _mel_to_hz(
        mel: np.ndarray | float,
    ):
        return (
            700.0
            * (
                np.power(
                    10.0,
                    np.asarray(mel)
                    / 2595.0,
                )
                - 1.0
            )
        )

    # ==============================================================
    # Build
    # ==============================================================

    def build(
        self,
        input_shape,
    ):
        input_channels = int(
            input_shape[-1]
        )

        if input_channels != 1:
            raise ValueError(
                "SincConv1D only supports mono "
                "raw-waveform input. "
                f"Received {input_channels} channels."
            )

        nyquist = (
            self.sample_rate
            / 2.0
        )

        highest_hz = (
            nyquist
            - self.frequency_margin_hz
        )

        # ----------------------------------------------------------
        # Mel-spaced initial filters.
        # ----------------------------------------------------------

        mel_low = self._hz_to_mel(
            self.min_low_hz
        )

        mel_high = self._hz_to_mel(
            highest_hz
        )

        mel_points = np.linspace(
            mel_low,
            mel_high,
            self.filters + 1,
            dtype=np.float64,
        )

        hz_points = (
            self._mel_to_hz(
                mel_points
            )
            .astype(
                np.float32
            )
        )

        initial_low = (
            hz_points[:-1]
        )

        initial_band = np.diff(
            hz_points
        )

        # Our actual forward equation is:
        #
        # low =
        #     min_low_hz + abs(low_hz_)
        #
        # band =
        #     min_band_hz + abs(band_hz_)
        #
        # Therefore initialize the underlying variables
        # after subtracting those minimums.

        raw_low_init = np.maximum(
            initial_low
            - self.min_low_hz,
            1e-3,
        ).astype(
            np.float32
        )

        raw_band_init = np.maximum(
            initial_band
            - self.min_band_hz,
            1e-3,
        ).astype(
            np.float32
        )

        # Keep cutoff parameters in float32 even if the rest
        # of the model later uses mixed precision.
        self.low_hz_ = self.add_weight(
            name="low_hz",
            shape=(
                self.filters,
            ),
            initializer=(
                keras.initializers.Constant(
                    raw_low_init
                )
            ),
            dtype="float32",
            trainable=True,
        )

        self.band_hz_ = self.add_weight(
            name="band_hz",
            shape=(
                self.filters,
            ),
            initializer=(
                keras.initializers.Constant(
                    raw_band_init
                )
            ),
            dtype="float32",
            trainable=True,
        )

        if self.use_bias:
            self.bias_weight = (
                self.add_weight(
                    name="bias",
                    shape=(
                        self.filters,
                    ),
                    initializer="zeros",
                    dtype="float32",
                    trainable=True,
                )
            )

        super().build(
            input_shape
        )

    # ==============================================================
    # Stable sinc
    # ==============================================================

    @staticmethod
    def _safe_sinc(
        x: tf.Tensor,
    ) -> tf.Tensor:
        """
        Numerically stable:

            sinc(x) = sin(pi*x) / (pi*x)

        Critical point:

        Do NOT write only:

            tf.where(
                x == 0,
                1,
                sin(pi*x)/(pi*x)
            )

        because the unused division-by-zero branch can still
        generate NaN gradients.

        We first make the denominator safe, then select the
        analytical value at zero.
        """

        dtype = x.dtype

        pi = tf.cast(
            np.pi,
            dtype,
        )

        pi_x = (
            pi * x
        )

        epsilon = tf.cast(
            1e-7,
            dtype,
        )

        near_zero = (
            tf.abs(
                pi_x
            )
            < epsilon
        )

        # IMPORTANT:
        # Make the dangerous branch finite BEFORE
        # the final tf.where.
        safe_denominator = (
            tf.where(
                near_zero,
                tf.ones_like(
                    pi_x
                ),
                pi_x,
            )
        )

        ratio = (
            tf.sin(
                pi_x
            )
            / safe_denominator
        )

        return tf.where(
            near_zero,
            tf.ones_like(
                ratio
            ),
            ratio,
        )

    # ==============================================================
    # Cutoff frequencies
    # ==============================================================

    def _cutoff_frequencies(
        self,
        dtype=tf.float32,
    ):
        dtype = tf.dtypes.as_dtype(
            dtype
        )

        nyquist = tf.cast(
            self.sample_rate / 2.0,
            dtype,
        )

        margin = tf.cast(
            self.frequency_margin_hz,
            dtype,
        )

        min_low = tf.cast(
            self.min_low_hz,
            dtype,
        )

        min_band = tf.cast(
            self.min_band_hz,
            dtype,
        )

        low_param = tf.cast(
            self.low_hz_,
            dtype,
        )

        band_param = tf.cast(
            self.band_hz_,
            dtype,
        )

        # ----------------------------------------------------------
        # Positive low cutoff.
        # ----------------------------------------------------------

        low = (
            min_low
            + tf.abs(
                low_param
            )
        )

        max_low = (
            nyquist
            - margin
            - min_band
        )

        low = tf.clip_by_value(
            low,
            clip_value_min=(
                min_low
            ),
            clip_value_max=(
                max_low
            ),
        )

        # ----------------------------------------------------------
        # Positive bandwidth.
        # ----------------------------------------------------------

        high = (
            low
            + min_band
            + tf.abs(
                band_param
            )
        )

        high = tf.minimum(
            high,
            nyquist
            - margin,
        )

        # Because low was bounded above, this should
        # always remain at least min_band.
        high = tf.maximum(
            high,
            low + min_band,
        )

        return (
            low,
            high,
        )

    def get_cutoff_frequencies(
        self,
    ):
        """
        Return actual effective low/high cutoff frequencies.
        """

        low, high = (
            self._cutoff_frequencies(
                tf.float32
            )
        )

        return (
            low,
            high,
        )

    # ==============================================================
    # Construct FIR kernels
    # ==============================================================

    def _make_filters(
        self,
    ):
        # Sinc/filter construction is intentionally kept
        # in float32 for numerical stability.
        dtype = tf.float32

        low_hz, high_hz = (
            self._cutoff_frequencies(
                dtype
            )
        )

        sample_rate = tf.cast(
            self.sample_rate,
            dtype,
        )

        # Normalize to cycles / sample.
        low = (
            low_hz
            / sample_rate
        )

        high = (
            high_hz
            / sample_rate
        )

        # ----------------------------------------------------------
        # Perfectly symmetric odd-length FIR.
        #
        # K = 99:
        #
        # -49 ... -1, 0, 1 ... 49
        # ----------------------------------------------------------

        half = (
            self.kernel_size
            // 2
        )

        n = tf.range(
            -half,
            half + 1,
            dtype=dtype,
        )

        n = n[
            tf.newaxis,
            :
        ]

        low = low[
            :,
            tf.newaxis,
        ]

        high = high[
            :,
            tf.newaxis,
        ]

        # ----------------------------------------------------------
        # Eq:
        #
        # BandPass =
        #
        # 2*f_high*sinc(2*f_high*n)
        # -
        # 2*f_low*sinc(2*f_low*n)
        # ----------------------------------------------------------

        two = tf.cast(
            2.0,
            dtype,
        )

        low_pass_high = (
            two
            * high
            * self._safe_sinc(
                two
                * high
                * n
            )
        )

        low_pass_low = (
            two
            * low
            * self._safe_sinc(
                two
                * low
                * n
            )
        )

        band_pass = (
            low_pass_high
            - low_pass_low
        )

        # ----------------------------------------------------------
        # Symmetric window for FIR filter design.
        # ----------------------------------------------------------

        window = (
            tf.signal.hamming_window(
                self.kernel_size,
                periodic=False,
                dtype=dtype,
            )
        )

        band_pass = (
            band_pass
            * window[
                tf.newaxis,
                :
            ]
        )

        # ----------------------------------------------------------
        # Stable normalization.
        #
        # At n=0:
        #
        # band_pass =
        #     2 * (high - low)
        #
        # Normalize directly using that known value instead
        # of reduce_max(), which gives us a simpler gradient.
        # ----------------------------------------------------------

        bandwidth = (
            high
            - low
        )

        epsilon = tf.cast(
            1e-8,
            dtype,
        )

        normalization = tf.maximum(
            two * bandwidth,
            epsilon,
        )

        band_pass = (
            band_pass
            / normalization
        )

        if self.debug_checks:
            tf.debugging.assert_all_finite(
                band_pass,
                (
                    "SincConv1D generated "
                    "non-finite FIR coefficients."
                ),
            )

        # ----------------------------------------------------------
        # TensorFlow Conv1D expects:
        #
        # (kernel_width, input_channels, output_channels)
        #
        # Current:
        #
        # (filters, kernel_width)
        # ----------------------------------------------------------

        filters = tf.transpose(
            band_pass,
            perm=(
                1,
                0,
            ),
        )

        filters = filters[
            :,
            tf.newaxis,
            :,
        ]

        return filters

    # ==============================================================
    # Forward pass
    # ==============================================================

    def call(
        self,
        inputs,
    ):
        # Perform the FIR filtering itself in float32.
        x = tf.cast(
            inputs,
            tf.float32,
        )

        filters = (
            self._make_filters()
        )

        if self.padding == "causal":
            effective_kernel = (
                self.dilation_rate
                * (
                    self.kernel_size
                    - 1
                )
                + 1
            )

            left_padding = (
                effective_kernel
                - 1
            )

            x = tf.pad(
                x,
                paddings=[
                    [0, 0],
                    [
                        left_padding,
                        0,
                    ],
                    [0, 0],
                ],
            )

            padding = "VALID"

        else:
            padding = (
                self.padding.upper()
            )

        outputs = tf.nn.conv1d(
            input=x,
            filters=filters,
            stride=self.strides,
            padding=padding,
            data_format="NWC",
            dilations=(
                self.dilation_rate
            ),
        )

        if self.bias_weight is not None:
            outputs = tf.nn.bias_add(
                outputs,
                self.bias_weight,
                data_format="NWC",
            )

        if self.debug_checks:
            tf.debugging.assert_all_finite(
                outputs,
                (
                    "SincConv1D produced "
                    "non-finite outputs."
                ),
            )

        output_dtype = (
            tf.dtypes.as_dtype(
                self.compute_dtype
            )
        )

        return tf.cast(
            outputs,
            output_dtype,
        )

    # ==============================================================
    # Output shape
    # ==============================================================

    def compute_output_shape(
        self,
        input_shape,
    ):
        batch_size = (
            input_shape[0]
        )

        input_length = (
            input_shape[1]
        )

        if input_length is None:
            output_length = None

        else:
            effective_kernel = (
                self.dilation_rate
                * (
                    self.kernel_size
                    - 1
                )
                + 1
            )

            if self.padding in {
                "same",
                "causal",
            }:
                output_length = (
                    input_length
                    + self.strides
                    - 1
                ) // self.strides

            else:
                output_length = (
                    input_length
                    - effective_kernel
                ) // self.strides + 1

                output_length = max(
                    output_length,
                    0,
                )

        return (
            batch_size,
            output_length,
            self.filters,
        )

    # ==============================================================
    # Serialization
    # ==============================================================

    def get_config(
        self,
    ):
        config = (
            super()
            .get_config()
        )

        config.update(
            {
                "filters": (
                    self.filters
                ),
                "kernel_size": (
                    self.kernel_size
                ),
                "sample_rate": (
                    self.sample_rate
                ),
                "strides": (
                    self.strides
                ),
                "padding": (
                    self.padding
                ),
                "dilation_rate": (
                    self.dilation_rate
                ),
                "min_low_hz": (
                    self.min_low_hz
                ),
                "min_band_hz": (
                    self.min_band_hz
                ),
                "frequency_margin_hz": (
                    self.frequency_margin_hz
                ),
                "use_bias": (
                    self.use_bias
                ),
                "debug_checks": (
                    self.debug_checks
                ),
            }
        )

        return config


__all__ = [
    "SincConv1D",
]
