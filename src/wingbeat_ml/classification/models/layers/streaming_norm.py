import tensorflow as tf


class StreamingBioacousticNormTF(tf.keras.layers.Layer):
    """
    Exact TensorFlow match for the ESP32 DMA streaming normalization pipeline.
    Performs 1-pole IIR DC blocking, chunked L1 energy tracking, steady-state IIR
    gain smoothing, and intra-block linear ramping. Supports both (T,) and (B, T).
    """
    def __init__(
        self,
        block_size: int = 128,
        r: float = 0.995,
        noise_floor: float = 1e-4,
        g_smooth_alpha: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.block_size = int(block_size)
        self.r = float(r)
        self.noise_floor = float(noise_floor)
        self.g_smooth_alpha = float(g_smooth_alpha)

    def build(self, input_shape):
        ramp = tf.linspace(0.0, 1.0, self.block_size)
        self.ramp_template = tf.reshape(tf.cast(ramp, tf.float32), (1, 1, self.block_size))
        super().build(input_shape)

    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.cast(x, tf.float32)

        # Auto-handle 1D (T,) vs 2D (B, T)
        is_1d = tf.equal(tf.rank(x), 1)
        x_2d = tf.cond(is_1d, lambda: tf.expand_dims(x, 0), lambda: x)

        shape = tf.shape(x_2d)
        B, T = shape[0], shape[1]
        N = self.block_size

        pad_len = (N - (T % N)) % N
        x_padded = tf.pad(x_2d, [[0, 0], [0, pad_len]], mode="CONSTANT", constant_values=0.0)
        total_samples = T + pad_len

        # Pass 1: 1-Pole IIR DC Blocker (d[n] = x[n] - x[n-1])
        x_shift = tf.pad(x_padded[:, :-1], [[0, 0], [1, 0]], mode="CONSTANT", constant_values=0.0)
        d = x_padded - x_shift

        d_t = tf.transpose(d, [1, 0])
        r_const = tf.constant(self.r, dtype=tf.float32)

        def iir_dc_step(y_prev, d_curr):
            return d_curr + r_const * y_prev

        y_t = tf.scan(
            iir_dc_step,
            d_t,
            initializer=tf.zeros_like(d_t[0]),
            parallel_iterations=1,
            swap_memory=True
        )
        y = tf.transpose(y_t, [1, 0])

        # Block Reshape & L1 Energy Tracking
        num_blocks = total_samples // N
        y_blocks = tf.reshape(y, (B, num_blocks, N))

        mean_l1 = tf.reduce_mean(tf.abs(y_blocks), axis=-1)
        scale = tf.maximum(mean_l1, self.noise_floor)
        g_target = 1.0 / scale

        # Steady-State Gain Smoothing across blocks
        g_init = g_target[:, :1]
        g_centered = g_target - g_init
        g_centered_t = tf.transpose(g_centered, [1, 0])
        alpha = tf.constant(self.g_smooth_alpha, dtype=tf.float32)

        def gain_smooth_step(v_prev, u_curr):
            return (1.0 - alpha) * v_prev + alpha * u_curr

        v_t = tf.scan(
            gain_smooth_step,
            g_centered_t,
            initializer=tf.zeros_like(g_centered_t[0]),
            parallel_iterations=1,
            swap_memory=True
        )
        v = tf.transpose(v_t, [1, 0])
        g_dest = v + g_init

        # Linear Intra-Block Gain Ramping
        g_prev = tf.concat([g_init, g_dest[:, :-1]], axis=1)
        g_start = tf.expand_dims(g_prev, -1)
        g_end = tf.expand_dims(g_dest, -1)

        gain_envelope = g_start + (g_end - g_start) * self.ramp_template

        # Pass 2: Linear Scaling & Unpad
        out_blocks = y_blocks * gain_envelope
        out = tf.reshape(out_blocks, (B, total_samples))
        out_cropped = out[:, :T]

        return tf.cond(is_1d, lambda: tf.squeeze(out_cropped, 0), lambda: out_cropped)

    def get_config(self):
        config = super().get_config()
        config.update({
            "block_size": self.block_size,
            "r": self.r,
            "noise_floor": self.noise_floor,
            "g_smooth_alpha": self.g_smooth_alpha,
        })
        return config
