from pathlib import Path
import tensorflow as tf
import numpy as np

class AudioAugmentor:
    def __init__(self, segment_length: int = 2400, config: dict = None,
                 seed: int = None, deterministic: bool = False,
                 nomos_index: int = None):
        self.segment_length = segment_length
        self.cfg = config or {}
        self.seed = seed
        self.deterministic = deterministic
        self.nomos_index = nomos_index
        self.parallel_calls = 1 if deterministic else tf.data.AUTOTUNE

        # Initialize augmentation parameters with robust merging
        self.noise_cfg = self._get_cfg('noise_overlay', {'p': 0.0, 'snr_db': [10, 20]})
        self.noise_envelope_cfg = self.noise_cfg.get('envelope_gain', [0.7, 1.0])
        self.noise_post_gain_cfg = self.noise_cfg.get('post_gain_db', [-6.0, 3.0])

        self.pitch_cfg = self._get_cfg('pitch_shift', {'p': 0.0, 'semitones': [-0.5, 0.5]})
        self.time_cfg = self._get_cfg('time_shift', {'p': 0.0, 'rate': [-0.1, 0.1]})
        self.gain_cfg = self._get_cfg('random_gain', {'p': 0.0, 'gain_db': [-6, 6]})
        self.gauss_cfg = self._get_cfg('gaussian_noise', {'p': 0.0, 'snr_db': [10, 20]})
        self.mask_cfg = self._get_cfg('time_masking', {'p': 0.0, 'num_masks': 1, 'max_mask_size': 400})
        self.pre_cfg = self._get_cfg('pre_emphasis', {'p': 0.0, 'coeff': 0.97})

    def _get_cfg(self, key, defaults):
        """
        Merges user config with defaults to prevent KeyErrors for missing sub-keys.
        """
        user_val = self.cfg.get(key, {})
        if user_val is None: return defaults
        return {**defaults, **user_val}

    @tf.function
    def pre_emphasis(self, x, coeff=0.97):
        """
        Applies pre-emphasis filter: y[t] = x[t] - coeff * x[t-1]
        """
        x = tf.cast(x, tf.float32)
        return tf.concat([x[:1], x[1:] - coeff * x[:-1]], axis=0)

    @tf.function
    def rms_normalize(self, audio, target_rms=0.1):
        rms = tf.sqrt(tf.reduce_mean(tf.square(audio)) + 1e-8)
        audio = audio * (target_rms / rms)
        return audio

    @tf.function
    def delta_waveform(self, x):
        """
        Computes the delta (first-order difference) of the waveform.
        """
        x = tf.cast(x, tf.float32)
        delta = tf.concat([[0.0], x[1:] - x[:-1]], axis=0)
        return delta

    @tf.function
    def apply_time_masking(self, audio):
        """
        Applies time masking by setting a random segment of the audio to zero.
        """
        num_masks = self.mask_cfg.get('num_masks', 1)
        max_mask_size = self.mask_cfg.get('max_mask_size', 400)

        for _ in range(num_masks):
            max_mask_size = tf.minimum(tf.cast(max_mask_size, tf.int32), self.segment_length)
            max_mask_size = tf.maximum(max_mask_size, 1)
            min_mask_size = tf.minimum(tf.constant(10, dtype=tf.int32), max_mask_size)

            mask_size = tf.random.uniform(
                [],
                minval=min_mask_size,
                maxval=max_mask_size + 1,
                dtype=tf.int32,
            )
            start_idx = tf.random.uniform(
                [],
                minval=0,
                maxval=self.segment_length - mask_size + 1,
                dtype=tf.int32,
            )

            mask = tf.concat([
                tf.ones([start_idx]),
                tf.zeros([mask_size]),
                tf.ones([self.segment_length - start_idx - mask_size])
            ], axis=0)
            audio = audio * mask
        return audio

    @tf.function
    def random_segment(self, audio):
        audio_len = tf.shape(audio)[0]
        pad_size = tf.maximum(0, self.segment_length - audio_len)
        audio = tf.pad(audio, [[0, pad_size]])
        audio_len = tf.shape(audio)[0]
        max_start = audio_len - self.segment_length
        start_idx = tf.random.uniform([], minval=0, maxval=max_start + 1, dtype=tf.int32)
        segment = audio[start_idx : start_idx + self.segment_length]
        segment.set_shape([self.segment_length])
        return segment

    @tf.function
    def create_segments(self, audio, label, step_ratio=0.5, training=True):
        audio = tf.cast(audio, tf.float32)
        audio_len = tf.shape(audio)[0]

        # During training, randomize step_ratio (overlap)
        # overlap [0.0, 0.7] -> step_ratio [0.3, 1.0]
        if training:
            dyn_step_ratio = tf.random.uniform([], minval=0.3, maxval=1.0)
        else:
            dyn_step_ratio = tf.cast(step_ratio, tf.float32)

        step = tf.cast(tf.cast(self.segment_length, tf.float32) * dyn_step_ratio, tf.int32)
        step = tf.maximum(step, 1)

        if training:
            # Slicing from offset is safer than tf.roll
            max_offset = tf.minimum(step, tf.maximum(audio_len, 1))
            offset = tf.random.uniform([], minval=0, maxval=max_offset, dtype=tf.int32)
            audio = audio[offset:]

        frames = tf.signal.frame(audio, frame_length=self.segment_length, frame_step=step, pad_end=True)
        num_frames = tf.shape(frames)[0]
        labels = tf.repeat(tf.expand_dims(label, 0), num_frames, axis=0)

        frames.set_shape([None, self.segment_length])
        return tf.data.Dataset.from_tensor_slices((frames, labels))

    def build_noise_dataset(self, noise_dirs, load_fn):
        noise_paths = []
        for n_dir in noise_dirs:
            path_obj = Path(n_dir)
            if path_obj.is_dir():
                noise_paths.extend([str(p) for p in path_obj.rglob("*.npy")])
                noise_paths.extend([str(p) for p in path_obj.rglob("*.wav")])
        noise_paths = sorted(set(noise_paths))
        if not noise_paths:
            return None
        noise_ds = tf.data.Dataset.from_tensor_slices(noise_paths)
        options = tf.data.Options()
        options.experimental_deterministic = self.deterministic
        noise_ds = noise_ds.with_options(options)
        noise_ds = noise_ds.map(
            lambda p: load_fn(p),
            num_parallel_calls=self.parallel_calls,
            deterministic=self.deterministic,
        )
        noise_ds = noise_ds.cache()
        noise_seed = self.seed if self.deterministic else None
        noise_ds = noise_ds.shuffle(
            len(noise_paths),
            seed=noise_seed,
            reshuffle_each_iteration=True,
        ).repeat()
        noise_ds = noise_ds.map(
            lambda x: self.random_segment(x),
            num_parallel_calls=self.parallel_calls,
            deterministic=self.deterministic,
        )
        return noise_ds.with_options(options)

    @tf.function
    def sample_noise_snr(self, fallback_range):
        if 'snr_distribution' not in self.noise_cfg:
            return tf.random.uniform([], minval=float(fallback_range[0]), maxval=float(fallback_range[1]))

        r = tf.random.uniform([])
        cumulative = 0.0
        branches = []
        for item in self.noise_cfg['snr_distribution']:
            cumulative += float(item.get('p', 0.0))
            low, high = item.get('snr_db', fallback_range)
            branches.append((
                r < cumulative,
                lambda low=low, high=high: tf.random.uniform([], minval=float(low), maxval=float(high)),
            ))

        low, high = self.noise_cfg['snr_distribution'][-1].get('snr_db', fallback_range)
        return tf.case(
            branches,
            default=lambda: tf.random.uniform([], minval=float(low), maxval=float(high)),
            exclusive=False,
        )

    @tf.function
    def apply_noise_envelope(self, noise):
        min_gain = float(self.noise_envelope_cfg[0])
        max_gain = float(self.noise_envelope_cfg[1])
        start_gain = tf.random.uniform([], min_gain, max_gain)
        end_gain = tf.random.uniform([], min_gain, max_gain)
        envelope = tf.linspace(start_gain, end_gain, tf.shape(noise)[0])
        return noise * envelope

    @tf.function
    def add_noise(self, audio, noise, snr_range):
        noise = self.apply_noise_envelope(noise)

        audio_rms = tf.sqrt(tf.reduce_mean(tf.square(audio)) + 1e-12)
        noise_rms = tf.sqrt(tf.reduce_mean(tf.square(noise)) + 1e-12)
        snr_db = self.sample_noise_snr(snr_range)
        snr_lin = tf.pow(10.0, snr_db / 20.0)
        scale = audio_rms / (noise_rms * snr_lin + 1e-12)
        augmented = audio + (noise * scale)

        gain_db = tf.random.uniform(
            [],
            minval=float(self.noise_post_gain_cfg[0]),
            maxval=float(self.noise_post_gain_cfg[1]),
        )
        augmented = augmented * tf.pow(10.0, gain_db / 20.0)

        peak = tf.reduce_max(tf.abs(augmented)) + 1e-8
        return tf.cond(peak > 0.95, lambda: augmented / peak * 0.95, lambda: augmented)

    @tf.function
    def pitch_shift(self, audio, semitones_range):
        """
        Approximates pitch shift via resampling using tf.image.resize.
        """
        semitones = tf.random.uniform([], float(semitones_range[0]), float(semitones_range[1]))
        factor = tf.pow(2.0, semitones / 12.0)
        new_len = tf.cast(tf.cast(self.segment_length, tf.float32) / factor, tf.int32)

        audio_4d = tf.reshape(audio, [1, 1, self.segment_length, 1])
        resized = tf.image.resize(audio_4d, [1, new_len], method='bilinear')
        resized = tf.reshape(resized, [-1])

        res_len = tf.shape(resized)[0]
        def pad_it():
            return tf.pad(resized, [[0, self.segment_length - res_len]])
        def crop_it():
            return resized[:self.segment_length]

        final = tf.cond(res_len < self.segment_length, pad_it, crop_it)
        final.set_shape([self.segment_length])
        return final

    @tf.function
    def time_shift(self, audio, rate_range):
        rate = tf.random.uniform([], float(rate_range[0]), float(rate_range[1]))
        shift = tf.cast(tf.cast(self.segment_length, tf.float32) * rate, tf.int32)
        return tf.roll(audio, shift=shift, axis=0)

    @tf.function
    def random_gain(self, audio, gain_db_range):
        gain_db = tf.random.uniform([], float(gain_db_range[0]), float(gain_db_range[1]))
        gain = tf.pow(10.0, gain_db / 20.0)
        return audio * gain

    @tf.function
    def add_gaussian_noise(self, audio, snr_range):
        audio_rms = tf.sqrt(tf.reduce_mean(tf.square(audio)) + 1e-9)
        snr_db = tf.random.uniform([], float(snr_range[0]), float(snr_range[1]))
        snr_lin = tf.pow(10.0, snr_db / 20.0)
        noise_rms = audio_rms / snr_lin
        noise = tf.random.normal(tf.shape(audio), mean=0.0, stddev=noise_rms)
        return audio + noise

    @tf.function
    def apply_post_processing(self, audio, label, noise=None, augment=True):
        # 1. Check if it's the background class (No.Mos)
        is_nomos = False
        if self.nomos_index is not None:
            is_nomos = tf.equal(tf.cast(label, tf.int32), tf.cast(self.nomos_index, tf.int32))

        # 2. Random Augmentations (ONLY for mosquitoes, NOT for No.Mos)
        if augment and not is_nomos:
            # Pitch Shift
            if tf.random.uniform([]) < float(self.pitch_cfg['p']):
                audio = self.pitch_shift(audio, self.pitch_cfg['semitones'])

            # Time Shift
            if tf.random.uniform([]) < float(self.time_cfg['p']):
                audio = self.time_shift(audio, self.time_cfg['rate'])

            # Random Gain
            if tf.random.uniform([]) < float(self.gain_cfg['p']):
                audio = self.random_gain(audio, self.gain_cfg['gain_db'])

            # Gaussian Noise
            if tf.random.uniform([]) < float(self.gauss_cfg['p']):
                audio = self.add_gaussian_noise(audio, self.gauss_cfg['snr_db'])

            # Noise Overlay (External Noise Bank)
            if noise is not None and tf.random.uniform([]) < float(self.noise_cfg['p']):
                audio = self.add_noise(audio, noise, self.noise_cfg['snr_db'])

            # Time Masking
            if tf.random.uniform([]) < float(self.mask_cfg['p']):
                audio = self.apply_time_masking(audio)

        audio = self.pre_emphasis(audio)
        audio = self.delta_waveform(audio)
        audio = self.rms_normalize(audio)
        audio = tf.clip_by_value(audio, -1.0, 1.0)
        audio.set_shape([self.segment_length])

        return audio, label
