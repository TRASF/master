"""Canonical TensorFlow audio augmentation implementation."""

from pathlib import Path
from typing import Mapping

import tensorflow as tf
import numpy as np

from wingbeat_ml.config.schema import AugmentConfig


class AudioAugmentor:
    def __init__(
        self,
        segment_length: int = 2400,
        config: AugmentConfig | Mapping[str, object] | None = None,
        seed: int | None = None,
        deterministic: bool = False,
        nomos_index: int | None = None,
    ):
        self.segment_length = segment_length
        self.aug_cfg = (
            config
            if isinstance(config, AugmentConfig)
            else AugmentConfig.model_validate(config or {})
        )
        self.seed = seed
        self.deterministic = deterministic
        self.nomos_index = nomos_index

        self.noise_cfg = self.aug_cfg.noise_overlay
        self.noise_envelope_cfg = self.noise_cfg.envelope_gain
        self.noise_post_gain_cfg = self.noise_cfg.post_gain_db
        self.pitch_cfg = self.aug_cfg.pitch_shift
        self.time_cfg = self.aug_cfg.time_shift
        self.gain_cfg = self.aug_cfg.random_gain
        self.gauss_cfg = self.aug_cfg.gaussian_noise
        self.mask_cfg = self.aug_cfg.time_masking
        self.pre_cfg = self.aug_cfg.pre_emphasis
        self.hpf_cfg = self.aug_cfg.high_pass
        self.rms_cfg = self.aug_cfg.rms_norm
        self.preprocess_cfg = self.aug_cfg.preprocess
        self.overlap_cfg = self.aug_cfg.segment_overlap

        import scipy.signal

        self.hpf_p = self.hpf_cfg.p
        self.pre_p = self.pre_cfg.p
        self.pitch_p = self.pitch_cfg.p
        self.time_p = self.time_cfg.p
        self.mask_p = self.mask_cfg.p
        self.gain_p = self.gain_cfg.p
        self.gauss_p = self.gauss_cfg.p
        self.noise_p = self.noise_cfg.p

        fc = self.hpf_cfg.fc
        if fc > 0:
            sr = 8000
            taps = scipy.signal.firwin(
                101,
                fc,
                fs=sr,
                pass_zero=False,
            )
            self.hpf_taps = tf.constant(taps, dtype=tf.float32)
        else:
            self.hpf_taps = None

    def pre_emphasis(self, x, coeff=0.97):
        """
        Applies pre-emphasis filter: y[t] = x[t] - coeff * x[t-1]
        """
        x = tf.cast(x, tf.float32)
        return tf.concat([x[:1], x[1:] - coeff * x[:-1]], axis=0)

    def rms_normalize(self, audio, target_rms=0., min_gain=0.1, max_gain=10.0):
        rms = tf.sqrt(tf.reduce_mean(tf.square(audio)) + 1e-8)
        gain = target_rms / rms
        gain = tf.clip_by_value(gain, min_gain, max_gain)
        audio = audio * gain
        audio.set_shape([self.segment_length])
        return audio

    def delta_waveform(self, x):
        """
        Computes the delta (first-order difference) of the waveform.
        """
        x = tf.cast(x, tf.float32)
        delta = tf.concat([[0.0], x[1:] - x[:-1]], axis=0)
        return delta

    def apply_time_masking(self, audio, seed=None):
        """
        Applies time masking by setting a random segment of the audio to zero.
        """
        if seed is None:
            seed = tf.constant([0, 0], dtype=tf.int64)
        num_masks = self.mask_cfg.num_masks
        max_mask_size = self.mask_cfg.max_mask_size

        for i in range(num_masks):
            max_mask_size = tf.minimum(tf.cast(max_mask_size, tf.int32), self.segment_length)
            max_mask_size = tf.maximum(max_mask_size, 1)
            min_mask_size = tf.minimum(tf.constant(10, dtype=tf.int32), max_mask_size)

            mask_size_seed = tf.stack([tf.gather(seed, 0), tf.cast(100 + i, tf.int64)])
            start_idx_seed = tf.stack([tf.gather(seed, 0), tf.cast(200 + i, tf.int64)])

            mask_size = tf.random.stateless_uniform(
                [],
                seed=mask_size_seed,
                minval=min_mask_size,
                maxval=max_mask_size + 1,
                dtype=tf.int32,
            )
            start_idx = tf.random.stateless_uniform(
                [],
                seed=start_idx_seed,
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

    def random_segment(self, audio, seed=None):
        if seed is None:
            stateless_seed = tf.constant([0, 42], dtype=tf.int64)
        else:
            seed = tf.cast(tf.convert_to_tensor(seed), tf.int64)

            # Dataset code supplies a scalar seed, while direct callers may
            # already supply TensorFlow's required two-element seed.
            if seed.shape.rank == 0:
                stateless_seed = tf.stack([
                    seed,
                    tf.constant(42, dtype=tf.int64),
                ])
            else:
                stateless_seed = tf.reshape(seed, [2])

        audio_len = tf.shape(audio)[0]
        pad_size = tf.maximum(0, self.segment_length - audio_len)
        audio = tf.pad(audio, [[0, pad_size]])
        audio_len = tf.shape(audio)[0]
        max_start = audio_len - self.segment_length
        start_idx = tf.random.stateless_uniform(
            [],
            seed=stateless_seed,
            minval=0,
            maxval=max_start + 1,
            dtype=tf.int32,
        )
        segment = audio[start_idx : start_idx + self.segment_length]
        segment.set_shape([self.segment_length])
        return segment

    def create_segments(self, audio, label, seed=None, training=True, sample_id=None):
        if seed is None:
            seed = tf.constant([0, 0], dtype=tf.int64)
        else:
            seed = tf.cast(tf.convert_to_tensor(seed), tf.int64)
            if seed.shape.rank == 0:
                seed = tf.stack([seed, tf.constant(0, dtype=tf.int64)])
            else:
                seed = tf.reshape(seed, [2])

        audio = tf.cast(audio, tf.float32)
        audio_len = tf.shape(audio)[0]

        if training:
            # Overlap seed
            overlap_seed = tf.stack([tf.gather(seed, 0), tf.gather(seed, 1) ^ tf.constant(1, dtype=tf.int64)])
            # Offset seed
            offset_seed = tf.stack([tf.gather(seed, 0), tf.gather(seed, 1) ^ tf.constant(2, dtype=tf.int64)])
            # Shuffle seed
            shuffle_seed = tf.stack([tf.gather(seed, 0), tf.gather(seed, 1) ^ tf.constant(3, dtype=tf.int64)])

            # Random overlap between the ranges provided in overlap_cfg
            overlap_range = self.overlap_cfg.train
            if isinstance(overlap_range, (int, float)):
                overlap_range = [overlap_range, overlap_range]

            random_overlap = tf.random.stateless_uniform([], seed=overlap_seed, minval=float(overlap_range[0]), maxval=float(overlap_range[1]))
            # Convert overlap ratio to step ratio
            current_step_ratio = 1.0 - random_overlap
        else:
            # Evaluation/Validation/Test: Use val overlap
            val_overlap = self.overlap_cfg.val
            current_step_ratio = 1.0 - float(val_overlap)

        step = tf.cast(tf.cast(self.segment_length, tf.float32) * current_step_ratio, tf.int32)
        step = tf.maximum(step, 1)

        if training:
            # Slicing from offset is safer than tf.roll and gives a random start point
            max_offset = tf.minimum(step, tf.maximum(audio_len, 1))
            offset = tf.random.stateless_uniform([], seed=offset_seed, minval=0, maxval=max_offset, dtype=tf.int32)
            audio = audio[offset:]

        # Create overlapping frames
        frames = tf.signal.frame(audio, frame_length=self.segment_length, frame_step=step, pad_end=True)
        num_frames = tf.shape(frames)[0]
        labels = tf.repeat(tf.expand_dims(tf.cast(label, tf.int32), 0), num_frames, axis=0)

        if sample_id is not None:
            sample_ids = tf.repeat(tf.expand_dims(sample_id, 0), num_frames, axis=0)
        else:
            sample_ids = tf.fill([num_frames], tf.constant("", dtype=tf.string))

        if training:
            max_segments = getattr(self.aug_cfg, "max_segments_per_file", 100)

            if self.nomos_index is not None:
                is_nomos = tf.equal(label, self.nomos_index)
                current_max = tf.cond(is_nomos, lambda: max_segments // 5, lambda: max_segments)
            else:
                current_max = max_segments

            indices = tf.range(num_frames)
            shuffled_indices = tf.random.experimental.stateless_shuffle(indices, seed=shuffle_seed)
            sliced_indices = shuffled_indices[:current_max]

            frames = tf.gather(frames, sliced_indices)
            labels = tf.gather(labels, sliced_indices)
            sample_ids = tf.gather(sample_ids, sliced_indices)
            # Create a unique seed per segment based on slice index
            idx_cast = tf.cast(sliced_indices, tf.int64)
            c1 = tf.constant(-7046029254386353131, dtype=tf.int64)
            c2 = tf.constant(-4658826500735392327, dtype=tf.int64)
            s0 = tf.gather(seed, 0) ^ (idx_cast * c1)
            s1 = tf.gather(seed, 1) ^ (idx_cast * c2 + 1000)
            segment_seeds = tf.stack([s0, s1], axis=1)
        else:
            segment_seeds = tf.zeros([num_frames, 2], dtype=tf.int64)

        frames.set_shape([None, self.segment_length])

        # Return as Dataset (Required for .interleave in dataset.py)
        return tf.data.Dataset.from_tensor_slices((frames, labels, segment_seeds, sample_ids))

    def build_noise_dataset(self, noise_dirs, load_fn):
        noise_paths = []

        for noise_dir in noise_dirs:
            path_obj = Path(noise_dir)
            if path_obj.is_dir():
                noise_paths.extend(
                    str(path) for path in path_obj.rglob("*.npy")
                )
                noise_paths.extend(
                    str(path) for path in path_obj.rglob("*.wav")
                )

        noise_paths = sorted(set(noise_paths))

        if not noise_paths:
            return None

        options = tf.data.Options()
        options.experimental_deterministic = self.deterministic

        noise_ds = tf.data.Dataset.from_tensor_slices(noise_paths)
        noise_ds = noise_ds.with_options(options)

        noise_ds = noise_ds.map(
            lambda path: load_fn(path),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=self.deterministic,
        )

        noise_ds = noise_ds.cache()

        noise_seed = self.seed if self.deterministic else None
        noise_ds = noise_ds.shuffle(
            buffer_size=len(noise_paths),
            seed=noise_seed,
            reshuffle_each_iteration=True,
        ).repeat()

        noise_seed_ds = tf.data.Dataset.random(seed=self.seed + 99 if self.seed is not None else None, rerandomize_each_iteration=True)
        noise_ds = tf.data.Dataset.zip((noise_ds, noise_seed_ds))

        random_parallel_calls = tf.data.AUTOTUNE

        noise_ds = noise_ds.map(
            lambda audio, seed: self.random_segment(audio, seed),
            num_parallel_calls=random_parallel_calls,
            deterministic=self.deterministic,
        )

        return noise_ds.with_options(options)


    def sample_noise_snr(self, fallback_range, seed):
        distribution = self.noise_cfg.snr_distribution
        if not distribution:
            return tf.random.stateless_uniform([], seed=seed, minval=float(fallback_range[0]), maxval=float(fallback_range[1]))

        r_seed = tf.stack([tf.gather(seed, 0), tf.constant(101, dtype=tf.int64)])
        r = tf.random.stateless_uniform([], seed=r_seed)
        cumulative = 0.0
        branches = []
        for i, item in enumerate(distribution):
            cumulative += item.p
            low, high = item.snr_db or fallback_range
            branch_seed = tf.stack([tf.gather(seed, 0), tf.cast(1000 + i, tf.int64)])
            branches.append((
                r < cumulative,
                lambda low=low, high=high, bs=branch_seed: tf.random.stateless_uniform([], seed=bs, minval=float(low), maxval=float(high)),
            ))

        last_low, last_high = distribution[-1].snr_db or fallback_range
        last_seed = tf.stack([tf.gather(seed, 0), tf.constant(999, dtype=tf.int64)])
        return tf.case(
            branches,
            default=lambda: tf.random.stateless_uniform([], seed=last_seed, minval=float(last_low), maxval=float(last_high)),
            exclusive=False,
        )

    def apply_noise_envelope(self, noise, seed):
        min_gain = float(self.noise_envelope_cfg[0])
        max_gain = float(self.noise_envelope_cfg[1])
        start_seed = tf.stack([tf.gather(seed, 0), tf.constant(201, dtype=tf.int64)])
        end_seed = tf.stack([tf.gather(seed, 0), tf.constant(202, dtype=tf.int64)])
        start_gain = tf.random.stateless_uniform([], seed=start_seed, minval=min_gain, maxval=max_gain)
        end_gain = tf.random.stateless_uniform([], seed=end_seed, minval=min_gain, maxval=max_gain)
        envelope = tf.linspace(start_gain, end_gain, tf.shape(noise)[0])
        return noise * envelope

    def add_noise(self, audio, noise, snr_range, seed):
        env_seed = tf.stack([tf.gather(seed, 0), tf.constant(301, dtype=tf.int64)])
        snr_seed = tf.stack([tf.gather(seed, 0), tf.constant(302, dtype=tf.int64)])
        gain_seed = tf.stack([tf.gather(seed, 0), tf.constant(303, dtype=tf.int64)])

        noise = self.apply_noise_envelope(noise, seed=env_seed)

        audio_rms = tf.sqrt(tf.reduce_mean(tf.square(audio)) + 1e-12)
        noise_rms = tf.sqrt(tf.reduce_mean(tf.square(noise)) + 1e-12)
        snr_db = self.sample_noise_snr(snr_range, seed=snr_seed)
        snr_lin = tf.pow(10.0, snr_db / 20.0)
        scale = audio_rms / (noise_rms * snr_lin + 1e-12)
        augmented = audio + (noise * scale)

        gain_db = tf.random.stateless_uniform(
            [],
            seed=gain_seed,
            minval=float(self.noise_post_gain_cfg[0]),
            maxval=float(self.noise_post_gain_cfg[1]),
        )
        augmented = augmented * tf.pow(10.0, gain_db / 20.0)

        peak = tf.reduce_max(tf.abs(augmented)) + 1e-8
        return tf.cond(peak > 0.95, lambda: augmented / peak * 0.95, lambda: augmented)

    def pitch_shift(self, audio, semitones_range, seed):
        """
        Approximates pitch shift via resampling using tf.image.resize.
        """
        semitones = tf.random.stateless_uniform([], seed=seed, minval=float(semitones_range[0]), maxval=float(semitones_range[1]))
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

    def time_shift(self, audio, rate_range, seed):
        rate = tf.random.stateless_uniform([], seed=seed, minval=float(rate_range[0]), maxval=float(rate_range[1]))
        shift = tf.cast(tf.cast(self.segment_length, tf.float32) * rate, tf.int32)

        def shift_right():
            s = tf.minimum(shift, self.segment_length - 1)
            pad = tf.zeros([s], dtype=audio.dtype)
            return tf.concat([pad, audio[:-s]], axis=0)

        def shift_left():
            s = tf.minimum(-shift, self.segment_length - 1)
            pad = tf.zeros([s], dtype=audio.dtype)
            return tf.concat([audio[s:], pad], axis=0)

        return tf.case(
            [(shift > 0, shift_right), (shift < 0, shift_left)],
            default=lambda: audio,
            exclusive=True
        )

    def random_gain(self, audio, gain_db_range, seed):
        gain_db = tf.random.stateless_uniform([], seed=seed, minval=float(gain_db_range[0]), maxval=float(gain_db_range[1]))
        gain = tf.pow(10.0, gain_db / 20.0)
        return audio * gain

    def add_gaussian_noise(self, audio, snr_range, seed):
        audio_rms = tf.sqrt(tf.reduce_mean(tf.square(audio)) + 1e-9)
        snr_seed = tf.stack([tf.gather(seed, 0), tf.constant(1001, dtype=tf.int64)])
        noise_seed = tf.stack([tf.gather(seed, 0), tf.constant(1002, dtype=tf.int64)])
        snr_db = tf.random.stateless_uniform([], seed=snr_seed, minval=float(snr_range[0]), maxval=float(snr_range[1]))
        snr_lin = tf.pow(10.0, snr_db / 20.0)
        noise_rms = audio_rms / snr_lin
        noise = tf.random.stateless_normal(tf.shape(audio), seed=noise_seed, mean=0.0, stddev=noise_rms)
        return audio + noise

    def apply_hpf(self, audio):
        if self.hpf_taps is None:
            return audio
        # Conv1d expects [batch, in_channels, in_width] or similar
        # audio is [length]
        audio_padded = tf.reshape(audio, [1, self.segment_length, 1])
        taps = tf.reshape(self.hpf_taps, [101, 1, 1])
        filtered = tf.nn.conv1d(audio_padded, taps, stride=1, padding='SAME')
        return tf.reshape(filtered, [self.segment_length])

    def apply_post_processing(self, audio, label, seed=None, noise=None, augment=True):
        if seed is None:
            seed = tf.constant([0, 0], dtype=tf.int64)
        else:
            seed = tf.cast(tf.convert_to_tensor(seed), tf.int64)
            if seed.shape.rank == 0:
                seed = tf.stack([seed, tf.constant(0, dtype=tf.int64)])
            else:
                seed = tf.reshape(seed, [2])
        seed_0 = tf.gather(seed, 0)
        seed_1 = tf.gather(seed, 1)
        # ----------------------------------------------------
        # Phase 1: Signal Conditioning (Structure)
        # ----------------------------------------------------
        # High-pass filter
        if self.hpf_p > 0.0 and self.hpf_taps is not None:
            if not augment:
                audio = self.apply_hpf(audio)
            else:
                hpf_toss_seed = tf.stack([seed_0, seed_1 ^ tf.constant(10, dtype=tf.int64)])
                if tf.random.stateless_uniform([], seed=hpf_toss_seed) < self.hpf_p:
                    audio = self.apply_hpf(audio)

        # Pre-emphasis
        if self.pre_p > 0.0:
            coeff = self.pre_cfg.coeff
            if not augment:
                audio = self.pre_emphasis(audio, coeff=coeff)
            else:
                pre_toss_seed = tf.stack([seed_0, seed_1 ^ tf.constant(20, dtype=tf.int64)])
                if tf.random.stateless_uniform([], seed=pre_toss_seed) < self.pre_p:
                    audio = self.pre_emphasis(audio, coeff=coeff)

        # ----------------------------------------------------
        # Phase 2 & 3: Augmentations (Timing & Energy)
        # ----------------------------------------------------
        if augment:
            # Pitch Shift
            if self.pitch_p > 0.0:
                pitch_toss_seed = tf.stack([seed_0, seed_1 ^ tf.constant(30, dtype=tf.int64)])
                pitch_val_seed = tf.stack([seed_0, seed_1 ^ tf.constant(31, dtype=tf.int64)])
                if tf.random.stateless_uniform([], seed=pitch_toss_seed) < self.pitch_p:
                    audio = self.pitch_shift(audio, self.pitch_cfg.semitones, seed=pitch_val_seed)

            # Time Shift
            if self.time_p > 0.0:
                time_toss_seed = tf.stack([seed_0, seed_1 ^ tf.constant(40, dtype=tf.int64)])
                time_val_seed = tf.stack([seed_0, seed_1 ^ tf.constant(41, dtype=tf.int64)])
                if tf.random.stateless_uniform([], seed=time_toss_seed) < self.time_p:
                    audio = self.time_shift(audio, self.time_cfg.rate, seed=time_val_seed)

            # Time Masking
            if self.mask_p > 0.0:
                mask_toss_seed = tf.stack([seed_0, seed_1 ^ tf.constant(50, dtype=tf.int64)])
                mask_val_seed = tf.stack([seed_0, seed_1 ^ tf.constant(51, dtype=tf.int64)])
                if tf.random.stateless_uniform([], seed=mask_toss_seed) < self.mask_p:
                    audio = self.apply_time_masking(audio, seed=mask_val_seed)

            # Gaussian Noise
            if self.gauss_p > 0.0:
                gauss_toss_seed = tf.stack([seed_0, seed_1 ^ tf.constant(70, dtype=tf.int64)])
                gauss_val_seed = tf.stack([seed_0, seed_1 ^ tf.constant(71, dtype=tf.int64)])
                if tf.random.stateless_uniform([], seed=gauss_toss_seed) < self.gauss_p:
                    audio = self.add_gaussian_noise(audio, self.gauss_cfg.snr_db, seed=gauss_val_seed)

            # Noise Overlay (External Noise Bank)
            if noise is not None and self.noise_p > 0.0:
                noise_toss_seed = tf.stack([seed_0, seed_1 ^ tf.constant(80, dtype=tf.int64)])
                noise_val_seed = tf.stack([seed_0, seed_1 ^ tf.constant(81, dtype=tf.int64)])
                if tf.random.stateless_uniform([], seed=noise_toss_seed) < self.noise_p:
                    audio = self.add_noise(audio, noise, self.noise_cfg.snr_db, seed=noise_val_seed)

        # ----------------------------------------------------
        # Phase 4: Final Standardization (The Capstone)
        # ----------------------------------------------------
        # 1. First, normalize energy so model sees consistent volume
        if self.preprocess_cfg.dc_removal:
            audio -= tf.reduce_mean(audio)

        audio = self.rms_normalize(
            audio,
            target_rms=self.rms_cfg.target_rms,
            min_gain=self.rms_cfg.min_gain,
            max_gain=self.rms_cfg.max_gain
        )

        # 2. Apply random gain AFTER RMS normalization so gain variation persists
        if augment and self.gain_p > 0.0:
            gain_toss_seed = tf.stack([seed_0, seed_1 ^ tf.constant(60, dtype=tf.int64)])
            gain_val_seed = tf.stack([seed_0, seed_1 ^ tf.constant(61, dtype=tf.int64)])
            if tf.random.stateless_uniform([], seed=gain_toss_seed) < self.gain_p:
                audio = self.random_gain(audio, self.gain_cfg.gain_db, seed=gain_val_seed)

        # 3. Finally, clip to prevent extreme outliers
        audio = tf.clip_by_value(audio, -1.0, 1.0)

        audio.set_shape([self.segment_length])

        return audio, tf.cast(label, tf.int32)


__all__ = ["AudioAugmentor"]
