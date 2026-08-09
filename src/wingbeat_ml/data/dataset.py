"""Canonical TensorFlow dataset pipeline for Wingbeat ML."""

import tensorflow as tf
from wingbeat_ml.data.loading import DataLoader
from wingbeat_ml.augmentations.transforms import AudioAugmentor
from wingbeat_ml.config.schema import AugmentConfig
import numpy as np
import os
from pathlib import Path
from typing import Mapping
from wingbeat_ml.data.cache import (
    CACHE_SCHEMA_VERSION,
    materialize_tensorflow_cache,
    stable_cache_key,
)
from wingbeat_ml.data.splits import _split_paths as split_paths

def derive_sample_seed(sample_id_or_hash, global_seed, iteration_rnd=0, stage_id=0):
    """Stateless seed derivation from sample identity, global seed, epoch component, and stage ID."""
    if isinstance(sample_id_or_hash, (int, np.integer)):
        id_hash = tf.constant(int(sample_id_or_hash), dtype=tf.int64)
    elif isinstance(sample_id_or_hash, tf.Tensor) and sample_id_or_hash.dtype in (tf.int64, tf.int32):
        id_hash = tf.cast(sample_id_or_hash, tf.int64)
    else:
        sample_id_1d = tf.reshape(tf.cast(sample_id_or_hash, tf.string), [1])
        fp_bytes = tf.fingerprint(sample_id_1d)
        id_hash = tf.reshape(tf.bitcast(fp_bytes, tf.int64), [])

    g_seed = (
        tf.cast(global_seed, tf.int64)
        if global_seed is not None
        else tf.constant(0, dtype=tf.int64)
    )
    iter_comp = tf.cast(iteration_rnd, tf.int64)
    stage_comp = tf.cast(stage_id, tf.int64)

    c1 = tf.constant(-7046029254386353131, dtype=tf.int64)
    c2 = tf.constant(-4658826500735392327, dtype=tf.int64)

    s0 = id_hash ^ g_seed ^ (stage_comp * c1) ^ iter_comp
    s1 = (id_hash * c1 + iter_comp * c2) ^ g_seed ^ (stage_comp * c2)

    return tf.stack([s0, s1], axis=0)


class SupervisedDataset:
    def __init__(
        self,
        dataset_dir: str,
        val_dir: str = None,
        test_dir: str = None,
        sample_rate: int = 8000,
        segment_length: int = 2400,
        classes: list = None,
        noise_dirs: list = None,
        augment_cfg: AugmentConfig | None = None,
        seed: int = 42,
        deterministic: bool = False,
        nomos_index: int = None,
        labels_dict: dict = None,
        cache_cfg: dict = None,
    ):
        self.dataset_dir = dataset_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length

        self.data_loader = DataLoader(
            dataset_dir,
            sample_rate,
            classes,
            labels_dict=labels_dict,
        )

        self.noise_dirs = noise_dirs
        self.seed = seed
        self.deterministic = deterministic
        self.cache_cfg = cache_cfg or {}

        self.pure_parallel_calls = tf.data.AUTOTUNE

        self.random_parallel_calls = tf.data.AUTOTUNE

        self.prefetch_buffer = tf.data.AUTOTUNE

        self.nomos_index = nomos_index
        if self.nomos_index is None and classes:
            for i, name in enumerate(classes):
                compact_name = "".join(
                    character
                    for character in name.casefold()
                    if character.isalnum()
                )

                if compact_name == "nomos":
                    self.nomos_index = i
                    break

        self.augmentor = AudioAugmentor(
            segment_length,
            augment_cfg,
            seed=seed,
            deterministic=deterministic,
            nomos_index=self.nomos_index,
        )

        self.train_paths = None
        self.train_labels = None
        self.val_paths = None
        self.val_labels = None
        self.test_paths = None
        self.test_labels = None
        self.class_weights = None
        self.class_counts = None

    def _compute_balanced_class_weights(self, file_paths, labels):
        import wave
        # Retrieve average overlap/step size
        overlap_range = getattr(self.augmentor.aug_cfg.segment_overlap, "train", [0.0, 0.8])
        if not isinstance(overlap_range, (list, tuple)):
            overlap_range = [0.0, float(overlap_range)]
        avg_overlap = np.mean(overlap_range)
        avg_step = int(self.segment_length * (1.0 - avg_overlap))
        avg_step = max(avg_step, 1)

        max_segments = getattr(self.augmentor.aug_cfg, "max_segments_per_file", 100)

        counts = np.zeros(self.data_loader.num_classes, dtype=np.float32)
        for path, label in zip(file_paths, labels):
            num_samples = 0
            try:
                if path.endswith('.npy'):
                    num_samples = np.load(path, mmap_mode='r').shape[0]
                elif path.endswith('.wav') or path.endswith('.WAV'):
                    with wave.open(str(path), 'rb') as f:
                        num_samples = f.getnframes()
            except Exception:
                pass

            if num_samples == 0:
                num_samples = self.segment_length

            num_segments = int(np.ceil(num_samples / avg_step))

            current_max = max_segments
            if self.nomos_index is not None and label == self.nomos_index:
                current_max = max_segments // 5

            num_segments = min(num_segments, current_max)
            counts[label] += num_segments

        nonzero = counts > 0
        weights = np.ones_like(counts, dtype=np.float32)
        weights[nonzero] = np.sum(counts[nonzero]) / (np.sum(nonzero) * counts[nonzero])
        self.class_counts = counts
        return weights

    def _load_file_py(self, file_path_str):
        if hasattr(file_path_str, "numpy"):
            file_path_str = file_path_str.numpy()
        if isinstance(file_path_str, bytes):
            file_path_str = file_path_str.decode("utf-8")
        return self.data_loader.load_file(file_path_str).astype(np.float32)

    def _tf_load_full_audio(self, file_path: tf.Tensor, label: tf.Tensor = None):
        audio = tf.py_function(self._load_file_py, [file_path], tf.float32)
        if label is not None:
            return audio, label
        return audio

    def _with_deterministic_options(self, dataset):
        options = tf.data.Options()
        options.deterministic = self.deterministic
        options.experimental_deterministic = self.deterministic
        return dataset.with_options(options)

    def _create_pipeline(
        self,
        file_paths,
        labels,
        augment,
        batch_size,
        shuffle,
        one_hot,
    ):
        cache_root = (
            os.environ.get("WINGBEAT_CACHE_DIR")
            or self.cache_cfg.get("root")
            or os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../../dataset/.tf_cache")
            )
        )
        relative_paths = []
        sample_ids = []
        sample_hashes = []
        dataset_root = Path(self.dataset_dir).resolve()
        for path, label in zip(file_paths, labels):
            resolved = Path(str(path)).resolve()
            try:
                relative = str(resolved.relative_to(dataset_root).as_posix())
            except ValueError:
                relative = str(resolved.as_posix())
            relative_paths.append(f"{relative}|label={int(label)}")
            sample_ids.append(relative)
            import hashlib
            h = int.from_bytes(
                hashlib.sha256(relative.encode("utf-8")).digest()[:8],
                "big",
                signed=True,
            )
            sample_hashes.append(h)

        preprocessing = {
            "sample_rate": self.sample_rate,
            "segment_length": self.segment_length,
            "file_extensions": self.data_loader.file_exts,
            "augment": self.augmentor.aug_cfg.model_dump(mode="json") if hasattr(self.augmentor.aug_cfg, "model_dump") else self.augmentor.cfg,
            "stage": "audio" if augment else "segments",
        }
        cache_key = stable_cache_key(
            relative_paths,
            preprocessing,
            manifest_sha256=self.cache_cfg.get("manifest_sha256", ""),
            schema_version=self.cache_cfg.get(
                "schema_version", CACHE_SCHEMA_VERSION
            ),
        )
        cache_prefix = "train" if augment else "val_test"
        cache_file = os.path.join(
            cache_root,
            f"{cache_prefix}_{cache_key}",
        )
        cache_enabled = bool(self.cache_cfg.get("enabled", True))

        dataset = tf.data.Dataset.from_tensor_slices(
            (file_paths, labels, sample_ids, sample_hashes)
        )
        dataset = self._with_deterministic_options(dataset)

        # Map to load audio
        dataset = dataset.map(
            lambda path, label, sample_id, sample_hash: (
                self._tf_load_full_audio(path),
                label,
                sample_id,
                sample_hash,
            ),
            num_parallel_calls=self.pure_parallel_calls,
            deterministic=self.deterministic,
        )

        # Cache audio for training dataset (loaded audio + label + sample_id + sample_hash)
        if augment and cache_enabled:
            dataset = materialize_tensorflow_cache(dataset, cache_file)

        if shuffle:
            shuffle_seed = self.seed if self.deterministic else None
            dataset = dataset.shuffle(
                buffer_size=max(
                    1,
                    10000,
                ),
                seed=shuffle_seed,
                reshuffle_each_iteration=True,
            )

        # Zip or assign seeds
        global_seed = self.seed if self.seed is not None else 42
        if augment:
            seed_ds = tf.data.Dataset.random(
                seed=global_seed, rerandomize_each_iteration=True
            )
            dataset = tf.data.Dataset.zip((dataset, seed_ds))
            dataset = dataset.map(
                lambda item, rnd: (
                    item[0],
                    item[1],
                    item[2],
                    item[3],
                    rnd,
                ),
                num_parallel_calls=self.pure_parallel_calls,
                deterministic=self.deterministic,
            )
        else:
            dataset = dataset.map(
                lambda audio, label, sample_id, sample_hash: (
                    audio,
                    label,
                    sample_id,
                    sample_hash,
                    tf.constant(0, dtype=tf.int64),
                ),
                num_parallel_calls=self.pure_parallel_calls,
                deterministic=self.deterministic,
            )

        # Interleave segments
        segment_parallel_calls = (
            self.random_parallel_calls if augment else self.pure_parallel_calls
        )
        dataset = dataset.interleave(
            lambda audio, label, sample_id, sample_hash, rnd: self.augmentor.create_segments(
                audio,
                label,
                seed=derive_sample_seed(sample_hash, global_seed, rnd, stage_id=1),
                training=augment,
                sample_id=sample_id,
            ),
            num_parallel_calls=segment_parallel_calls,
            deterministic=self.deterministic,
        )

        noise_probability = self.augmentor.noise_cfg.p

        use_noise = augment and noise_probability > 0.0

        post_parallel_calls = (
            self.random_parallel_calls
            if augment
            else self.pure_parallel_calls
        )

        if use_noise:
            valid_noise_dirs = [d for d in (self.noise_dirs or []) if Path(d).exists()]
            if valid_noise_dirs:
                noise_ds = self.augmentor.build_noise_dataset(
                    valid_noise_dirs,
                    load_fn=lambda path: self._tf_load_full_audio(path),
                )

                if noise_ds is None:
                    raise ValueError(
                        "noise_overlay.p is greater than zero, but no "
                        ".wav or .npy noise files were found."
                    )

                dataset = tf.data.Dataset.zip((dataset, noise_ds))
                dataset = dataset.map(
                    lambda item, noise: self.augmentor.apply_post_processing(
                        item[0],
                        item[1],
                        item[2],
                        noise=noise,
                        augment=True,
                    ),
                    num_parallel_calls=post_parallel_calls,
                    deterministic=self.deterministic,
                )
            else:
                use_noise = False

        if not use_noise:
            dataset = dataset.map(
                lambda frame, label, seed, sample_id: (
                    self.augmentor.apply_post_processing(
                        frame,
                        label,
                        seed,
                        augment=augment,
                    )
                ),
                num_parallel_calls=post_parallel_calls,
                deterministic=self.deterministic,
            )

        if shuffle:
            shuffle_seed = self.seed if self.deterministic else None
            dataset = dataset.shuffle(
                buffer_size=10000,
                seed=shuffle_seed,
                reshuffle_each_iteration=True,
            )

        if one_hot:
            dataset = dataset.map(
                lambda audio, label: (
                    tf.expand_dims(audio, -1),
                    tf.one_hot(
                        tf.cast(label, tf.int32),
                        self.data_loader.num_classes,
                    ),
                ),
                num_parallel_calls=self.pure_parallel_calls,
                deterministic=self.deterministic,
            )
        else:
            dataset = dataset.map(
                lambda audio, label: (
                    tf.expand_dims(audio, -1),
                    label,
                ),
                num_parallel_calls=self.pure_parallel_calls,
                deterministic=self.deterministic,
            )

        # Cache validation and test datasets at the end
        if not augment and cache_enabled:
            dataset = materialize_tensorflow_cache(dataset, cache_file)

        dataset = dataset.batch(batch_size, drop_remainder=augment)

        mixup_cfg = self.augmentor.aug_cfg.mixup
        mixup_probability = mixup_cfg.p

        if augment and mixup_probability > 0.0:
            mixup_seed_ds = tf.data.Dataset.random(seed=self.seed + 888 if self.seed is not None else None, rerandomize_each_iteration=True)
            dataset = tf.data.Dataset.zip((dataset, mixup_seed_ds))
            dataset = dataset.map(
                lambda audio_label, seed: self._apply_targeted_mixup(
                    audio_label[0],
                    audio_label[1],
                    mixup_cfg,
                    seed,
                ),
                num_parallel_calls=self.random_parallel_calls,
                deterministic=self.deterministic,
            )

        dataset = dataset.prefetch(self.prefetch_buffer)

        def _set_static_shape(audio, label):
            audio.set_shape([None, self.segment_length, 1])
            return audio, label

        dataset = dataset.map(
            _set_static_shape,
            num_parallel_calls=self.pure_parallel_calls,
            deterministic=self.deterministic,
        )

        return self._with_deterministic_options(dataset)

    def _get_allowed_mixup_tensor(self, mixup_cfg):
        if not hasattr(self, "_allowed_mixup_tensor") or self._allowed_mixup_tensor is None:
            num_classes = self.data_loader.num_classes
            allowed = np.zeros((num_classes, num_classes), dtype=bool)
            mappings = (
                getattr(mixup_cfg, "class_mappings", None)
                if not isinstance(mixup_cfg, dict)
                else mixup_cfg.get("class_mappings", {})
            )
            has_mappings = bool(mappings)
            if has_mappings:
                for src_class_str, allowed_list in mappings.items():
                    src_class = int(src_class_str)
                    for dst_class in allowed_list:
                        allowed[src_class, int(dst_class)] = True
                        allowed[int(dst_class), src_class] = True
            else:
                allowed = np.ones((num_classes, num_classes), dtype=bool)

            self._allowed_mixup_tensor = tf.constant(allowed, dtype=tf.bool)
            self._has_mixup_mappings = has_mappings

        return self._allowed_mixup_tensor, self._has_mixup_mappings

    @tf.function
    def _apply_targeted_mixup(self, x, y, mixup_cfg, seed):
        p = float(mixup_cfg.p) if hasattr(mixup_cfg, "p") else float(mixup_cfg.get("p", 1.0))
        alpha = float(mixup_cfg.alpha) if hasattr(mixup_cfg, "alpha") else float(mixup_cfg.get("alpha", 0.2))

        batch_size = tf.shape(x)[0]

        shuffle_seed = tf.stack([seed, tf.constant(501, dtype=tf.int64)])
        do_mix_seed = tf.stack([seed, tf.constant(502, dtype=tf.int64)])
        gamma_seed = tf.stack([seed, tf.constant(503, dtype=tf.int64)])

        indices = tf.random.experimental.stateless_shuffle(tf.range(batch_size), seed=shuffle_seed)
        x2 = tf.gather(x, indices)
        y2 = tf.gather(y, indices)

        label1 = tf.argmax(y, axis=1, output_type=tf.int32)
        label2 = tf.argmax(y2, axis=1, output_type=tf.int32)

        allowed_tensor, has_mappings = self._get_allowed_mixup_tensor(mixup_cfg)

        pair_indices = tf.stack([label1, label2], axis=1)
        is_mapped_pair = tf.gather_nd(allowed_tensor, pair_indices)

        outside_prob_scale = float(getattr(mixup_cfg, "outside_prob_scale", 0.2)) if not isinstance(mixup_cfg, dict) else float(mixup_cfg.get("outside_prob_scale", 0.2))
        if has_mappings:
            prob_scale = tf.where(is_mapped_pair, tf.ones([batch_size]), tf.fill([batch_size], outside_prob_scale))
        else:
            prob_scale = tf.ones([batch_size])

        mix_prob = p * prob_scale
        do_mix = tf.random.stateless_uniform([batch_size], seed=do_mix_seed) < mix_prob

        gamma1_seed = tf.stack([gamma_seed[0], tf.constant(1, dtype=tf.int64)])
        gamma2_seed = tf.stack([gamma_seed[0], tf.constant(2, dtype=tf.int64)])
        gamma1 = tf.random.stateless_gamma([batch_size], seed=gamma1_seed, alpha=alpha)
        gamma2 = tf.random.stateless_gamma([batch_size], seed=gamma2_seed, alpha=alpha)
        lam = gamma1 / (gamma1 + gamma2 + 1e-8)

        lam = tf.where(do_mix, lam, tf.ones_like(lam))

        lam_x = tf.reshape(lam, [-1, 1, 1])
        lam_y = tf.reshape(lam, [-1])

        x_mixed = lam_x * x + (1.0 - lam_x) * x2
        y_mixed = tf.expand_dims(lam_y, -1) * y + tf.expand_dims(1.0 - lam_y, -1) * y2

        return x_mixed, y_mixed

    def _require_files(self, paths, split_name, directory):
        if len(paths) == 0:
            raise ValueError(
                f"No {split_name} files found in {directory}. "
                "Check dataset.dataset_dir/val_dir/test_dir and supported extensions."
            )

    def _split_paths(self, paths, labels, test_size, split_name):
        # Keep runtime datasets and standalone split tooling on one policy.
        return split_paths(
            np.asarray(paths),
            np.asarray(labels),
            test_size=test_size,
            split_name=split_name,
            seed=self.seed,
        )

    def build(self, split=[0.8, 0.1, 0.1], batch_size=32,
            shuffle=True, one_hot=True):

        train_paths, train_labels = self.data_loader.gather_files()
        self._require_files(train_paths, "training", self.dataset_dir)

        if self.val_dir:
            if self.test_dir and self.test_dir != self.val_dir:
                val_paths, val_labels = self.data_loader.gather_files(self.val_dir)
                test_paths, test_labels = self.data_loader.gather_files(self.test_dir)
                self._require_files(val_paths, "validation", self.val_dir)
                self._require_files(test_paths, "test", self.test_dir)
            else:
                eval_paths, eval_labels = self.data_loader.gather_files(self.val_dir)
                self._require_files(eval_paths, "evaluation", self.val_dir)

                val_test_sum = split[1] + split[2]
                val_ratio = split[1] / val_test_sum if val_test_sum > 0 else 0.5

                val_paths, test_paths, val_labels, test_labels = self._split_paths(
                    eval_paths, eval_labels,
                    test_size=1.0 - val_ratio,
                    split_name="validation/test"
                )
        else:
            # Three-way stratified split
            val_test_size = split[1] + split[2]
            train_paths, eval_paths, train_labels, eval_labels = self._split_paths(
                train_paths, train_labels,
                test_size=val_test_size,
                split_name="training/evaluation"
            )

            val_ratio = split[1] / val_test_size if val_test_size > 0 else 0.5
            val_paths, test_paths, val_labels, test_labels = self._split_paths(
                eval_paths, eval_labels,
                test_size=1.0 - val_ratio,
                split_name="validation/test"
            )

        # Store split attributes
        self.train_paths = train_paths
        self.train_labels = train_labels
        self.val_paths = val_paths
        self.val_labels = val_labels
        self.test_paths = test_paths
        self.test_labels = test_labels

        self.class_weights = self._compute_balanced_class_weights(train_paths, train_labels)

        train_ds = self._create_pipeline(
            train_paths, train_labels,
            augment=True, batch_size=batch_size,
            shuffle=shuffle, one_hot=one_hot
        )

        val_ds = self._create_pipeline(
            val_paths, val_labels,
            augment=False, batch_size=batch_size,
            shuffle=False, one_hot=one_hot
        )

        test_ds = self._create_pipeline(
            test_paths, test_labels,
            augment=False, batch_size=batch_size,
            shuffle=False, one_hot=one_hot
        )

        return train_ds, val_ds, test_ds


def build_datasets(
    dataset_dir: str | Path,
    config: Mapping[str, object],
    *,
    val_dir: str | Path | None = None,
    test_dir: str | Path | None = None,
    noise_dirs: list[str] | None = None,
    return_builder: bool = False,
):
    """Build train, validation and test tf.data.Dataset objects.

    Preserves all pipeline behaviour from SupervisedDataset:
      - batching, shuffle, caching, prefetch, parallel-map
      - deterministic mode, label encoding (one-hot)
      - augmentation placement (training only)
      - training / validation / test differences

    Args:
        dataset_dir: Root directory containing per-class subdirectories.
        config: Resolved configuration mapping.  Expected keys:
            train.seed, train.deterministic, data.classes, data.labels,
            data.segment_length, data.sample_rate, data.split,
            train.batch_size, augmentation (optional).
        val_dir: Optional separate validation directory.
        test_dir: Optional separate test directory.
        noise_dirs: Optional list of noise audio directories.
        return_builder: Include the configured builder before the datasets.

    Returns:
        Dataset tuple, optionally prefixed by the SupervisedDataset builder.
    """
    from wingbeat_ml.config.schema import validate_config

    app_cfg = validate_config(config)
    seed = app_cfg.reproducibility.seed
    deterministic = app_cfg.reproducibility.deterministic_data
    batch_size = app_cfg.train.batch_size
    shuffle = app_cfg.train.shuffle
    classes = app_cfg.classes
    labels_dict = app_cfg.labels
    segment_length = app_cfg.audio.segment_length
    sample_rate = app_cfg.audio.sample_rate
    split = [
        app_cfg.dataset.split_ratios.train,
        app_cfg.dataset.split_ratios.val,
        app_cfg.dataset.split_ratios.test,
    ]
    if val_dir is None:
        val_dir = app_cfg.dataset.val_dir
    if test_dir is None:
        test_dir = app_cfg.dataset.test_dir
    if noise_dirs is None:
        noise_dirs = list(app_cfg.augment.noise_banks)
    augment_cfg = app_cfg.augment
    cache_cfg = {
        **app_cfg.cache.model_dump(),
        "manifest_sha256": app_cfg.dataset.manifest_sha256 or "",
    }
    nomos_index = None
    if classes:
        for i, name in enumerate(classes):
            compact = "".join(c for c in name.casefold() if c.isalnum())
            if compact == "nomos":
                nomos_index = i
                break

    ds = SupervisedDataset(
        dataset_dir=str(dataset_dir),
        val_dir=str(val_dir) if val_dir else None,
        test_dir=str(test_dir) if test_dir else None,
        sample_rate=sample_rate,
        segment_length=segment_length,
        classes=classes,
        noise_dirs=noise_dirs,
        augment_cfg=augment_cfg,
        seed=seed,
        deterministic=deterministic,
        nomos_index=nomos_index,
        labels_dict=labels_dict,
        cache_cfg=cache_cfg,
    )

    datasets = ds.build(
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        one_hot=True,
    )
    if return_builder:
        return (ds, *datasets)
    return datasets

# Compatibility name retained for the incremental migration.
SupervisedDatasetWrapper = SupervisedDataset

__all__ = [
    "SupervisedDataset",
    "SupervisedDatasetWrapper",
    "build_datasets",
]
