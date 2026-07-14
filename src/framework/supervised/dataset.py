import tensorflow as tf
from src.framework.helper.data_loader import DataLoader
from src.framework.helper.augment import AudioAugmentor
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

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
        augment_cfg: dict = None,
        seed: int = 42,
        deterministic: bool = False,
        nomos_index: int = None,
        labels_dict: dict = None,
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

        self.pure_parallel_calls = tf.data.AUTOTUNE

        self.random_parallel_calls = (
            1 if deterministic else tf.data.AUTOTUNE
        )

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

    def _compute_balanced_class_weights(self, file_paths, labels):
        import wave
        # Retrieve average overlap/step size
        raw_config = self.augmentor.cfg.get('config', {})
        overlap_cfg = raw_config.get('segment_overlap') or raw_config.get('overlap') or {}
        if isinstance(overlap_cfg, dict):
            overlap_range = overlap_cfg.get('train', [0.0, 0.8])
        elif isinstance(overlap_cfg, (list, tuple)) and len(overlap_cfg) == 2:
            overlap_range = overlap_cfg
        else:
            overlap_range = [0.0, 0.8]

        avg_overlap = np.mean(overlap_range)
        avg_step = int(self.segment_length * (1.0 - avg_overlap))
        avg_step = max(avg_step, 1)

        max_segments = raw_config.get('max_segments_per_file', 100)

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
        dataset = tf.data.Dataset.from_tensor_slices(
            (file_paths, labels)
        )
        dataset = self._with_deterministic_options(dataset)

        if shuffle:
            shuffle_seed = self.seed if self.deterministic else None
            dataset = dataset.shuffle(
                buffer_size=max(
                    1,
                    int(np.ceil(len(file_paths) / 4)),
                ),
                seed=shuffle_seed,
                reshuffle_each_iteration=True,
            )

        dataset = dataset.map(
            self._tf_load_full_audio,
            num_parallel_calls=self.pure_parallel_calls,
            deterministic=self.deterministic,
        )

        dataset = dataset.cache()

        if augment:
            segment_parallel_calls = self.random_parallel_calls

            dataset = dataset.interleave(
                lambda audio, label: self.augmentor.create_segments(
                    audio,
                    label,
                    training=True,
                ),
                num_parallel_calls=segment_parallel_calls,
                deterministic=self.deterministic,
            )
        else:
            segment_parallel_calls = self.pure_parallel_calls

            dataset = dataset.interleave(
                lambda audio, label: self.augmentor.create_segments(
                    audio,
                    label,
                    training=False,
                ),
                num_parallel_calls=segment_parallel_calls,
                deterministic=self.deterministic,
            )

        noise_probability = float(
            self.augmentor.noise_cfg.get("p", 0.0)
        )

        use_noise = augment and noise_probability > 0.0

        post_parallel_calls = (
            self.random_parallel_calls
            if augment
            else self.pure_parallel_calls
        )

        if use_noise:
            if not self.noise_dirs:
                raise ValueError(
                    "noise_overlay.p is greater than zero, but no "
                    "noise directories were configured."
                )

            noise_ds = self.augmentor.build_noise_dataset(
                self.noise_dirs,
                load_fn=lambda path: self._tf_load_full_audio(path),
            )

            if noise_ds is None:
                raise ValueError(
                    "noise_overlay.p is greater than zero, but no "
                    ".wav or .npy noise files were found."
                )

            dataset = tf.data.Dataset.zip((dataset, noise_ds))

            dataset = dataset.map(
                lambda signal_label, noise: (
                    self.augmentor.apply_post_processing(
                        signal_label[0],
                        signal_label[1],
                        noise=noise,
                        augment=True,
                    )
                ),
                num_parallel_calls=post_parallel_calls,
                deterministic=self.deterministic,
            )
        else:

            dataset = dataset.map(
                lambda audio, label: (
                    self.augmentor.apply_post_processing(
                        audio,
                        label,
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
                        label,
                        self.data_loader.num_classes,
                    ),
                ),
                num_parallel_calls=self.pure_parallel_calls,
                deterministic=self.deterministic,
            )

        dataset = dataset.batch(batch_size)

        mixup_cfg = self.augmentor.cfg.get("mixup", {})
        mixup_probability = float(mixup_cfg.get("p", 0.0))

        if augment and mixup_probability > 0.0:
            dataset = dataset.map(
                lambda audio, label: self._apply_targeted_mixup(
                    audio,
                    label,
                    mixup_cfg,
                ),
                num_parallel_calls=self.random_parallel_calls,
                deterministic=self.deterministic,
            )

        dataset = dataset.prefetch(self.prefetch_buffer)

        return self._with_deterministic_options(dataset)

    @tf.function
    def _apply_targeted_mixup(self, x, y, mixup_cfg):
        p = float(mixup_cfg.get('p', 1.0))
        alpha = float(mixup_cfg.get('alpha', 0.2))

        batch_size = tf.shape(x)[0]

        indices = tf.random.shuffle(tf.range(batch_size))
        x2 = tf.gather(x, indices)
        y2 = tf.gather(y, indices)

        label1 = tf.argmax(y, axis=1, output_type=tf.int32)
        label2 = tf.argmax(y2, axis=1, output_type=tf.int32)

        num_classes = self.data_loader.num_classes
        allowed = np.zeros((num_classes, num_classes), dtype=bool)

        mappings = mixup_cfg.get('class_mappings', {})
        if mappings:
            for src_class_str, allowed_list in mappings.items():
                src_class = int(src_class_str)
                for dst_class in allowed_list:
                    allowed[src_class, int(dst_class)] = True
                    allowed[int(dst_class), src_class] = True
        else:
            allowed = np.ones((num_classes, num_classes), dtype=bool)

        allowed_tensor = tf.constant(allowed, dtype=tf.bool)

        pair_indices = tf.stack([label1, label2], axis=1)
        is_mapped_pair = tf.gather_nd(allowed_tensor, pair_indices)

        outside_prob_scale = float(mixup_cfg.get(' ', 0.2))
        if mappings:
            prob_scale = tf.where(is_mapped_pair, tf.ones([batch_size]), tf.fill([batch_size], outside_prob_scale))
        else:
            prob_scale = tf.ones([batch_size])

        mix_prob = p * prob_scale
        do_mix = tf.random.uniform([batch_size]) < mix_prob

        gamma1 = tf.random.gamma([batch_size], alpha)
        gamma2 = tf.random.gamma([batch_size], alpha)
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
        try:
            return train_test_split(
                paths, labels,
                test_size=test_size,
                stratify=labels,
                random_state=self.seed
            )
        except ValueError:
            classes, counts = np.unique(labels, return_counts=True)
            sparse_classes = classes[counts < 2]
            if len(sparse_classes) == 0:
                raise

            print(
                f"Warning: cannot stratify {split_name} split because classes "
                f"{sparse_classes.tolist()} have fewer than 2 files. "
                "Falling back to a seeded non-stratified split."
            )
            return train_test_split(
                paths, labels,
                test_size=test_size,
                stratify=None,
                random_state=self.seed
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
