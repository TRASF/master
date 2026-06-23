import tensorflow as tf
from src.framework.helper.data_loader import DataLoader
from src.framework.helper.augment import AudioAugmentor
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

class SupervisedDataset:
    def __init__(self, dataset_dir: str, val_dir: str = None, test_dir: str = None,
                 sample_rate: int = 8000, segment_length: int = 2400,
                 classes: list = None, noise_dirs: list = None,
                 augment_cfg: dict = None, seed: int = 42,
                 deterministic: bool = False, nomos_index: int = None):
        self.dataset_dir = dataset_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.data_loader = DataLoader(dataset_dir, sample_rate, classes)
        self.noise_dirs = noise_dirs
        self.seed = seed
        self.deterministic = deterministic
        self.parallel_calls = 1 if deterministic else tf.data.AUTOTUNE
        self.prefetch_buffer = 1 if deterministic else tf.data.AUTOTUNE

        # Use provided nomos_index or try to find it
        self.nomos_index = nomos_index
        if self.nomos_index is None and classes:
            for i, name in enumerate(classes):
                if "No.Mos" in name or "Nomos" in name:
                    self.nomos_index = i
                    break

        self.augmentor = AudioAugmentor(
            segment_length,
            augment_cfg,
            seed=seed,
            deterministic=deterministic,
            nomos_index=self.nomos_index
        )

        # Attributes to store splits for debugging and metrics
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
                
            # Expected number of segments
            num_segments = int(np.ceil(num_samples / avg_step))
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

    def _create_pipeline(self, file_paths, labels, augment, batch_size, shuffle, one_hot):
        # 1. Base Dataset (File Paths)
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        dataset = self._with_deterministic_options(dataset)

        if shuffle:
            shuffle_seed = self.seed if self.deterministic else None
            dataset = dataset.shuffle(
                buffer_size=(int(np.ceil(len(file_paths) / 4))),
                seed=shuffle_seed,
                reshuffle_each_iteration=True,
            )

        # 2. Load and Cache FULL audio (resampled once)
        dataset = dataset.map(
            self._tf_load_full_audio,
            num_parallel_calls=self.parallel_calls,
            deterministic=self.deterministic,
        )
        dataset = dataset.cache()

        # 3. Slicing Strategy
        if augment:
            # Training: Dynamic exhaustive framing with random offset and random step
            dataset = dataset.interleave(
                lambda x, y: self.augmentor.create_segments(x, y, training=True),
                num_parallel_calls=self.parallel_calls,
                deterministic=self.deterministic
            )
        else:
            # Val/Test: Deterministic exhaustive slicing with specified overlap
            dataset = dataset.interleave(
                lambda x, y: self.augmentor.create_segments(x, y, training=False),
                num_parallel_calls=self.parallel_calls,
                deterministic=self.deterministic
            )

        # 4. Noise Augmentation & Post-processing
        if augment and self.noise_dirs:
            noise_ds = self.augmentor.build_noise_dataset(
                self.noise_dirs,
                load_fn=lambda p: self._tf_load_full_audio(p)
            )

            if noise_ds:
                dataset = tf.data.Dataset.zip((dataset, noise_ds))
                dataset = dataset.map(
                    lambda signal_label, noise: self.augmentor.apply_post_processing(
                        signal_label[0], signal_label[1], noise=noise, augment=True
                    ),
                    num_parallel_calls=self.parallel_calls,
                    deterministic=self.deterministic,
                )
            else:
                dataset = dataset.map(
                    lambda x, y: self.augmentor.apply_post_processing(x, y, augment=True),
                    num_parallel_calls=self.parallel_calls,
                    deterministic=self.deterministic,
                )
        else:
            # Post-processing (DC Removal + Range Clipping)
            dataset = dataset.map(
                lambda x, y: self.augmentor.apply_post_processing(x, y, augment=augment),
                num_parallel_calls=self.parallel_calls,
                deterministic=self.deterministic,
            )

        if shuffle:
            shuffle_seed = self.seed if self.deterministic else None
            dataset = dataset.shuffle(
                buffer_size=10000,
                seed=shuffle_seed,
                reshuffle_each_iteration=True,
            )

        # 5. Encoding & Batching (Adding channel dimension here)
        if one_hot:
            dataset = dataset.map(
                lambda x, y: (tf.expand_dims(x, -1), tf.one_hot(y, self.data_loader.num_classes)),
                num_parallel_calls=self.parallel_calls,
                deterministic=self.deterministic,
            )

        dataset = dataset.batch(batch_size).prefetch(self.prefetch_buffer)
        return self._with_deterministic_options(dataset)

    def _require_files(self, paths, split_name, directory):
        if len(paths) == 0:
            raise ValueError(
                f"No {split_name} files found in {directory}. "
                "Check dataset.dataset_dir/val_dir/test_dir and supported extensions."
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

                val_paths, test_paths, val_labels, test_labels = train_test_split(
                    eval_paths, eval_labels,
                    test_size=1.0 - val_ratio,
                    stratify=eval_labels,
                    random_state=self.seed
                )
        else:
            # Three-way stratified split
            val_test_size = split[1] + split[2]
            train_paths, eval_paths, train_labels, eval_labels = train_test_split(
                train_paths, train_labels,
                test_size=val_test_size,
                stratify=train_labels,
                random_state=self.seed
            )

            val_ratio = split[1] / val_test_size if val_test_size > 0 else 0.5
            val_paths, test_paths, val_labels, test_labels = train_test_split(
                eval_paths, eval_labels,
                test_size=1.0 - val_ratio,
                stratify=eval_labels,
                random_state=self.seed
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
