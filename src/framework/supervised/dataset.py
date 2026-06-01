import tensorflow as tf
from src.framework.helper.data_loader import DataLoader
from src.framework.helper.augment import AudioAugmentor
import numpy as np
import os
from pathlib import Path

class SupervisedDataset:
    def __init__(self, dataset_dir: str, val_dir: str = None, test_dir: str = None, 
                 sample_rate: int = 8000, segment_length: int = 2400, 
                 classes: list = None, noise_dirs: list = None, augment_cfg: dict = None):
        self.dataset_dir = dataset_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.data_loader = DataLoader(dataset_dir, sample_rate, classes)
        self.augmentor = AudioAugmentor(segment_length, augment_cfg)
        self.noise_dirs = noise_dirs
        self.class_weights = None

    def _compute_balanced_class_weights(self, labels):
        counts = np.bincount(labels, minlength=self.data_loader.num_classes).astype(np.float32)
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

    def _create_pipeline(self, file_paths, labels, augment, batch_size, shuffle, one_hot, step_ratio=0.5):
        # 1. Main Mosquito Dataset
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        dataset = dataset.map(self._tf_load_full_audio, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.cache() # Cache full resampled files

        # 2. Exhaustive Slicing (Encapsulated in Augmentor)
        dataset = dataset.interleave(
            lambda x, y: self.augmentor.create_segments(x, y, step_ratio=step_ratio, training=augment),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        ) 

        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)

        # 4. Noise Augmentation & DC Removal
        if augment and self.noise_dirs:
            # Pass our load method to the augmentor's noise builder
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
                    num_parallel_calls=tf.data.AUTOTUNE
                )
            else:
                dataset = dataset.map(
                    lambda x, y: self.augmentor.apply_post_processing(x, y, augment=False),
                    num_parallel_calls=tf.data.AUTOTUNE
                )
        else:
            # Val/Test pipeline: No noise, but still DC Removal
            dataset = dataset.map(
                lambda x, y: self.augmentor.apply_post_processing(x, y, augment=False),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # 5. Encoding & Batching
        if one_hot:
            dataset = dataset.map(
                lambda x, y: (x, tf.one_hot(y, self.data_loader.num_classes)), 
                num_parallel_calls=tf.data.AUTOTUNE
            )

        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def build(self, split: list = [0.8, 0.1, 0.1], batch_size: int = 32, shuffle: bool = True, one_hot: bool = True, step_ratio: float = 0.5):
        # 1. Gather Training Files
        train_paths, train_labels = self.data_loader.gather_files()
        
        # 2. Determine Evaluation Sets
        if self.val_dir:
            # Case: Dedicated Evaluation Source
            if self.test_dir and self.test_dir != self.val_dir:
                # User provided two different directories for val and test
                val_paths, val_labels = self.data_loader.gather_files(self.val_dir)
                test_paths, test_labels = self.data_loader.gather_files(self.test_dir)
                print(f"Loaded dedicated sets: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")
            else:
                # User provided one evaluation directory (or val_dir == test_dir)
                # We split this directory into Val and Test
                eval_paths, eval_labels = self.data_loader.gather_files(self.val_dir)
                indices = np.arange(len(eval_paths))
                np.random.shuffle(indices)
                eval_paths, eval_labels = eval_paths[indices], eval_labels[indices]
                
                # Split evaluation directory (e.g., 50/50 if not specified, otherwise proportional to val/test split)
                val_ratio = split[1] / (split[1] + split[2]) if (split[1] + split[2]) > 0 else 0.5
                val_end = int(val_ratio * len(eval_paths))
                
                val_paths, val_labels = eval_paths[:val_end], eval_labels[:val_end]
                test_paths, test_labels = eval_paths[val_end:], eval_labels[val_end:]
                print(f"Loaded Eval from {self.val_dir}: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")
        else:
            # Case: Random Split from Training Source
            indices = np.arange(len(train_paths))
            np.random.shuffle(indices)
            all_paths, all_labels = train_paths[indices], train_labels[indices]

            train_end = int(split[0] * len(all_paths))
            val_end = train_end + int(split[1] * len(all_paths))

            train_paths, train_labels = all_paths[:train_end], all_labels[:train_end]
            val_paths, val_labels = all_paths[train_end:val_end], all_labels[train_end:val_end]
            test_paths, test_labels = all_paths[val_end:], all_labels[val_end:]
            print(f"Using random split: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")

        self.class_weights = self._compute_balanced_class_weights(train_labels)
        print(f"Training class counts: {np.bincount(train_labels, minlength=self.data_loader.num_classes).tolist()}")
        print(f"Balanced class weights: {np.round(self.class_weights, 3).tolist()}")

        # 3. Create Pipelines
        train_ds = self._create_pipeline(train_paths, train_labels, 
                                       augment=True, batch_size=batch_size, shuffle=shuffle, one_hot=one_hot, step_ratio=step_ratio)
        
        val_ds = self._create_pipeline(val_paths, val_labels, 
                                     augment=False, batch_size=batch_size, shuffle=False, one_hot=one_hot, step_ratio=1.0) 
        
        test_ds = self._create_pipeline(test_paths, test_labels, 
                                      augment=False, batch_size=batch_size, shuffle=False, one_hot=one_hot, step_ratio=1.0)

        return train_ds, val_ds, test_ds
