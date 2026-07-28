"""TensorFlow SSL Dataset pipeline for Indoor (Source) and Outdoor (Target) domains.

Supports per-class sample sub-sampling (50-450 samples/class for training, 50 for val/test).
"""

import numpy as np
import tensorflow as tf
from typing import Any, Dict, Optional, Tuple
from wingbeat_ml.data.dataset import SupervisedDataset


def filter_paths_by_sample_limit(
    paths: np.ndarray,
    labels: np.ndarray,
    samples_per_class: Optional[int],
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sub-sample file paths and labels reproducibly so that at most
    `samples_per_class` samples are selected per class.
    """
    if samples_per_class is None or samples_per_class <= 0:
        return paths, labels

    paths = np.asarray(paths)
    labels = np.asarray(labels)

    rng = np.random.RandomState(seed)
    unique_classes = np.unique(labels)
    selected_indices = []

    for c in unique_classes:
        c_indices = np.where(labels == c)[0]
        if len(c_indices) > samples_per_class:
            c_selected = rng.choice(c_indices, size=samples_per_class, replace=False)
        else:
            c_selected = c_indices
        selected_indices.extend(c_selected)

    selected_indices = np.sort(selected_indices)
    return paths[selected_indices], labels[selected_indices]


class SSLDatasetBuilder:
    """
    Builds TensorFlow dataset pipelines for SSL:
      - Source Labeled Dataset (Indoor)
      - Target Unlabeled Dataset (Outdoor)
      - Dual-Domain Evaluation Datasets (Indoor Val/Test & Outdoor Val/Test)
    """

    def __init__(self, config: Any):
        from wingbeat_ml.config.schema import validate_config

        self.config = validate_config(config)
        self.ssl_cfg = self.config.ssl

        self.source_dir = self.ssl_cfg.source_dir
        self.target_dir = self.ssl_cfg.target_dir

        self.train_limit = self.ssl_cfg.train_samples_per_class
        self.val_limit = self.ssl_cfg.val_samples_per_class
        self.test_limit = self.ssl_cfg.test_samples_per_class

        self.batch_size = self.config.train.batch_size
        self.seed = self.config.reproducibility.seed

        # Initialize underlying SupervisedDataset objects
        self.source_builder = SupervisedDataset(
            dataset_dir=self.source_dir,
            sample_rate=self.config.audio.sample_rate,
            segment_length=self.config.audio.segment_length,
            classes=self.config.classes,
            augment_cfg=self.config.augment,
            seed=self.seed,
            deterministic=self.config.reproducibility.deterministic_data,
        )

        self.target_builder = SupervisedDataset(
            dataset_dir=self.target_dir,
            sample_rate=self.config.audio.sample_rate,
            segment_length=self.config.audio.segment_length,
            classes=self.config.classes,
            augment_cfg=self.config.augment,
            seed=self.seed + 1,
            deterministic=self.config.reproducibility.deterministic_data,
        )

    def build_datasets(self) -> Dict[str, Any]:
        """
        Build and sub-sample all SSL TensorFlow datasets.
        """
        # Gather source & target files
        src_paths, src_labels = self.source_builder.data_loader.gather_files()
        tgt_paths, tgt_labels = self.target_builder.data_loader.gather_files()

        # Split source into train, val, test (80/10/10)
        src_tr_p, src_ev_p, src_tr_l, src_ev_l = self.source_builder._split_paths(
            src_paths, src_labels, test_size=0.2, split_name="source_train_eval"
        )
        src_val_p, src_ts_p, src_val_l, src_ts_l = self.source_builder._split_paths(
            src_ev_p, src_ev_l, test_size=0.5, split_name="source_val_test"
        )

        # Split target into train, val, test (80/10/10)
        tgt_tr_p, tgt_ev_p, tgt_tr_l, tgt_ev_l = self.target_builder._split_paths(
            tgt_paths, tgt_labels, test_size=0.2, split_name="target_train_eval"
        )
        tgt_val_p, tgt_ts_p, tgt_val_l, tgt_ts_l = self.target_builder._split_paths(
            tgt_ev_p, tgt_ev_l, test_size=0.5, split_name="target_val_test"
        )

        # Apply per-class sample limits
        src_tr_p, src_tr_l = filter_paths_by_sample_limit(src_tr_p, src_tr_l, self.train_limit, seed=self.seed)
        src_val_p, src_val_l = filter_paths_by_sample_limit(src_val_p, src_val_l, self.val_limit, seed=self.seed)
        src_ts_p, src_ts_l = filter_paths_by_sample_limit(src_ts_p, src_ts_l, self.test_limit, seed=self.seed)

        tgt_tr_p, tgt_tr_l = filter_paths_by_sample_limit(tgt_tr_p, tgt_tr_l, self.train_limit, seed=self.seed + 2)
        tgt_val_p, tgt_val_l = filter_paths_by_sample_limit(tgt_val_p, tgt_val_l, self.val_limit, seed=self.seed + 2)
        tgt_ts_p, tgt_ts_l = filter_paths_by_sample_limit(tgt_ts_p, tgt_ts_l, self.test_limit, seed=self.seed + 2)

        # Create TF pipelines
        source_train_ds = self.source_builder._create_pipeline(
            src_tr_p, src_tr_l, augment=True, batch_size=self.batch_size, shuffle=True, one_hot=True
        )

        # Target unlabeled pipeline produces weak and strong augmented pairs
        target_weak_ds = self.target_builder._create_pipeline(
            tgt_tr_p, tgt_tr_l, augment=False, batch_size=self.batch_size, shuffle=True, one_hot=True
        )
        target_strong_ds = self.target_builder._create_pipeline(
            tgt_tr_p, tgt_tr_l, augment=True, batch_size=self.batch_size, shuffle=True, one_hot=True
        )

        target_unlabeled_ds = tf.data.Dataset.zip((target_weak_ds, target_strong_ds)).map(
            lambda w_batch, s_batch: (w_batch[0], s_batch[0]),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Zip source labeled and target unlabeled
        train_zipped_ds = tf.data.Dataset.zip((source_train_ds, target_unlabeled_ds))

        # Evaluation pipelines for both domains
        source_val_ds = self.source_builder._create_pipeline(
            src_val_p, src_val_l, augment=False, batch_size=self.batch_size, shuffle=False, one_hot=True
        )
        source_test_ds = self.source_builder._create_pipeline(
            src_ts_p, src_ts_l, augment=False, batch_size=self.batch_size, shuffle=False, one_hot=True
        )

        target_val_ds = self.target_builder._create_pipeline(
            tgt_val_p, tgt_val_l, augment=False, batch_size=self.batch_size, shuffle=False, one_hot=True
        )
        target_test_ds = self.target_builder._create_pipeline(
            tgt_ts_p, tgt_ts_l, augment=False, batch_size=self.batch_size, shuffle=False, one_hot=True
        )

        return {
            "train_ds": train_zipped_ds,
            "source_val_ds": source_val_ds,
            "source_test_ds": source_test_ds,
            "target_val_ds": target_val_ds,
            "target_test_ds": target_test_ds,
            "counts": {
                "source_train": len(src_tr_p),
                "source_val": len(src_val_p),
                "source_test": len(src_ts_p),
                "target_train": len(tgt_tr_p),
                "target_val": len(tgt_val_p),
                "target_test": len(tgt_ts_p),
            },
        }


def build_ssl_datasets(config: Any) -> Dict[str, Any]:
    """Helper entrypoint to construct sample-limited SSL datasets."""
    builder = SSLDatasetBuilder(config)
    return builder.build_datasets()


__all__ = [
    "filter_paths_by_sample_limit",
    "SSLDatasetBuilder",
    "build_ssl_datasets",
]
