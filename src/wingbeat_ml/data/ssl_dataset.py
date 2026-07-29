"""TensorFlow SSL Dataset pipeline with decoupled domain roles and recording-aware sub-sampling.

Supports flexible domain assignment for labeled and unlabeled training pools:
  - Labeled domain choices: Indoor, Outdoor, or Both
  - Unlabeled domain choices: Indoor, Outdoor, or Both
  - Independent sample limits for labeled and unlabeled pools
  - Recording-aware sampling to prevent segment-level data leakage
"""

import os
import numpy as np
import tensorflow as tf
from typing import Any, Dict, List, Optional, Tuple, Set
from wingbeat_ml.data.dataset import SupervisedDataset


def get_recording_id(path: str) -> str:
    """Extract recording/session ID from file path to group segments from the same source."""
    filename = os.path.basename(path)
    stem = os.path.splitext(filename)[0]
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and (parts[1].isdigit() or parts[1].startswith("seg") or parts[1].startswith("sample")):
        return parts[0]
    return stem


def filter_paths_by_sample_limit(
    paths: np.ndarray,
    labels: np.ndarray,
    samples_per_class: Optional[int],
    seed: int = 42,
    group_by_recording: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sub-sample file paths and labels reproducibly so that at most
    `samples_per_class` samples are selected per class.
    Optionally groups by recording identity to preserve recording diversity.
    """
    if samples_per_class is None or samples_per_class <= 0:
        return np.asarray(paths), np.asarray(labels)

    paths = np.asarray(paths)
    labels = np.asarray(labels)
    if len(paths) == 0:
        return paths, labels

    rng = np.random.RandomState(seed)
    unique_classes = np.unique(labels)
    selected_indices = []

    for c in unique_classes:
        c_indices = np.where(labels == c)[0]
        if len(c_indices) <= samples_per_class:
            selected_indices.extend(c_indices)
            continue

        if group_by_recording:
            rec_map: Dict[str, List[int]] = {}
            for idx in c_indices:
                rec_id = get_recording_id(paths[idx])
                rec_map.setdefault(rec_id, []).append(idx)

            rec_ids = list(rec_map.keys())
            rng.shuffle(rec_ids)

            c_selected = []
            for r_id in rec_ids:
                group_indices = rec_map[r_id]
                if len(c_selected) + len(group_indices) <= samples_per_class or len(c_selected) == 0:
                    c_selected.extend(group_indices)
                else:
                    needed = samples_per_class - len(c_selected)
                    c_selected.extend(group_indices[:needed])
                if len(c_selected) >= samples_per_class:
                    break
            selected_indices.extend(c_selected[:samples_per_class])
        else:
            c_selected = rng.choice(c_indices, size=samples_per_class, replace=False)
            selected_indices.extend(c_selected)

    selected_indices = np.sort(selected_indices)
    return paths[selected_indices], labels[selected_indices]


class SSLDatasetBuilder:
    """
    Builds TensorFlow dataset pipelines for SSL with decoupled domain roles:
      - Flexible Labeled Dataset (Indoor, Outdoor, or Mixed)
      - Flexible Unlabeled Dataset (Indoor, Outdoor, or Mixed)
      - Dual-Domain Evaluation Datasets (Indoor & Outdoor Val/Test)
    """

    def __init__(self, config: Any):
        from wingbeat_ml.config.schema import validate_config

        self.config = validate_config(config)
        self.ssl_cfg = self.config.ssl

        self.source_dir = self.ssl_cfg.source_dir
        self.target_dir = self.ssl_cfg.target_dir

        self.labeled_domains = [d.lower() for d in (self.ssl_cfg.labeled_domains or ["indoor"])]
        self.unlabeled_domains = [d.lower() for d in (self.ssl_cfg.unlabeled_domains or ["outdoor"])]

        # Sample limits
        self.labeled_limit = (
            self.ssl_cfg.labeled_samples_per_class
            if self.ssl_cfg.labeled_samples_per_class is not None
            else self.ssl_cfg.train_samples_per_class
        )
        self.unlabeled_limit = self.ssl_cfg.unlabeled_samples_per_class
        self.val_limit = self.ssl_cfg.val_samples_per_class
        self.test_limit = self.ssl_cfg.test_samples_per_class

        self.exclude_labeled_from_unlabeled = self.ssl_cfg.exclude_labeled_from_unlabeled
        self.subset_seed = self.ssl_cfg.subset_seed
        self.batch_size = self.config.train.batch_size
        self.seed = self.config.reproducibility.seed

        # Initialize underlying SupervisedDataset objects for indoor and outdoor
        self.indoor_builder = SupervisedDataset(
            dataset_dir=self.source_dir,
            sample_rate=self.config.audio.sample_rate,
            segment_length=self.config.audio.segment_length,
            classes=self.config.classes,
            augment_cfg=self.config.augment,
            seed=self.seed,
            deterministic=self.config.reproducibility.deterministic_data,
        )

        self.outdoor_builder = SupervisedDataset(
            dataset_dir=self.target_dir,
            sample_rate=self.config.audio.sample_rate,
            segment_length=self.config.audio.segment_length,
            classes=self.config.classes,
            augment_cfg=self.config.augment,
            seed=self.seed + 1,
            deterministic=self.config.reproducibility.deterministic_data,
        )

    def _prepare_domain_splits(self, builder: SupervisedDataset, prefix: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        paths, labels = builder.data_loader.gather_files()
        tr_p, ev_p, tr_l, ev_l = builder._split_paths(paths, labels, test_size=0.2, split_name=f"{prefix}_train_eval")
        val_p, ts_p, val_l, ts_l = builder._split_paths(ev_p, ev_l, test_size=0.5, split_name=f"{prefix}_val_test")
        return {
            "train": (tr_p, tr_l),
            "val": (val_p, val_l),
            "test": (ts_p, ts_l),
        }

    def build_datasets(self) -> Dict[str, Any]:
        """Build and return dataset dict containing train, validation, test pipelines and manifests."""
        indoor_splits = self._prepare_domain_splits(self.indoor_builder, "indoor")
        outdoor_splits = self._prepare_domain_splits(self.outdoor_builder, "outdoor")

        domain_splits = {
            "indoor": (self.indoor_builder, indoor_splits),
            "outdoor": (self.outdoor_builder, outdoor_splits),
        }

        # 1. Build Labeled Training Pool
        labeled_paths_list: List[str] = []
        labeled_labels_list: List[int] = []

        for d_name in self.labeled_domains:
            if d_name in domain_splits:
                builder, splits = domain_splits[d_name]
                tr_p, tr_l = splits["train"]
                sub_p, sub_l = filter_paths_by_sample_limit(
                    tr_p, tr_l, self.labeled_limit, seed=self.subset_seed, group_by_recording=True
                )
                labeled_paths_list.extend(sub_p)
                labeled_labels_list.extend(sub_l)

        labeled_tr_paths = np.array(labeled_paths_list)
        labeled_tr_labels = np.array(labeled_labels_list)
        used_labeled_set: Set[str] = set(labeled_tr_paths)

        # 2. Build Unlabeled Training Pool
        unlabeled_paths_list: List[str] = []
        unlabeled_labels_list: List[int] = []

        for d_name in self.unlabeled_domains:
            if d_name in domain_splits:
                builder, splits = domain_splits[d_name]
                tr_p, tr_l = splits["train"]
                if self.exclude_labeled_from_unlabeled:
                    mask = np.array([p not in used_labeled_set for p in tr_p])
                    eligible_p = tr_p[mask]
                    eligible_l = tr_l[mask]
                else:
                    eligible_p, eligible_l = tr_p, tr_l

                if self.unlabeled_limit is not None:
                    sub_p, sub_l = filter_paths_by_sample_limit(
                        eligible_p, eligible_l, self.unlabeled_limit, seed=self.subset_seed + 10, group_by_recording=True
                    )
                else:
                    sub_p, sub_l = eligible_p, eligible_l

                unlabeled_paths_list.extend(sub_p)
                unlabeled_labels_list.extend(sub_l)

        unlabeled_tr_paths = np.array(unlabeled_paths_list)
        unlabeled_tr_labels = np.array(unlabeled_labels_list)

        # 3. Validation & Test splits per domain
        ind_val_p, ind_val_l = filter_paths_by_sample_limit(
            indoor_splits["val"][0], indoor_splits["val"][1], self.val_limit, seed=self.seed
        )
        ind_ts_p, ind_ts_l = filter_paths_by_sample_limit(
            indoor_splits["test"][0], indoor_splits["test"][1], self.test_limit, seed=self.seed
        )

        out_val_p, out_val_l = filter_paths_by_sample_limit(
            outdoor_splits["val"][0], outdoor_splits["val"][1], self.val_limit, seed=self.seed + 2
        )
        out_ts_p, out_ts_l = filter_paths_by_sample_limit(
            outdoor_splits["test"][0], outdoor_splits["test"][1], self.test_limit, seed=self.seed + 2
        )

        # 4. Construct TF Pipelines
        primary_builder = self.indoor_builder

        labeled_train_ds = primary_builder._create_pipeline(
            labeled_tr_paths, labeled_tr_labels, augment=True, batch_size=self.batch_size, shuffle=True, one_hot=True
        )

        unlabeled_weak_ds = primary_builder._create_pipeline(
            unlabeled_tr_paths, unlabeled_tr_labels, augment=False, batch_size=self.batch_size, shuffle=True, one_hot=True
        )
        unlabeled_strong_ds = primary_builder._create_pipeline(
            unlabeled_tr_paths, unlabeled_tr_labels, augment=True, batch_size=self.batch_size, shuffle=True, one_hot=True
        )

        unlabeled_train_ds = tf.data.Dataset.zip((unlabeled_weak_ds, unlabeled_strong_ds)).map(
            lambda w_batch, s_batch: (w_batch[0], s_batch[0]),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        train_zipped_ds = tf.data.Dataset.zip((labeled_train_ds, unlabeled_train_ds))

        # Val & Test Pipelines
        indoor_val_ds = self.indoor_builder._create_pipeline(
            ind_val_p, ind_val_l, augment=False, batch_size=self.batch_size, shuffle=False, one_hot=True
        )
        indoor_test_ds = self.indoor_builder._create_pipeline(
            ind_ts_p, ind_ts_l, augment=False, batch_size=self.batch_size, shuffle=False, one_hot=True
        )

        outdoor_val_ds = self.outdoor_builder._create_pipeline(
            out_val_p, out_val_l, augment=False, batch_size=self.batch_size, shuffle=False, one_hot=True
        )
        outdoor_test_ds = self.outdoor_builder._create_pipeline(
            out_ts_p, out_ts_l, augment=False, batch_size=self.batch_size, shuffle=False, one_hot=True
        )

        manifests = {
            "labeled": list(labeled_tr_paths),
            "unlabeled": list(unlabeled_tr_paths),
            "indoor_val": list(ind_val_p),
            "indoor_test": list(ind_ts_p),
            "outdoor_val": list(out_val_p),
            "outdoor_test": list(out_ts_p),
        }

        counts = {
            "labeled_train": len(labeled_tr_paths),
            "unlabeled_train": len(unlabeled_tr_paths),
            "indoor_train": len(indoor_splits["train"][0]),
            "indoor_val": len(ind_val_p),
            "indoor_test": len(ind_ts_p),
            "outdoor_train": len(outdoor_splits["train"][0]),
            "outdoor_val": len(out_val_p),
            "outdoor_test": len(out_ts_p),
        }

        return {
            "train_ds": train_zipped_ds,
            "labeled_train_ds": labeled_train_ds,
            "unlabeled_train_ds": unlabeled_train_ds,
            "source_val_ds": indoor_val_ds,
            "source_test_ds": indoor_test_ds,
            "target_val_ds": outdoor_val_ds,
            "target_test_ds": outdoor_test_ds,
            "validation": {
                "indoor": indoor_val_ds,
                "outdoor": outdoor_val_ds,
            },
            "test": {
                "indoor": indoor_test_ds,
                "outdoor": outdoor_test_ds,
            },
            "manifests": manifests,
            "counts": counts,
        }


def build_ssl_datasets(config: Any) -> Dict[str, Any]:
    """Helper entrypoint to construct sample-limited SSL datasets."""
    builder = SSLDatasetBuilder(config)
    return builder.build_datasets()


__all__ = [
    "get_recording_id",
    "filter_paths_by_sample_limit",
    "SSLDatasetBuilder",
    "build_ssl_datasets",
]
