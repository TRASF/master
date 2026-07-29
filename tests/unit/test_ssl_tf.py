"""Unit tests for TensorFlow SSL loss functions, dataset sample limiting, and multi-domain evaluation."""

import numpy as np
import pytest
import tensorflow as tf

from wingbeat_ml.data.ssl_dataset import filter_paths_by_sample_limit, get_recording_id
from wingbeat_ml.training.tf_ssl_losses import (
    TFFlexMatchLoss,
    compute_classification_metrics,
    evaluate_tf_domain_performance,
    tf_compute_fixmatch_loss,
    train_tf_fixmatch_step,
    train_tf_flexmatch_step,
)


def test_get_recording_id():
    path1 = "/data/indoor/mosquito_rec01_seg0.wav"
    path2 = "/data/indoor/mosquito_rec01_seg1.wav"
    path3 = "/data/outdoor/rec02.wav"

    assert get_recording_id(path1) == "mosquito_rec01"
    assert get_recording_id(path2) == "mosquito_rec01"
    assert get_recording_id(path3) == "rec02"


def test_filter_paths_by_sample_limit():
    paths = np.array([f"rec_{i//5}_seg_{i%5}.wav" for i in range(100)])
    # 50 samples class 0, 50 samples class 1
    labels = np.array([0] * 50 + [1] * 50)

    # Sub-sample to 10 samples per class without grouping
    filtered_p, filtered_l = filter_paths_by_sample_limit(paths, labels, samples_per_class=10, seed=42)

    assert len(filtered_p) == 20
    assert (filtered_l == 0).sum() == 10
    assert (filtered_l == 1).sum() == 10

    # Sub-sample with group_by_recording
    filtered_gp, filtered_gl = filter_paths_by_sample_limit(
        paths, labels, samples_per_class=10, seed=42, group_by_recording=True
    )
    assert len(filtered_gp) == 20
    assert (filtered_gl == 0).sum() == 10
    assert (filtered_gl == 1).sum() == 10


def test_compute_classification_metrics():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 0, 2, 2])

    metrics = compute_classification_metrics(y_true, y_pred, num_classes=3)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "per_class_f1" in metrics
    assert len(metrics["per_class_f1"]) == 3
    assert metrics["macro_f1"] > 0.0


def test_tf_fixmatch_loss_calculation():
    labeled_logits = tf.constant([[2.0, 0.5, -1.0], [0.1, 3.0, 0.2]])
    labels = tf.constant([0, 1], dtype=tf.int64)

    unlabeled_w_logits = tf.constant([[3.0, 0.1, 0.0], [0.0, 0.1, 3.0]])
    unlabeled_s_logits = tf.constant([[2.5, 0.2, 0.1], [0.1, 0.2, 2.8]])

    results = tf_compute_fixmatch_loss(
        labeled_logits=labeled_logits,
        labels=labels,
        unlabeled_weak_logits=unlabeled_w_logits,
        unlabeled_strong_logits=unlabeled_s_logits,
        tau=0.90,
    )

    assert "total_loss" in results
    assert float(results["mask_ratio"].numpy()) == 1.0
    assert list(results["pseudo_labels"].numpy()) == [0, 2]
    assert float(results["total_loss"].numpy()) > 0.0


def test_tf_flexmatch_loss_and_thresholds():
    flex_layer = TFFlexMatchLoss(num_classes=3, tau=0.95, mapping="convex")

    pseudo_labels = tf.constant([0, 0, 0, 1], dtype=tf.int64)
    max_probs = tf.constant([0.98, 0.97, 0.96, 0.99], dtype=tf.float32)

    flex_layer.update_class_counts(pseudo_labels, max_probs)

    counts = flex_layer.class_counts.numpy()
    assert counts[0] == 3.0
    assert counts[1] == 1.0
    assert counts[2] == 0.0

    thresholds = flex_layer.compute_class_thresholds().numpy()
    assert np.isclose(thresholds[0], 0.95)
    assert np.isclose(thresholds[1], 0.19)
    assert np.isclose(thresholds[2], 0.0)


def test_tf_training_steps():
    # Build tiny sequential model
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(2400, 1)),
        tf.keras.layers.Conv1D(8, 5),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(3),
    ])
    optimizer = tf.keras.optimizers.Adam(1e-3)

    x_l = tf.random.normal((4, 2400, 1))
    y_l = tf.constant([0, 1, 2, 0], dtype=tf.int64)
    x_u_w = tf.random.normal((6, 2400, 1))
    x_u_s = tf.random.normal((6, 2400, 1))

    # Test FixMatch step
    step_res = train_tf_fixmatch_step(
        model, optimizer, x_l, y_l, x_u_w, x_u_s, tau=0.5
    )
    assert "total_loss" in step_res
    assert step_res["total_loss"] > 0.0

    # Test FlexMatch step
    flex_layer = TFFlexMatchLoss(num_classes=3, tau=0.90)
    flex_res = train_tf_flexmatch_step(
        model, flex_layer, optimizer, x_l, y_l, x_u_w, x_u_s
    )
    assert "total_loss" in flex_res
    assert flex_res["total_loss"] > 0.0
