"""TensorFlow / Keras implementations of FixMatch and FlexMatch loss functions,
training step functions, and cross-domain evaluation routines.

Formulations:
- FixMatch: Sohn et al., NeurIPS 2020.
- FlexMatch: Zhang et al., NeurIPS 2021.
"""

import numpy as np
import tensorflow as tf
from typing import Any, Dict, Tuple


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 11,
) -> Dict[str, Any]:
    """Compute per-class and macro classification metrics."""
    acc = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0
    f1s = []
    precisions = []
    recalls = []

    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = float(tp / (tp + fp + 1e-12))
        recall = float(tp / (tp + fn + 1e-12))
        f1 = float(2 * precision * recall / (precision + recall + 1e-12))

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": f1s,
        "per_class_precision": precisions,
        "per_class_recall": recalls,
    }


def tf_compute_fixmatch_loss(
    labeled_logits: tf.Tensor,
    labels: tf.Tensor,
    unlabeled_weak_logits: tf.Tensor,
    unlabeled_strong_logits: tf.Tensor,
    tau: float = 0.95,
    lambda_u: float = 1.0,
) -> Dict[str, tf.Tensor]:
    """
    Compute mathematically correct FixMatch loss in TensorFlow.

    Gradients are isolated from weak-view logits via tf.stop_gradient.
    """
    num_classes = tf.shape(labeled_logits)[-1]

    # 1. Supervised Loss (Categorical Cross-Entropy)
    if len(labels.shape) > 1 and labels.shape[-1] == num_classes:
        loss_s = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(labels, labeled_logits, from_logits=True)
        )
    else:
        loss_s = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(labels, labeled_logits, from_logits=True)
        )

    # 2. Detached Pseudo-labeling from Weak Logits
    probs_w = tf.stop_gradient(tf.nn.softmax(unlabeled_weak_logits))
    max_probs = tf.reduce_max(probs_w, axis=-1)
    pseudo_labels = tf.argmax(probs_w, axis=-1, output_type=tf.int64)

    mask = tf.cast(max_probs >= tau, tf.float32)
    mask_ratio = tf.reduce_mean(mask)

    # 3. Unsupervised Loss on Strongly Augmented Logits
    mask_sum = tf.reduce_sum(mask)
    ce_u = tf.keras.losses.sparse_categorical_crossentropy(pseudo_labels, unlabeled_strong_logits, from_logits=True)

    # Normalize by number of accepted samples or safely output 0
    loss_u = tf.cond(
        mask_sum > 0.0,
        lambda: tf.reduce_sum(ce_u * mask) / mask_sum,
        lambda: 0.0 * tf.reduce_sum(unlabeled_strong_logits),
    )

    total_loss = loss_s + lambda_u * loss_u

    return {
        "total_loss": total_loss,
        "loss_s": loss_s,
        "loss_u": loss_u,
        "mask_ratio": mask_ratio,
        "pseudo_labels": pseudo_labels,
        "max_probs": max_probs,
        "mask": mask,
    }


class TFFlexMatchLoss(tf.keras.layers.Layer):
    """
    Stateful Keras Layer tracking per-class pseudo-label counts and calculating
    dynamic class thresholds for FlexMatch Curriculum Pseudo-Labeling (CPL).
    """

    def __init__(
        self,
        num_classes: int,
        tau: float = 0.95,
        lambda_u: float = 1.0,
        mapping: str = "convex",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.tau = tau
        self.lambda_u = lambda_u
        self.mapping = mapping

        # Per-class selection counts tracker
        self.class_counts = self.add_weight(
            name="flexmatch_class_counts",
            shape=(num_classes,),
            initializer="zeros",
            trainable=False,
            dtype=tf.float32,
        )

    def compute_class_thresholds(self) -> tf.Tensor:
        """Calculate dynamic class thresholds tau_t(c)."""
        max_count = tf.reduce_max(self.class_counts)
        beta = tf.cond(
            max_count > 0.0,
            lambda: self.class_counts / tf.maximum(max_count, 1.0),
            lambda: tf.zeros_like(self.class_counts),
        )

        if self.mapping == "convex":
            mapping_val = beta / (2.0 - beta + 1e-12)
        elif self.mapping == "concave":
            mapping_val = tf.sin(0.5 * tf.constant(3.141592653589793) * beta)
        elif self.mapping == "linear":
            mapping_val = beta
        else:
            raise ValueError(f"Unknown mapping: {self.mapping!r}")

        return mapping_val * self.tau

    def update_class_counts(self, pseudo_labels: tf.Tensor, max_probs: tf.Tensor) -> None:
        """Update per-class selection counts for samples passing base threshold tau."""
        valid_mask = max_probs >= self.tau
        valid_labels = tf.boolean_mask(pseudo_labels, valid_mask)

        if tf.shape(valid_labels)[0] > 0:
            batch_counts = tf.math.unsorted_segment_sum(
                tf.ones_like(valid_labels, dtype=tf.float32),
                tf.cast(valid_labels, tf.int32),
                self.num_classes,
            )
            self.class_counts.assign_add(batch_counts)

    def call(
        self,
        labeled_logits: tf.Tensor,
        labels: tf.Tensor,
        unlabeled_weak_logits: tf.Tensor,
        unlabeled_strong_logits: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:

        num_classes = tf.shape(labeled_logits)[-1]

        # 1. Supervised Loss
        if len(labels.shape) > 1 and labels.shape[-1] == num_classes:
            loss_s = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(labels, labeled_logits, from_logits=True)
            )
        else:
            loss_s = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(labels, labeled_logits, from_logits=True)
            )

        # 2. Detached Pseudo-labeling from Weak Logits
        probs_w = tf.stop_gradient(tf.nn.softmax(unlabeled_weak_logits))
        max_probs = tf.reduce_max(probs_w, axis=-1)
        pseudo_labels = tf.argmax(probs_w, axis=-1, output_type=tf.int64)

        # Update per-class counts inside stop_gradient
        self.update_class_counts(pseudo_labels, max_probs)

        # Compute dynamic class thresholds
        class_thresholds = self.compute_class_thresholds()

        # Gather threshold for each sample according to predicted class
        sample_thresholds = tf.gather(class_thresholds, pseudo_labels)
        mask = tf.cast(max_probs >= sample_thresholds, tf.float32)
        mask_ratio = tf.reduce_mean(mask)

        # 3. Unsupervised Loss on Strongly Augmented Logits
        mask_sum = tf.reduce_sum(mask)
        ce_u = tf.keras.losses.sparse_categorical_crossentropy(pseudo_labels, unlabeled_strong_logits, from_logits=True)

        loss_u = tf.cond(
            mask_sum > 0.0,
            lambda: tf.reduce_sum(ce_u * mask) / mask_sum,
            lambda: 0.0 * tf.reduce_sum(unlabeled_strong_logits),
        )

        total_loss = loss_s + self.lambda_u * loss_u

        return {
            "total_loss": total_loss,
            "loss_s": loss_s,
            "loss_u": loss_u,
            "mask_ratio": mask_ratio,
            "pseudo_labels": pseudo_labels,
            "max_probs": max_probs,
            "mask": mask,
            "class_thresholds": class_thresholds,
            "class_counts": self.class_counts,
        }


def train_tf_fixmatch_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    x_l: tf.Tensor,
    y_l: tf.Tensor,
    x_u_w: tf.Tensor,
    x_u_s: tf.Tensor,
    tau: float = 0.95,
    lambda_u: float = 1.0,
) -> Dict[str, float]:
    """Execute one TensorFlow FixMatch training step using tf.GradientTape."""
    with tf.GradientTape() as tape:
        labeled_logits = model(x_l, training=True)
        unlabeled_weak_logits = tf.stop_gradient(model(x_u_w, training=True))
        unlabeled_strong_logits = model(x_u_s, training=True)

        loss_res = tf_compute_fixmatch_loss(
            labeled_logits=labeled_logits,
            labels=y_l,
            unlabeled_weak_logits=unlabeled_weak_logits,
            unlabeled_strong_logits=unlabeled_strong_logits,
            tau=tau,
            lambda_u=lambda_u,
        )
        total_loss = loss_res["total_loss"]

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return {
        "total_loss": float(loss_res["total_loss"].numpy()),
        "loss_s": float(loss_res["loss_s"].numpy()),
        "loss_u": float(loss_res["loss_u"].numpy()),
        "mask_ratio": float(loss_res["mask_ratio"].numpy()),
    }


def train_tf_flexmatch_step(
    model: tf.keras.Model,
    flexmatch_layer: TFFlexMatchLoss,
    optimizer: tf.keras.optimizers.Optimizer,
    x_l: tf.Tensor,
    y_l: tf.Tensor,
    x_u_w: tf.Tensor,
    x_u_s: tf.Tensor,
) -> Dict[str, float]:
    """Execute one TensorFlow FlexMatch training step using tf.GradientTape."""
    with tf.GradientTape() as tape:
        labeled_logits = model(x_l, training=True)
        unlabeled_weak_logits = tf.stop_gradient(model(x_u_w, training=True))
        unlabeled_strong_logits = model(x_u_s, training=True)

        loss_res = flexmatch_layer(
            labeled_logits=labeled_logits,
            labels=y_l,
            unlabeled_weak_logits=unlabeled_weak_logits,
            unlabeled_strong_logits=unlabeled_strong_logits,
        )
        total_loss = loss_res["total_loss"]

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return {
        "total_loss": float(loss_res["total_loss"].numpy()),
        "loss_s": float(loss_res["loss_s"].numpy()),
        "loss_u": float(loss_res["loss_u"].numpy()),
        "mask_ratio": float(loss_res["mask_ratio"].numpy()),
    }


def evaluate_tf_domain_performance(
    model: tf.keras.Model,
    source_val_ds: Any,
    target_val_ds: Any,
    num_classes: int = 11,
) -> Dict[str, float]:
    """Evaluate accuracy, loss, macro-F1 and worst-domain performance of TensorFlow model across domains."""
    def _eval(ds: Any) -> Tuple[float, float, float]:
        if ds is None:
            return 0.0, 0.0, 0.0

        total_loss = 0.0
        total_samples = 0.0
        all_true = []
        all_pred = []

        for x_batch, y_batch in ds:
            logits = model(x_batch, training=False)
            if len(y_batch.shape) > 1 and y_batch.shape[-1] > 1:
                labels = tf.argmax(y_batch, axis=-1)
                loss = tf.reduce_sum(
                    tf.keras.losses.categorical_crossentropy(y_batch, logits, from_logits=True)
                )
            else:
                labels = tf.cast(y_batch, tf.int64)
                loss = tf.reduce_sum(
                    tf.keras.losses.sparse_categorical_crossentropy(y_batch, logits, from_logits=True)
                )

            preds = tf.argmax(logits, axis=-1, output_type=tf.int64)
            total_loss += float(loss.numpy())
            total_samples += float(tf.shape(y_batch)[0].numpy())
            all_true.extend(labels.numpy())
            all_pred.extend(preds.numpy())

        if total_samples == 0:
            return 0.0, 0.0, 0.0

        avg_loss = total_loss / total_samples
        metrics = compute_classification_metrics(np.array(all_true), np.array(all_pred), num_classes=num_classes)
        return avg_loss, metrics["accuracy"], metrics["macro_f1"]

    src_loss, src_acc, src_f1 = _eval(source_val_ds)
    tgt_loss, tgt_acc, tgt_f1 = _eval(target_val_ds)

    worst_f1 = min(src_f1, tgt_f1)
    mean_f1 = (src_f1 + tgt_f1) / 2.0

    return {
        "source_loss": src_loss,
        "source_accuracy": src_acc,
        "source_macro_f1": src_f1,
        "target_loss": tgt_loss,
        "target_accuracy": tgt_acc,
        "target_macro_f1": tgt_f1,
        "indoor_macro_f1": src_f1,
        "outdoor_macro_f1": tgt_f1,
        "worst_domain_macro_f1": worst_f1,
        "mean_domain_macro_f1": mean_f1,
    }


__all__ = [
    "tf_compute_fixmatch_loss",
    "TFFlexMatchLoss",
    "train_tf_fixmatch_step",
    "train_tf_flexmatch_step",
    "evaluate_tf_domain_performance",
    "compute_classification_metrics",
]
