import os
import yaml
import numpy as np
import tensorflow as tf
import keras
from typing import Dict, Tuple, List
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

class ModelEvaluator:
    def __init__(self, model: keras.Model, classes: list, loss_fn=None):
        self.model = model
        self.classes = classes
        self.loss_fn = loss_fn or tf.keras.losses.CategoricalCrossentropy()
        self._audio_cache = {}
        
        # Define Metrics for Custom Eval
        self.val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
        self.val_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

    @tf.function
    def val_step(self, x, y):
        """
        Fast evaluation step for a single batch.
        Can be called inside your custom training loop.
        """
        predictions = self.model(x, training=False)
        loss = self.loss_fn(y, predictions)
        
        # Update running metrics
        self.val_loss_metric.update_state(loss)
        self.val_acc_metric.update_state(y, predictions)
        return loss, predictions

    def _collect_predictions(self, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Helper to run the model on a dataset and return true/pred indices.
        """
        y_true = []
        y_pred = []
        
        for x, y in dataset:
            _, preds = self.val_step(x, y)
            y_true.extend(np.argmax(y.numpy(), axis=1))
            y_pred.extend(np.argmax(preds.numpy(), axis=1))
            
        return np.array(y_true), np.array(y_pred)

    def evaluate_epoch(self, dataset: tf.data.Dataset) -> Dict[str, float]:
        """
        End-of-epoch evaluation including Macro/Weighted F1, Precision, and Recall.
        """
        self.val_loss_metric.reset_state()
        self.val_acc_metric.reset_state()
        
        y_true, y_pred = self._collect_predictions(dataset)

        # 1. Calculate per-class metrics for group averaging
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=range(len(self.classes)), zero_division=0)
        per_class_prec = precision_score(y_true, y_pred, average=None, labels=range(len(self.classes)), zero_division=0)
        per_class_rec = recall_score(y_true, y_pred, average=None, labels=range(len(self.classes)), zero_division=0)
        
        # 2. Identify indices for Female and Male classes
        female_indices = [i for i, name in enumerate(self.classes) if 'Female' in name]
        male_indices = [i for i, name in enumerate(self.classes) if 'Male' in name]

        def get_group_metric(scores, indices):
            if not indices: return 0.0
            return float(np.mean([scores[i] for i in indices]))

        return {
            "loss": float(self.val_loss_metric.result()),
            "accuracy": float(self.val_acc_metric.result()),
            "macro_f1": float(np.mean(per_class_f1)),
            "female_f1": get_group_metric(per_class_f1, female_indices),
            "female_prec": get_group_metric(per_class_prec, female_indices),
            "female_rec": get_group_metric(per_class_rec, female_indices),
            "male_f1": get_group_metric(per_class_f1, male_indices),
            "male_prec": get_group_metric(per_class_prec, male_indices),
            "male_rec": get_group_metric(per_class_rec, male_indices),
            "weighted_f1": float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }

    def evaluate_files(
        self,
        file_paths,
        labels,
        load_fn,
        augmentor,
        batch_size: int = 256,
        save_dir: str = None,
        filename: str = "file_level_results.yaml",
    ) -> Dict:
        """
        File-level evaluation: frame each recording, average segment logits per file,
        then compute one prediction per original recording.
        """
        y_true = []
        y_pred = []
        losses = []
        num_classes = len(self.classes)

        for file_path, label in zip(file_paths, labels):
            label = int(label)
            file_path = str(file_path)
            if file_path not in self._audio_cache:
                self._audio_cache[file_path] = load_fn(file_path)
            audio = self._audio_cache[file_path]
            segments = augmentor.create_segments(
                tf.convert_to_tensor(audio, dtype=tf.float32),
                tf.constant(label, dtype=tf.int32),
                step_ratio=1.0,
                training=False,
            )
            segments = segments.map(
                lambda x, y: (tf.expand_dims(x, -1), tf.one_hot(y, num_classes)),
                num_parallel_calls=tf.data.AUTOTUNE,
            ).batch(batch_size)

            preds = []
            for x, _ in segments:
                preds.append(self.model(x, training=False))

            if not preds:
                continue

            file_logits = tf.reduce_mean(tf.concat(preds, axis=0), axis=0, keepdims=True)
            file_label = tf.one_hot([label], num_classes)
            loss = self.loss_fn(file_label, file_logits)
            if len(loss.shape) > 0:
                loss = tf.reduce_mean(loss)

            losses.append(float(loss.numpy()))
            y_true.append(label)
            y_pred.append(int(tf.argmax(file_logits, axis=-1).numpy()[0]))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        labels_arg = list(range(num_classes))

        metrics = {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels_arg, average='macro', zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels_arg, average='weighted', zero_division=0)),
        }
        report = classification_report(
            y_true,
            y_pred,
            labels=labels_arg,
            target_names=self.classes,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels_arg)

        results = {
            "evaluation_level": "file",
            "metrics": metrics,
            "report": report,
            "confusion_matrix": cm.tolist(),
        }

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, filename), "w") as f:
                yaml.dump(results, f)
            print(f"File-level results saved to {save_dir}")

        return results

    def evaluate_final_test(self, dataset: tf.data.Dataset, save_dir: str = None) -> Dict:
        """
        Comprehensive evaluation for the final test set.
        """
        self.val_loss_metric.reset_state()
        self.val_acc_metric.reset_state()
        
        y_true, y_pred = self._collect_predictions(dataset)

        # 1. Get average loss/acc
        metrics = {
            "loss": float(self.val_loss_metric.result()),
            "accuracy": float(self.val_acc_metric.result()),
            "macro_f1": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }

        # 2. Get detailed Sklearn metrics
        report = classification_report(y_true, y_pred, target_names=self.classes, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        results = {
            "metrics": metrics,
            "report": report,
            "confusion_matrix": cm.tolist()
        }

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "final_test_results.yaml"), "w") as f:
                yaml.dump(results, f)
            print(f"Final test results saved to {save_dir}")

        return results
