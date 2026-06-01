import os
import yaml
import numpy as np
import tensorflow as tf
import keras
from typing import Dict, Tuple, List
from sklearn.metrics import classification_report, confusion_matrix, f1_score

class ModelEvaluator:
    def __init__(self, model: keras.Model, classes: list, loss_fn=None):
        self.model = model
        self.classes = classes
        self.loss_fn = loss_fn or tf.keras.losses.CategoricalCrossentropy()
        
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
        End-of-epoch evaluation including Macro/Weighted F1.
        """
        self.val_loss_metric.reset_state()
        self.val_acc_metric.reset_state()
        
        y_true, y_pred = self._collect_predictions(dataset)

        return {
            "loss": float(self.val_loss_metric.result()),
            "accuracy": float(self.val_acc_metric.result()),
            "macro_f1": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }

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
