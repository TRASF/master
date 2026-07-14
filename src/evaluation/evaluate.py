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
        self.is_contrastive = "contrastive" in getattr(self.loss_fn, "name", "").lower()
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
        if not self.is_contrastive:
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

    def _to_probabilities(self, predictions):
        if getattr(self.loss_fn, "from_logits", False):
            return tf.nn.softmax(predictions, axis=-1)
        return predictions

    def _per_sample_loss(self, y_true, predictions):
        if self.is_contrastive:
            return tf.zeros([tf.shape(y_true)[0]], dtype=tf.float32)

        return tf.keras.losses.categorical_crossentropy(
            y_true,
            predictions,
            from_logits=getattr(self.loss_fn, "from_logits", False),
        )

    def collect_prediction_diagnostics(
        self,
        dataset: tf.data.Dataset,
        split_name: str = "test",
        max_rows: int = None,
        sample_rate: int = 8000,
        include_audio: bool = False,
        wandb_module=None,
    ) -> Dict:
        """
        Collect per-segment prediction rows for W&B Tables and downstream plots.
        """
        columns = [
            "split",
            "sample_index",
            "true_label",
            "pred_label",
            "is_correct",
            "confidence",
            "top2_label",
            "top2_confidence",
            "top2_margin",
            "per_sample_loss",
        ]
        prob_columns = [f"prob/{class_name}" for class_name in self.classes]
        columns.extend(prob_columns)
        if include_audio:
            columns.append("audio")

        rows = []
        all_true = []
        all_pred = []
        all_conf = []
        sample_index = 0

        for x, y in dataset:
            predictions = self.model(x, training=False)
            probabilities = self._to_probabilities(predictions).numpy()
            losses = self._per_sample_loss(y, predictions).numpy()
            true_idx = np.argmax(y.numpy(), axis=1)
            pred_idx = np.argmax(probabilities, axis=1)
            top2_idx = np.argsort(probabilities, axis=1)[:, -2:][:, ::-1]

            x_np = x.numpy() if include_audio else None

            for i in range(len(true_idx)):
                true_label = int(true_idx[i])
                pred_label = int(pred_idx[i])
                confidence = float(probabilities[i, pred_label])
                second_label = int(top2_idx[i, 1])
                second_conf = float(probabilities[i, second_label])

                all_true.append(true_label)
                all_pred.append(pred_label)
                all_conf.append(confidence)

                if max_rows is not None and len(rows) >= max_rows:
                    sample_index += 1
                    continue

                row = [
                    split_name,
                    sample_index,
                    self.classes[true_label],
                    self.classes[pred_label],
                    bool(true_label == pred_label),
                    confidence,
                    self.classes[second_label],
                    second_conf,
                    confidence - second_conf,
                    float(losses[i]),
                ]
                row.extend([float(p) for p in probabilities[i]])

                if include_audio:
                    audio = np.squeeze(x_np[i]).astype(np.float32)
                    if wandb_module is not None:
                        row.append(wandb_module.Audio(audio, sample_rate=sample_rate))
                    else:
                        row.append(audio.tolist())

                rows.append(row)
                sample_index += 1

        return {
            "columns": columns,
            "data": rows,
            "y_true": np.array(all_true, dtype=np.int32),
            "y_pred": np.array(all_pred, dtype=np.int32),
            "confidence": np.array(all_conf, dtype=np.float32),
        }

    def evaluate_epoch(self, dataset: tf.data.Dataset) -> Dict[str, float]:
        """
        End-of-epoch evaluation including Macro/Weighted F1, Precision, and Recall.
        """
        self.val_loss_metric.reset_state()
        self.val_acc_metric.reset_state()
        
        if self.is_contrastive:
            # Just compute loss over the dataset
            for x, y in dataset:
                self.val_step(x, y)
            return {
                "loss": float(self.val_loss_metric.result()),
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "female_f1": 0.0, "female_prec": 0.0, "female_rec": 0.0,
                "male_f1": 0.0, "male_prec": 0.0, "male_rec": 0.0,
                "weighted_f1": 0.0
            }

        y_true, y_pred = self._collect_predictions(dataset)

        # 1. Calculate per-class metrics for group averaging
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=range(len(self.classes)), zero_division=0)
        per_class_prec = precision_score(y_true, y_pred, average=None, labels=range(len(self.classes)), zero_division=0)
        per_class_rec = recall_score(y_true, y_pred, average=None, labels=range(len(self.classes)), zero_division=0)
        
        # 2. Identify indices for Female and Male classes (filtered by present classes in y_true)
        present_classes = np.unique(y_true)
        female_indices = [i for i, name in enumerate(self.classes) if 'Female' in name and i in present_classes]
        male_indices = [i for i, name in enumerate(self.classes) if 'Male' in name and i in present_classes]

        def get_group_metric(scores, indices):
            if not indices: return 0.0
            return float(np.mean([scores[i] for i in indices]))

        # Calculate macro_f1 only over classes actually present in y_true
        macro_f1 = float(np.mean(per_class_f1[present_classes])) if len(present_classes) > 0 else 0.0

        metrics_dict = {
            "loss": float(self.val_loss_metric.result()),
            "accuracy": float(self.val_acc_metric.result()),
            "macro_f1": macro_f1,
            "female_f1": get_group_metric(per_class_f1, female_indices),
            "female_prec": get_group_metric(per_class_prec, female_indices),
            "female_rec": get_group_metric(per_class_rec, female_indices),
            "male_f1": get_group_metric(per_class_f1, male_indices),
            "male_prec": get_group_metric(per_class_prec, male_indices),
            "male_rec": get_group_metric(per_class_rec, male_indices),
            "weighted_f1": float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }

        # Add per-class f1, precision, and recall scores
        for i, class_name in enumerate(self.classes):
            metrics_dict[f"class_f1/{class_name}"] = float(per_class_f1[i])
            metrics_dict[f"class_precision/{class_name}"] = float(per_class_prec[i])
            metrics_dict[f"class_recall/{class_name}"] = float(per_class_rec[i])

        return metrics_dict

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
        File-level evaluation: frame each recording, average segment predictions per file,
        then compute one prediction per original recording.
        """
        y_true = []
        y_pred = []
        losses = []
        num_classes = len(self.classes)
        file_diagnostics = []

        for file_path, label in zip(file_paths, labels):
            label = int(label)
            file_path = str(file_path)
            if file_path not in self._audio_cache:
                self._audio_cache[file_path] = load_fn(file_path)
            audio = self._audio_cache[file_path]
            segments = augmentor.create_segments(
                tf.convert_to_tensor(audio, dtype=tf.float32),
                tf.constant(label, dtype=tf.int32),
                training=False,
            )
            segments = segments.map(
                lambda x, y, seed: augmentor.apply_post_processing(x, y, seed=seed, augment=False),
                num_parallel_calls=tf.data.AUTOTUNE,
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

            preds_concat = tf.concat(preds, axis=0)
            file_logits = tf.reduce_mean(preds_concat, axis=0, keepdims=True)
            file_label = tf.one_hot([label], num_classes)
            loss = self.loss_fn(file_label, file_logits)
            if len(loss.shape) > 0:
                loss = tf.reduce_mean(loss)

            losses.append(float(loss.numpy()))

            # Predict based on average probabilities (soft voting)
            if self.loss_fn.from_logits:
                preds_prob = tf.nn.softmax(preds_concat, axis=-1)
            else:
                preds_prob = preds_concat
            file_probs = tf.reduce_mean(preds_prob, axis=0, keepdims=True)

            file_pred_idx = int(tf.argmax(file_probs, axis=-1).numpy()[0])
            y_true.append(label)
            y_pred.append(file_pred_idx)

            # Segment-level correctness analysis
            seg_probs = preds_prob.numpy()
            seg_preds = np.argmax(seg_probs, axis=1)
            total_segments = len(seg_preds)
            correct_segments = int(np.sum(seg_preds == label))
            segment_accuracy = float(correct_segments / total_segments) if total_segments > 0 else 0.0
            incorrect_indices = np.where(seg_preds != label)[0].tolist()

            file_diagnostics.append({
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "true_label": self.classes[label],
                "pred_label": self.classes[file_pred_idx],
                "is_correct": bool(label == file_pred_idx),
                "loss": float(loss.numpy()),
                "confidence": float(file_probs.numpy()[0, file_pred_idx]),
                "total_segments": total_segments,
                "correct_segments": correct_segments,
                "segment_accuracy": segment_accuracy,
                "incorrect_segments": incorrect_indices,
            })

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        labels_arg = list(range(num_classes))

        # Calculate macro_f1 only over classes actually present in y_true
        present_classes = np.unique(y_true)
        per_class_f1 = f1_score(y_true, y_pred, labels=labels_arg, average=None, zero_division=0)
        macro_f1 = float(np.mean(per_class_f1[present_classes])) if len(present_classes) > 0 else 0.0

        metrics = {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
            "macro_f1": macro_f1,
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
            "file_diagnostics": file_diagnostics,
        }

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            saved_results = {
                k: v for k, v in results.items()
                if k not in ("file_diagnostics") # Keep diagnostic list in runtime but omit raw dump if too large, or dump it to a separate file
            }
            # Let's save the full results including diagnostics to file_level_results.yaml
            with open(os.path.join(save_dir, filename), "w") as f:
                yaml.dump(results, f)
            print(f"File-level results saved to {save_dir}")

        return results

    def evaluate_final_test(
        self,
        dataset: tf.data.Dataset,
        save_dir: str = None,
        return_predictions: bool = False,
    ) -> Dict:
        """
        Comprehensive evaluation for the final test set.
        """
        self.val_loss_metric.reset_state()
        self.val_acc_metric.reset_state()
        
        if self.is_contrastive:
            for x, y in dataset:
                self.val_step(x, y)
            results = {
                "metrics": {
                    "loss": float(self.val_loss_metric.result()),
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                    "weighted_f1": 0.0
                },
                "report": {},
                "confusion_matrix": []
            }
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, "final_test_results.yaml"), "w") as f:
                    yaml.dump(results, f)
            return results

        y_true, y_pred = self._collect_predictions(dataset)

        present_classes = np.unique(y_true)
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=range(len(self.classes)), zero_division=0)
        macro_f1 = float(np.mean(per_class_f1[present_classes])) if len(present_classes) > 0 else 0.0

        # 1. Get average loss/acc
        metrics = {
            "loss": float(self.val_loss_metric.result()),
            "accuracy": float(self.val_acc_metric.result()),
            "macro_f1": macro_f1,
            "weighted_f1": float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }

        # 2. Get detailed Sklearn metrics
        report = classification_report(
            y_true,
            y_pred,
            labels=range(len(self.classes)),
            target_names=self.classes,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred)

        results = {
            "metrics": metrics,
            "report": report,
            "confusion_matrix": cm.tolist()
        }

        if return_predictions:
            results["y_true"] = y_true.tolist()
            results["y_pred"] = y_pred.tolist()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            saved_results = {
                k: v for k, v in results.items()
                if k not in ("y_true", "y_pred")
            }
            with open(os.path.join(save_dir, "final_test_results.yaml"), "w") as f:
                yaml.dump(saved_results, f)
            print(f"Final test results saved to {save_dir}")

        return results
