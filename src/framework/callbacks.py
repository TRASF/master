import os
import tensorflow as tf


class MetricMonitor:
    def __init__(self, monitor="val_loss", mode="min", min_delta=0.0):
        self.monitor, self.mode, self.min_delta = monitor, mode, min_delta
        self.best = float("inf") if mode == "min" else float("-inf")

    def get_value(self, logs):
        return float(logs[self.monitor]) if isinstance(logs, dict) else float(logs)

    def is_improved(self, current):
        return current < self.best - self.min_delta if self.mode == "min" else current > self.best + self.min_delta

    def update(self, current):
        if self.is_improved(current):
            self.best = current
            return True
        return False


class EarlyStopping:
    def __init__(self, patience=10, monitor="val_loss", mode="min", min_delta=0.0,
                 restore_best_weights=False, model=None, checkpoint_path=None):
        self.monitor = MetricMonitor(monitor, mode, min_delta)
        self.patience, self.wait, self.restore_best_weights, self.model, self.checkpoint_path = patience, 0, restore_best_weights, model, checkpoint_path

    def check(self, logs, epoch=None):
        if self.monitor.update(self.monitor.get_value(logs)):
            self.wait = 0
            return False
        self.wait += 1
        if self.wait >= self.patience:
            if self.restore_best_weights and self.model and self.checkpoint_path and os.path.exists(self.checkpoint_path):
                self.model.load_weights(self.checkpoint_path)
                print(f"Restored best weights from {self.checkpoint_path} before stopping.")
            return True
        return False


class ModelCheckpoint:
    def __init__(self, filepath, monitor="val_loss", mode="min", save_best_only=True, min_delta=0.0):
        self.filepath, self.save_best_only = filepath, save_best_only
        self.monitor = MetricMonitor(monitor, mode, min_delta)

    def save(self, model, logs):
        if not self.save_best_only or self.monitor.update(self.monitor.get_value(logs)):
            dirname = os.path.dirname(self.filepath)
            if dirname: os.makedirs(dirname, exist_ok=True)
            model.save_weights(self.filepath)
            return True
        return False


class ReduceLROnPlateau:
    def __init__(self, optimizer, model=None, factor=0.5, patience=5, monitor="val_loss",
                 mode="min", min_lr=1e-6, min_delta=0.0, restore_best_weights=False, checkpoint_path=None):
        self.optimizer, self.model, self.factor, self.patience, self.min_lr = optimizer, model, factor, patience, min_lr
        self.wait, self.restore_best_weights, self.checkpoint_path = 0, restore_best_weights, checkpoint_path
        self.monitor = MetricMonitor(monitor, mode, min_delta)

    def on_epoch_end(self, logs):
        if self.monitor.update(self.monitor.get_value(logs)):
            self.wait = 0
            return
        self.wait += 1
        if self.wait >= self.patience:
            old_lr = float(tf.keras.backend.get_value(self.optimizer.learning_rate))
            new_lr = max(old_lr * self.factor, self.min_lr)
            if new_lr < old_lr:
                self.optimizer.learning_rate.assign(new_lr)
                print(f"\nLearning rate reduced from {old_lr:.8f} to {new_lr:.8f}")
                if self.restore_best_weights and self.model and self.checkpoint_path and os.path.exists(self.checkpoint_path):
                    self.model.load_weights(self.checkpoint_path)
                    print(f"Restored best weights from {self.checkpoint_path}")
            self.wait = 0


class CosineAnnealing:
    def __init__(self, optimizer, t_max=100, eta_min=1e-6):
        import math
        self.optimizer, self.t_max, self.eta_min = optimizer, t_max, eta_min
        self.initial_lr = float(tf.keras.backend.get_value(self.optimizer.learning_rate))
        self.current_epoch = 0

    def on_epoch_end(self, logs=None):
        import math
        self.current_epoch += 1
        new_lr = self.eta_min if self.current_epoch > self.t_max else self.eta_min + (self.initial_lr - self.eta_min) * (0.5 * (1 + math.cos(math.pi * self.current_epoch / self.t_max)))
        self.optimizer.learning_rate.assign(new_lr)



class WandbLogger:
    def __init__(self, model=None, val_x=None, log_weights_freq=10,
                 classes=None, aggregate_plot_freq=1):
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            self.wandb = None
        self.model = model
        self.val_x = val_x
        self.log_weights_freq = log_weights_freq
        self.classes = classes or []
        self.aggregate_plot_freq = aggregate_plot_freq
        self.metric_history = {
            "epoch": [],
            "macro_f1": [],
            "weighted_f1": [],
            "male_f1": [],
            "female_f1": [],
            "class_f1": {class_name: [] for class_name in self.classes},
            "class_precision": {class_name: [] for class_name in self.classes},
            "class_recall": {class_name: [] for class_name in self.classes},
        }

    def _append_aggregate_history(self, logs, epoch):
        self.metric_history["epoch"].append(int(epoch) + 1)
        self.metric_history["macro_f1"].append(float(logs.get("val_macro_f1", 0.0)))
        self.metric_history["weighted_f1"].append(float(logs.get("val_weighted_f1", 0.0)))
        self.metric_history["male_f1"].append(float(logs.get("val_male_f1", 0.0)))
        self.metric_history["female_f1"].append(float(logs.get("val_female_f1", 0.0)))

        for class_name in self.classes:
            self.metric_history["class_f1"][class_name].append(
                float(logs.get(f"val_class_f1/{class_name}", 0.0))
            )
            self.metric_history["class_precision"][class_name].append(
                float(logs.get(f"val_class_precision/{class_name}", 0.0))
            )
            self.metric_history["class_recall"][class_name].append(
                float(logs.get(f"val_class_recall/{class_name}", 0.0))
            )

    def _add_colored_history_plot(self, log_dict, key, title, series_dict):
            if not self.metric_history["epoch"] or not series_dict:
                return

            try:
                import matplotlib.pyplot as plt
                import numpy as np
            except ImportError:
                return

            epochs = self.metric_history["epoch"]
            fig, ax = plt.subplots(figsize=(12, 6))

            num_series = len(series_dict)

            # 1. Color Strategy: Use discrete qualitative colors for up to 20 lines
            if num_series <= 20:
                # tab20 has exactly 20 highly distinct colors designed not to overlap
                colors = plt.cm.tab20.colors
            else:
                # Fallback for >20: 'turbo' is a perceptually uniform, wide-spectrum map
                colors = plt.cm.turbo(np.linspace(0, 1, num_series))

            plotted = 0
            for idx, (name, values) in enumerate(series_dict.items()):
                if len(values) != len(epochs):
                    continue
                ax.plot(
                    epochs,
                    values,
                    label=name,
                    color=colors[idx % len(colors)],
                    linestyle="-",  # 2. Force solid lines for all features
                    linewidth=1.8,
                    alpha=0.95,
                )
                plotted += 1

            if plotted == 0:
                plt.close(fig)
                return

            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.set_ylabel("score")
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, alpha=0.25)

            # 3. Legend formatting: Split into 2 columns if there are many features
            # so the legend doesn't run off the bottom of the W&B image.
            ncol = 2 if plotted > 10 else 1
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=ncol)

            fig.tight_layout()
            log_dict[key] = self.wandb.Image(fig)
            plt.close(fig)

    def _add_aggregate_plots(self, log_dict, epoch):
        if not self.classes or self.aggregate_plot_freq <= 0:
            return
        if epoch % self.aggregate_plot_freq != 0:
            return

        self._add_colored_history_plot(
            log_dict,
            "val/per_class_f1_lines_colored",
            "Validation F1 by class",
            self.metric_history["class_f1"],
        )
        self._add_colored_history_plot(
            log_dict,
            "val/per_class_precision_lines_colored",
            "Validation precision by class",
            self.metric_history["class_precision"],
        )
        self._add_colored_history_plot(
            log_dict,
            "val/per_class_recall_lines_colored",
            "Validation recall by class",
            self.metric_history["class_recall"],
        )
        self._add_colored_history_plot(
            log_dict,
            "val/group_f1_lines_colored",
            "Validation aggregate F1 comparison",
            {
                "macro_f1": self.metric_history["macro_f1"],
                "weighted_f1": self.metric_history["weighted_f1"],
                "male_f1": self.metric_history["male_f1"],
                "female_f1": self.metric_history["female_f1"],
            },
        )

    def on_epoch_end(self, logs):
        if self.wandb is not None and self.wandb.run is not None:
            epoch = logs.get("epoch", 0)
            log_dict = {**logs}
            self._append_aggregate_history(logs, epoch)
            self._add_aggregate_plots(log_dict, epoch)

            # Periodically log weights, biases, and activations as histograms to WandB
            if self.model is not None and self.log_weights_freq > 0 and (epoch % self.log_weights_freq == 0):
                # 1. Log weights and biases
                try:
                    for layer in self.model.layers:
                        weights = layer.get_weights()
                        if len(weights) > 0:
                            # Log kernel weights
                            log_dict[f"weights/{layer.name}"] = self.wandb.Histogram(weights[0])
                            # Log biases if they exist
                            if len(weights) > 1:
                                log_dict[f"biases/{layer.name}"] = self.wandb.Histogram(weights[1])
                except Exception as e:
                    print(f"Warning: Failed to log weights to WandB: {e}")

                # 2. Log layer activations (outputs of Conv1D and Dense layers)
                if self.val_x is not None:
                    try:
                        # Find layers we want to track (Conv1D and Dense)
                        target_layers = []
                        for layer in self.model.layers:
                            cls_name = layer.__class__.__name__.lower()
                            if "conv1d" in cls_name or "dense" in cls_name:
                                target_layers.append(layer)

                        if target_layers:
                            # Create an intermediate model that outputs activations
                            import tensorflow as tf
                            activation_model = tf.keras.Model(inputs=self.model.input, outputs=[l.output for l in target_layers])
                            activations = activation_model(self.val_x, training=False)

                            # Keras returns a single tensor instead of a list if there is only 1 layer
                            if not isinstance(activations, list):
                                activations = [activations]

                            for layer, act in zip(target_layers, activations):
                                log_dict[f"activations/{layer.name}"] = self.wandb.Histogram(act.numpy())
                    except Exception as e:
                        print(f"Warning: Failed to log activations to WandB: {e}")

            self.wandb.log(log_dict)


class CallbackFactory:
    @staticmethod
    def get_callbacks(config: dict, optimizer, model, model_save_path, val_x=None):
        cb_config = config.get("callbacks", {})
        callbacks = {}

        if "early_stopping" in cb_config:
            cfg = cb_config["early_stopping"]
            if cfg is not None:
                callbacks["early_stopping"] = EarlyStopping(
                    patience=cfg.get("patience", 10),
                    monitor=cfg.get("monitor", "val_loss"),
                    mode=cfg.get("mode", "min"),
                    min_delta=float(cfg.get("min_delta", 0.0)),
                    restore_best_weights=cfg.get("restore_best_weights", False),
                    model=model,
                    checkpoint_path=model_save_path
                )

        if "model_checkpoint" in cb_config:
            cfg = cb_config["model_checkpoint"]
            if cfg is not None:
                callbacks["model_checkpoint"] = ModelCheckpoint(
                    filepath=model_save_path,
                    monitor=cfg.get("monitor", "val_loss"),
                    mode=cfg.get("mode", "min"),
                    save_best_only=cfg.get("save_best_only", True),
                    min_delta=float(cfg.get("min_delta", 0.0)),
                )

        if "reduce_lr_on_plateau" in cb_config:
            cfg = cb_config["reduce_lr_on_plateau"]
            if cfg is not None:
                callbacks["reduce_lr_on_plateau"] = ReduceLROnPlateau(
                    optimizer=optimizer,
                    model=model,
                    factor=cfg.get("factor", 0.5),
                    patience=cfg.get("patience", 5),
                    monitor=cfg.get("monitor", "val_loss"),
                    mode=cfg.get("mode", "min"),
                    min_lr=float(cfg.get("min_lr", 1e-6)),
                    min_delta=float(cfg.get("min_delta", 0.0)),
                    restore_best_weights=cfg.get("restore_best_weights", False),
                    checkpoint_path=model_save_path
                )

        if "cosine_annealing" in cb_config:
            cfg = cb_config["cosine_annealing"]
            if cfg is not None:
                callbacks["cosine_annealing"] = CosineAnnealing(
                    optimizer=optimizer,
                    t_max=cfg.get("t_max", 100),
                    eta_min=float(cfg.get("eta_min", 1e-6))
                )

        if config.get("wandb", {}).get("enabled", False):
            wandb_cfg = config.get("wandb", {})
            freq = wandb_cfg.get("log_weights_freq", 10)
            aggregate_freq = int(wandb_cfg.get("aggregate_plot_freq", 1))
            callbacks["wandb_logger"] = WandbLogger(
                model=model,
                val_x=val_x,
                log_weights_freq=freq,
                classes=config.get("classes", []),
                aggregate_plot_freq=aggregate_freq,
            )

        return callbacks
