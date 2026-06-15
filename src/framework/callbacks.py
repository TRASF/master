import os
import tensorflow as tf


class MetricMonitor:
    def __init__(self, monitor="val_loss", mode="min", min_delta=0.0):
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta

        if mode == "min":
            self.best = float("inf")
        elif mode == "max":
            self.best = float("-inf")
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def get_value(self, logs):
        if isinstance(logs, dict):
            if self.monitor not in logs:
                raise KeyError(f"Metric '{self.monitor}' not found in logs. Available: {list(logs.keys())}")
            return float(logs[self.monitor])

        return float(logs)

    def is_improved(self, current):
        if self.mode == "min":
            return current < self.best - self.min_delta
        return current > self.best + self.min_delta

    def update(self, current):
        if self.is_improved(current):
            self.best = current
            return True
        return False


class EarlyStopping:
    def __init__(self, patience=10, monitor="val_loss", mode="min", min_delta=0.0,
                 restore_best_weights=False, model=None, checkpoint_path=None):
        self.monitor = MetricMonitor(monitor, mode, min_delta)
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.model = model
        self.checkpoint_path = checkpoint_path

    def check(self, logs, epoch=None):
        current = self.monitor.get_value(logs)

        if self.monitor.update(current):
            self.wait = 0
            return False

        self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = epoch if epoch is not None else 0

            if self.restore_best_weights and self.model is not None and self.checkpoint_path is not None:
                if os.path.exists(self.checkpoint_path):
                    self.model.load_weights(self.checkpoint_path)
                    print(f"Restored best weights from {self.checkpoint_path} before stopping.")

            return True

        return False


class ModelCheckpoint:
    def __init__(self, filepath, monitor="val_loss", mode="min", save_best_only=True, min_delta=0.0):
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.monitor = MetricMonitor(monitor, mode, min_delta)

    def save(self, model, logs):
        if not self.save_best_only:
            self._save_weights(model)
            return True

        current = self.monitor.get_value(logs)

        if self.monitor.update(current):
            self._save_weights(model)
            return True

        return False

    def _save_weights(self, model):
        dirname = os.path.dirname(self.filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        model.save_weights(self.filepath)


class ReduceLROnPlateau:
    def __init__(self, optimizer, model=None, factor=0.5, patience=5, monitor="val_loss",
                 mode="min", min_lr=1e-6, min_delta=0.0, restore_best_weights=False, checkpoint_path=None):
        self.optimizer = optimizer
        self.model = model
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.wait = 0
        self.monitor = MetricMonitor(monitor, mode, min_delta)
        self.restore_best_weights = restore_best_weights
        self.checkpoint_path = checkpoint_path

    def on_epoch_end(self, logs):
        current = self.monitor.get_value(logs)

        if self.monitor.update(current):
            self.wait = 0
            return

        self.wait += 1

        if self.wait >= self.patience:
            old_lr = float(tf.keras.backend.get_value(self.optimizer.learning_rate))
            new_lr = max(old_lr * self.factor, self.min_lr)

            if new_lr < old_lr:
                self.optimizer.learning_rate.assign(new_lr)
                print(f"\nLearning rate reduced from {old_lr:.8f} to {new_lr:.8f}")

                if self.restore_best_weights and self.model is not None and self.checkpoint_path is not None:
                    if os.path.exists(self.checkpoint_path):
                        self.model.load_weights(self.checkpoint_path)
                        print(f"Restored best weights from {self.checkpoint_path}")
                    else:
                        print(f"Warning: Best weights file not found at {self.checkpoint_path}")

            self.wait = 0


class CallbackFactory:
    @staticmethod
    def get_callbacks(config: dict, optimizer, model, model_save_path):
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

        return callbacks
