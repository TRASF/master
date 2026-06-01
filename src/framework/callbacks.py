import os
import tensorflow as tf

class EarlyStopping:
    def __init__(self, patience=10, monitor='val_loss', mode='min'):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.wait = 0
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.stopped_epoch = 0

    def check(self, current_val):
        if self.mode == 'min':
            improved = current_val < self.best
        else:
            improved = current_val > self.best

        if improved:
            self.best = current_val
            self.wait = 0
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
            return False

class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True):
        self.filepath = filepath
        self.save_best_only = save_best_only

        self.monitors = list(monitor) if isinstance(monitor, (list, tuple)) else [monitor]
        self.modes = list(mode) if isinstance(mode, (list, tuple)) else [mode] * len(self.monitors)

        if len(self.modes) != len(self.monitors):
            raise ValueError("ModelCheckpoint mode length must match monitor length")

        self.best = {}
        for monitor_name, mode_name in zip(self.monitors, self.modes):
            self.best[monitor_name] = float('inf') if mode_name == 'min' else float('-inf')

    def _is_improved(self, current_val, best_val, mode):
        return current_val < best_val if mode == 'min' else current_val > best_val

    def save(self, model, current_val):
        if not self.save_best_only:
            model.save_weights(self.filepath)
            return True

        if not isinstance(current_val, dict):
            if len(self.monitors) != 1:
                raise ValueError("Current value must be a dict when monitoring multiple metrics")
            current_val = {self.monitors[0]: current_val}

        improved_any = False
        for monitor_name, mode_name in zip(self.monitors, self.modes):
            if monitor_name not in current_val:
                raise KeyError(f"Missing monitored value '{monitor_name}' in current_val")
            current_metric = current_val[monitor_name]
            if self._is_improved(current_metric, self.best[monitor_name], mode_name):
                self.best[monitor_name] = current_metric
                improved_any = True

        if improved_any:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            model.save_weights(self.filepath)
            return True
        return False

class ReduceLROnPlateau:
    def __init__(self, optimizer, factor=0.5, patience=5, monitor='val_loss', mode='min', min_lr=1e-6):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_lr = min_lr
        self.wait = 0
        self.best = float('inf') if mode == 'min' else float('-inf')

    def on_epoch_end(self, current_val):
        if self.mode == 'min':
            improved = current_val < self.best
        else:
            improved = current_val > self.best

        if improved:
            self.best = current_val
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(self.optimizer.learning_rate)
                new_lr = max(old_lr * self.factor, self.min_lr)
                if old_lr > new_lr:
                    self.optimizer.learning_rate.assign(new_lr)
                    print(f"\nLearning rate reduced to {new_lr:.8f}")
                self.wait = 0

class CallbackFactory:
    @staticmethod
    def get_callbacks(config: dict, optimizer, model_save_path):
        cb_config = config.get('callbacks', {})
        
        callbacks = {}
        
        if 'early_stopping' in cb_config:
            cfg = cb_config['early_stopping']
            callbacks['early_stopping'] = EarlyStopping(
                patience=cfg.get('patience', 10),
                monitor=cfg.get('monitor', 'val_loss'),
                mode=cfg.get('mode', 'min')
            )
            
        if 'model_checkpoint' in cb_config:
            cfg = cb_config['model_checkpoint']
            callbacks['model_checkpoint'] = ModelCheckpoint(
                filepath=model_save_path,
                monitor=cfg.get('monitor', 'val_loss'),
                mode=cfg.get('mode', 'min'),
                save_best_only=cfg.get('save_best_only', True)
            )
            
        if 'reduce_lr_on_plateau' in cb_config:
            cfg = cb_config['reduce_lr_on_plateau']
            callbacks['reduce_lr_on_plateau'] = ReduceLROnPlateau(
                optimizer=optimizer,
                factor=cfg.get('factor', 0.5),
                patience=cfg.get('patience', 5),
                monitor=cfg.get('monitor', 'val_loss'),
                mode=cfg.get('mode', 'min'),
                min_lr=float(cfg.get('min_lr', 1e-6))
            )
            
        return callbacks
