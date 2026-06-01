import os
import tensorflow as tf

class EarlyStopping:
    def __init__(self, patience=10, monitor='val_loss', mode='min'):
        self.patience = patience
        
        # Ensure monitors is a list
        if isinstance(monitor, (list, tuple)):
            self.monitors = list(monitor)
        else:
            self.monitors = [monitor]
            
        # Ensure modes is a list of same length
        if isinstance(mode, (list, tuple)):
            self.modes = list(mode)
        else:
            self.modes = [mode] * len(self.monitors)

        if len(self.modes) != len(self.monitors):
            raise ValueError(f"EarlyStopping mode length ({len(self.modes)}) must match monitor length ({len(self.monitors)}). "
                             f"Monitors: {self.monitors}, Modes: {self.modes}")

        self.wait = 0
        self.best = {}
        for monitor_name, mode_name in zip(self.monitors, self.modes):
            self.best[monitor_name] = float('inf') if mode_name == 'min' else float('-inf')
        
        self.stopped_epoch = 0

    def _is_improved(self, current_val, best_val, mode):
        if mode == 'min':
            return current_val < best_val
        return current_val > best_val

    def check(self, current_val):
        # Convert single value to dict if needed
        if not isinstance(current_val, dict):
            if len(self.monitors) != 1:
                raise ValueError("Current value must be a dict when monitoring multiple metrics")
            current_val = {self.monitors[0]: current_val}

        improved_any = False
        for monitor_name, mode_name in zip(self.monitors, self.modes):
            if monitor_name not in current_val:
                # If a monitored metric is missing, we skip it but don't fail yet
                continue
            
            current_metric = current_val[monitor_name]
            if self._is_improved(current_metric, self.best[monitor_name], mode_name):
                self.best[monitor_name] = current_metric
                improved_any = True

        if improved_any:
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

        if isinstance(monitor, (list, tuple)):
            self.monitors = list(monitor)
        else:
            self.monitors = [monitor]

        if isinstance(mode, (list, tuple)):
            self.modes = list(mode)
        else:
            self.modes = [mode] * len(self.monitors)

        if len(self.modes) != len(self.monitors):
            raise ValueError(f"ModelCheckpoint mode length ({len(self.modes)}) must match monitor length ({len(self.monitors)})")

        self.best = {}
        for monitor_name, mode_name in zip(self.monitors, self.modes):
            self.best[monitor_name] = float('inf') if mode_name == 'min' else float('-inf')

    def _is_improved(self, current_val, best_val, mode):
        if mode == 'min':
            return current_val < best_val
        return current_val > best_val

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
                continue
            
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
        
        if isinstance(monitor, (list, tuple)):
            self.monitors = list(monitor)
        else:
            self.monitors = [monitor]

        if isinstance(mode, (list, tuple)):
            self.modes = list(mode)
        else:
            self.modes = [mode] * len(self.monitors)

        if len(self.modes) != len(self.monitors):
            raise ValueError(f"ReduceLROnPlateau mode length ({len(self.modes)}) must match monitor length ({len(self.monitors)})")

        self.min_lr = min_lr
        self.wait = 0
        self.best = {}
        for monitor_name, mode_name in zip(self.monitors, self.modes):
            self.best[monitor_name] = float('inf') if mode_name == 'min' else float('-inf')

    def _is_improved(self, current_val, best_val, mode):
        if mode == 'min':
            return current_val < best_val
        return current_val > best_val

    def on_epoch_end(self, current_val):
        if not isinstance(current_val, dict):
            if len(self.monitors) != 1:
                raise ValueError("Current value must be a dict when monitoring multiple metrics")
            current_val = {self.monitors[0]: current_val}

        improved_any = False
        for monitor_name, mode_name in zip(self.monitors, self.modes):
            if monitor_name not in current_val:
                continue
            
            current_metric = current_val[monitor_name]
            if self._is_improved(current_metric, self.best[monitor_name], mode_name):
                self.best[monitor_name] = current_metric
                improved_any = True

        if improved_any:
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
            # Support both singular and plural keys
            monitor = cfg.get('monitor') or cfg.get('monitors') or 'val_loss'
            callbacks['early_stopping'] = EarlyStopping(
                patience=cfg.get('patience', 10),
                monitor=monitor,
                mode=cfg.get('mode', 'min')
            )
            
        if 'model_checkpoint' in cb_config:
            cfg = cb_config['model_checkpoint']
            monitor = cfg.get('monitor') or cfg.get('monitors') or 'val_loss'
            callbacks['model_checkpoint'] = ModelCheckpoint(
                filepath=model_save_path,
                monitor=monitor,
                mode=cfg.get('mode', 'min'),
                save_best_only=cfg.get('save_best_only', True)
            )
            
        if 'reduce_lr_on_plateau' in cb_config:
            cfg = cb_config['reduce_lr_on_plateau']
            monitor = cfg.get('monitor') or cfg.get('monitors') or 'val_loss'
            callbacks['reduce_lr_on_plateau'] = ReduceLROnPlateau(
                optimizer=optimizer,
                factor=cfg.get('factor', 0.5),
                patience=cfg.get('patience', 5),
                monitor=monitor,
                mode=cfg.get('mode', 'min'),
                min_lr=float(cfg.get('min_lr', 1e-6))
            )
            
        return callbacks
