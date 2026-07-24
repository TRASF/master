"""Minimal contract every training strategy must satisfy."""


class TrainingStrategy:
    def setup(self, context, datasets):
        pass

    def train_epoch(self, datasets, *, epoch):
        raise NotImplementedError

    def validate_epoch(self, dataset, *, epoch):
        raise NotImplementedError

    def on_epoch_end(self, epoch, logs):
        pass

    def checkpoint_objects(self):
        return {}

    def finalize(self, context, datasets):
        pass


__all__ = ["TrainingStrategy"]
