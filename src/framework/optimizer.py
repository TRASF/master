"""Legacy optimizer API; new code should use ``build_optimizer``."""

from wingbeat_ml.training.optimizers import build_optimizer


class OptimizerFactory:
    """Compatibility shim for the former class-based API."""

    @staticmethod
    def get_optimizer(config):
        return build_optimizer(config.get("optimizer", config))

__all__ = ["OptimizerFactory"]
