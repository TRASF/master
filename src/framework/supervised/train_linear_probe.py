"""Legacy wrapper for the canonical linear-probe pipeline."""

from wingbeat_ml.pipelines.linear_probe import train_linear_probe

__all__ = ["train_linear_probe"]


if __name__ == "__main__":
    train_linear_probe()
