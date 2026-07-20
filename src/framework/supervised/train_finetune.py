"""Legacy wrapper for the canonical fine-tuning pipeline."""

from wingbeat_ml.pipelines.fine_tune import train_finetune

__all__ = ["train_finetune"]


if __name__ == "__main__":
    train_finetune()
