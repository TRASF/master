"""Legacy wrapper for the canonical pretraining pipeline."""

from wingbeat_ml.pipelines.pretrain import train_supervised

__all__ = ["train_supervised"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--defaults_path",
        default="configs/defaults.yaml",
    )
    parser.add_argument(
        "--model_cfg_path",
        default="configs/model.yaml",
    )
    args, _ = parser.parse_known_args()

    train_supervised(
        defaults_path=args.defaults_path,
        model_cfg_path=args.model_cfg_path,
    )
