import unittest
from wingbeat_ml.config.schema import validate_config

class TestConfigSchema(unittest.TestCase):
    def get_valid_config(self):
        return {
            "model": {"id": "mossong_plus"},
            "training_mode": "pretrain",
            "audio": {"sample_rate": 8000, "duration": 0.3},
            "train": {"epochs": 10, "batch_size": 32, "seed": 42},
            "dataset": {"split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1}},
        }

    def test_valid_config(self):
        cfg = self.get_valid_config()
        # Should not raise any exception
        validate_config(cfg)

    def test_missing_required_sections(self):
        cfg = self.get_valid_config()
        cfg.pop("model")
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("Missing required top-level section: 'model'", str(ctx.exception))

    def test_invalid_model_id(self):
        cfg = self.get_valid_config()
        cfg["model"]["id"] = "invalid_model"
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("Invalid model ID: expected 'mossong_plus'", str(ctx.exception))

    def test_invalid_training_mode(self):
        cfg = self.get_valid_config()
        cfg["training_mode"] = "invalid_mode"
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("Invalid training mode", str(ctx.exception))

    def test_invalid_class_count(self):
        cfg = self.get_valid_config()
        cfg["num_classes"] = 10  # expects 11
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("Invalid num_classes: expected 11", str(ctx.exception))

    def test_invalid_sample_rate(self):
        cfg = self.get_valid_config()
        cfg["audio"]["sample_rate"] = -100
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("Invalid sample_rate", str(ctx.exception))

    def test_invalid_seed_type(self):
        cfg = self.get_valid_config()
        cfg["train"]["seed"] = "not_an_int"
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("Invalid train.seed type", str(ctx.exception))

    def test_invalid_batch_size(self):
        cfg = self.get_valid_config()
        cfg["train"]["batch_size"] = 0
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("Invalid train.batch_size", str(ctx.exception))

    def test_invalid_epochs(self):
        cfg = self.get_valid_config()
        cfg["train"]["epochs"] = -10
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("Invalid train.epochs", str(ctx.exception))

    def test_valid_linear_probe_and_finetune(self):
        cfg = self.get_valid_config()
        cfg["training_mode"] = "linear_probe"
        cfg["checkpoint"] = "dummy_checkpoint.h5"
        validate_config(cfg)
        
        cfg["training_mode"] = "fine_tune"
        validate_config(cfg)

    def test_duplicate_class(self):
        cfg = self.get_valid_config()
        cfg["classes"] = ["A"] * 11
        cfg["labels"] = {f"class_{i}": i for i in range(11)}
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("Class names must be unique", str(ctx.exception))

    def test_wrong_label_order(self):
        cfg = self.get_valid_config()
        cfg["labels"] = {"Ae_aegypti_Female": 5} # index 5 is An_dirus_Male
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("Invalid label index", str(ctx.exception))

    def test_invalid_segment_length(self):
        cfg = self.get_valid_config()
        cfg["audio"]["segment_length"] = -5
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("Invalid segment_length", str(ctx.exception))

    def test_invalid_overlap(self):
        cfg = self.get_valid_config()
        cfg["augment"] = {"segment_overlap": 1.5} # must be <= 1.0
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("Invalid segment_overlap", str(ctx.exception))

    def test_invalid_learning_rate(self):
        cfg = self.get_valid_config()
        cfg["optimizer"] = {"learning_rate": -0.01}
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("Invalid learning_rate", str(ctx.exception))

    def test_model_input_mismatch(self):
        cfg = self.get_valid_config()
        cfg["audio"]["segment_length"] = 2400
        cfg["model"]["input_shape"] = [1200, 1] # length doesn't match segment_length 2400
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("Model input length", str(ctx.exception))

    def test_ci_tracking_must_be_disabled(self):
        cfg = self.get_valid_config()
        cfg["dataset"]["train_dir"] = "tests/fixtures/audio_11class"
        cfg["wandb"] = {"enabled": True}
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("W&B tracking must be disabled in CI profile", str(ctx.exception))

    def test_secrets_rejected(self):
        cfg = self.get_valid_config()
        cfg["wandb"] = {"api_key": "some_sensitive_key"}
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("Secrets are not allowed in configuration file", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
