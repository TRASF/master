"""Unit tests for Pydantic v2 AppConfig, validation rules, and merge precedence."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from pydantic import ValidationError
import yaml

from wingbeat_ml.config import AppConfig, generate_json_schema, load_config, validate_config


class TestPydanticAppConfig(TestCase):

    def test_default_instantiation_uses_operational_defaults(self):
        config = AppConfig()
        self.assertEqual(config.train.epochs, 1000)
        self.assertEqual(config.train.batch_size, 128)
        self.assertEqual(config.audio.sample_rate, 8000)
        self.assertEqual(config.audio.segment_length, 2400)
        self.assertFalse(config.adabn.enabled)
        self.assertEqual(config.adabn.mode, "adhoc")
        self.assertEqual(config.performance.precision, "float32")
        self.assertEqual(config.logging.console, "normal")
        self.assertEqual(config.num_classes, 11)

    def test_extra_forbidden_keys_raise_validation_error(self):
        with self.assertRaises(ValidationError):
            AppConfig.model_validate({"extra_unrecognized_key": "invalid_value"})

        with self.assertRaises(ValidationError):
            AppConfig.model_validate({"train": {"epochs": 100, "unknown_field": True}})

    def test_immutability_frozen_models(self):
        config = AppConfig()
        with self.assertRaises((TypeError, ValidationError)):
            config.train.epochs = 500  # Frozen model assignment raises error

    def test_merge_precedence_defaults_model_experiment_profile_cli(self):
        with TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            base_file = tmppath / "base.yaml"
            model_file = tmppath / "model.yaml"
            exp_file = tmppath / "experiment.yaml"
            profile_file = tmppath / "profile.yaml"

            base_file.write_text(yaml.dump({
                "train": {"epochs": 100, "batch_size": 32},
                "audio": {"sample_rate": 8000},
            }))
            model_file.write_text(yaml.dump({
                "model": {"id": "custom_model", "output_activation": "softmax"},
            }))
            exp_file.write_text(yaml.dump({
                "train": {"epochs": 200},
                "experiment_name": "exp_test",
            }))
            profile_file.write_text(yaml.dump({
                "train": {"batch_size": 64},
                "performance": {"precision": "mixed_float16"},
            }))

            resolved = load_config(
                base_path=base_file,
                model_path=model_file,
                experiment_path=exp_file,
                profile_path=profile_file,
                overrides=["train.epochs=300"],
            )

            # Precedence checks: CLI override (300) > Experiment (200) > Base (100)
            self.assertEqual(resolved.train.epochs, 300)
            # Profile (64) > Base (32)
            self.assertEqual(resolved.train.batch_size, 64)
            # Model config
            self.assertEqual(resolved.model.id, "custom_model")
            self.assertEqual(resolved.model.output_activation, "softmax")
            # Profile config
            self.assertEqual(resolved.performance.precision, "mixed_float16")

    def test_single_point_validation_and_json_schema(self):
        raw_dict = {
            "train": {"epochs": 50},
            "audio": {"sample_rate": 16000, "segment_length": 4800},
            "dataset": {"train_dir": "custom/path"},
        }
        validated = validate_config(raw_dict)
        self.assertIsInstance(validated, AppConfig)
        self.assertEqual(validated.train.epochs, 50)
        self.assertEqual(validated.audio.sample_rate, 16000)

        schema = generate_json_schema()
        self.assertIn("properties", schema)
        self.assertIn("train", schema["properties"])
        self.assertIn("audio", schema["properties"])
