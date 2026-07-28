import unittest
import os
import tempfile
import yaml
from pathlib import Path
from wingbeat_ml.config import AppConfig
from wingbeat_ml.config.loader import (
    load_yaml,
    deep_merge,
    parse_override,
    apply_overrides,
    load_config,
    write_resolved_config,
    ResolvedConfig
)

class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        self.temp_files = []

    def tearDown(self):
        for path in self.temp_files:
            if os.path.exists(path):
                os.remove(path)

    def create_temp_yaml(self, data_str):
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        with open(path, "w") as f:
            f.write(data_str)
        self.temp_files.append(path)
        return path

    def test_load_yaml_valid(self):
        path = self.create_temp_yaml("key: value")
        self.assertEqual(load_yaml(path), {"key": "value"})

    def test_load_yaml_nonexistent(self):
        with self.assertRaises(FileNotFoundError):
            load_yaml("nonexistent_file.yaml")

    def test_load_yaml_empty(self):
        path = self.create_temp_yaml("")
        self.assertEqual(load_yaml(path), {})

    def test_load_yaml_invalid(self):
        path = self.create_temp_yaml("invalid: [yaml")
        with self.assertRaises(Exception):
            load_yaml(path)

    def test_reject_non_mapping_root(self):
        path = self.create_temp_yaml("- item1\n- item2")
        with self.assertRaises(ValueError):
            load_config(path)

    def test_deep_merge_basic(self):
        base = {"a": 1, "b": {"c": 2}}
        override = {"b": {"d": 3}, "e": 4}
        expected = {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}
        self.assertEqual(deep_merge(base, override), expected)

    def test_deep_merge_scalar_replacement(self):
        base = {"a": 1}
        override = {"a": 2}
        self.assertEqual(deep_merge(base, override), {"a": 2})

    def test_deep_merge_list_replacement(self):
        base = {"a": [1, 2]}
        override = {"a": [3, 4]}
        self.assertEqual(deep_merge(base, override), {"a": [3, 4]})

    def test_deep_merge_null_replacement(self):
        base = {"a": 1}
        override = {"a": None}
        self.assertEqual(deep_merge(base, override), {"a": None})

    def test_deep_merge_type_replacement(self):
        base = {"a": {"b": 1}}
        override = {"a": 2}
        self.assertEqual(deep_merge(base, override), {"a": 2})

    def test_source_dictionaries_not_mutated(self):
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        merged = deep_merge(base, override)
        
        merged["a"]["b"] = 99
        self.assertEqual(base["a"]["b"], 1)

    def test_exact_merge_precedence(self):
        base = self.create_temp_yaml("model: {id: mossong_plus}\ntraining_mode: pretrain\naudio: {sample_rate: 8000}\ntrain: {epochs: 10}\ndataset: {}")
        model = self.create_temp_yaml("model: {id: mossong_plus}")
        experiment = self.create_temp_yaml("train: {epochs: 50}")
        profile = self.create_temp_yaml("train: {batch_size: 64}")
        
        resolved = load_config(
            base_path=base,
            model_path=model,
            experiment_path=experiment,
            profile_path=profile,
            overrides=["train.epochs=100"]
        )
        self.assertIsInstance(resolved, AppConfig)
        self.assertEqual(resolved.train.epochs, 100)
        self.assertEqual(resolved.train.batch_size, 64)

    def test_parse_override_types(self):
        self.assertEqual(parse_override("k=123"), ("k", 123))
        self.assertEqual(parse_override("k=1.23"), ("k", 1.23))
        self.assertEqual(parse_override("k=true"), ("k", True))
        self.assertEqual(parse_override("k=null"), ("k", None))
        self.assertEqual(parse_override("k=hello"), ("k", "hello"))
        self.assertEqual(parse_override("k=[1, 2]"), ("k", [1, 2]))
        self.assertEqual(parse_override("k={a: 1}"), ("k", {"a": 1}))

    def test_parse_override_missing_equals(self):
        with self.assertRaises(ValueError):
            parse_override("invalid_override")

    def test_parse_override_empty_key(self):
        with self.assertRaises(ValueError):
            parse_override("=value")

    def test_write_resolved_config(self):
        base_path = self.create_temp_yaml(
            "model: {id: mossong_plus}\ntraining_mode: pretrain\naudio: {sample_rate: 8000}\ntrain: {epochs: 10}\ndataset: {}"
        )
        resolved = load_config(base_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_yaml = os.path.join(tmpdir, "resolved.yaml")
            write_resolved_config(resolved, out_yaml)
            self.assertTrue(os.path.exists(out_yaml))

if __name__ == "__main__":
    unittest.main()
