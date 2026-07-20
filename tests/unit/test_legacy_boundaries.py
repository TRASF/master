"""Static contracts for the temporary legacy compatibility layer."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ROOT = ROOT / "src" / "wingbeat_ml"

LEGACY_WRAPPERS = (
    ROOT / "configs" / "mos_config.py",
    ROOT / "src" / "evaluation" / "evaluate.py",
    ROOT / "src" / "evaluation" / "report.py",
    ROOT / "src" / "framework" / "callbacks.py",
    ROOT / "src" / "framework" / "helper" / "augment.py",
    ROOT / "src" / "framework" / "helper" / "data_loader.py",
    ROOT / "src" / "framework" / "loss.py",
    ROOT / "src" / "framework" / "optimizer.py",
    ROOT / "src" / "framework" / "supervised" / "dataset.py",
    ROOT / "src" / "framework" / "supervised" / "train.py",
    ROOT / "src" / "framework" / "supervised" / "train_finetune.py",
    ROOT / "src" / "framework" / "supervised" / "train_linear_probe.py",
    ROOT / "src" / "framework" / "supervised" / "train_step.py",
    ROOT / "src" / "io" / "loader.py",
    ROOT / "src" / "quantization" / "tf_quantize.py",
)


class TestLegacyBoundaries(unittest.TestCase):
    def test_canonical_package_never_imports_legacy_modules(self):
        forbidden = (
            "src.framework",
            "src.evaluation",
            "src.quantization",
            "src.io",
            "configs.mos_config",
        )

        for path in CANONICAL_ROOT.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT)):
                for module_name in forbidden:
                    self.assertNotIn(module_name, source)

    def test_compatibility_wrappers_use_explicit_imports(self):
        for path in LEGACY_WRAPPERS:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertNotIn("import *", source)

    def test_compatibility_wrappers_only_depend_on_canonical_or_stdlib(self):
        for path in LEGACY_WRAPPERS:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            with self.subTest(path=path.relative_to(ROOT)):
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        self.assertTrue(
                            (node.module or "").startswith("wingbeat_ml."),
                            f"Unexpected import in {path}: {node.module}",
                        )
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            self.assertEqual(alias.name, "argparse")


if __name__ == "__main__":
    unittest.main()
