"""Contracts for the final installable-package boundary."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class TestDistributionBoundary(unittest.TestCase):
    def test_distribution_contains_only_the_canonical_package(self):
        project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('[tool.setuptools.packages.find]', project)
        self.assertIn('where = ["src"]', project)
        self.assertIn('include = ["wingbeat_ml*"]', project)
        self.assertIn('namespaces = false', project)

    def test_architecture_document_exists(self):
        architecture = ROOT / "docs" / "architecture.md"
        self.assertTrue(architecture.is_file())
        source = architecture.read_text(encoding="utf-8")
        self.assertIn("Canonical package", source)
        self.assertIn("Compatibility boundary", source)
        self.assertIn("WINGBEAT_RUNTIME_ROOT", source)

    def test_readme_links_to_architecture_document(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("docs/architecture.md", readme)


if __name__ == "__main__":
    unittest.main()
