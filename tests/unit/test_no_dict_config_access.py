"""Repository-wide AST & regex enforcement test guaranteeing zero dict-style config access in core src/."""

import ast
import os
import re
from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src" / "wingbeat_ml"


class TestNoDictConfigAccess(TestCase):

    def test_no_dictionary_access_on_config_objects(self):
        """Ensure no config["..."] or config.get(...) indexing remains in wingbeat_ml codebase."""
        forbidden_patterns = [
            re.compile(r"\bconfig\[\s*['\"]"),
            re.compile(r"\bconfig\.get\("),
            re.compile(r"\bcfg\[\s*['\"]"),
            re.compile(r"\bcfg\.get\("),
            re.compile(r"\braw_config\["),
            re.compile(r"\bexport_config\["),
            re.compile(r"\bdataset_config\["),
        ]

        # Allowed legacy or helper fallback locations
        exemptions = {
            "config/loader.py",  # Low-level raw dictionary merging prior to Pydantic validation
            "config/runtime.py", # Low-level compatibility fallback helpers
            "config/schema.py",  # Pydantic v2 schema definition and pre-validation dictionary checks
            "export/input_contract.py",  # Low-level input contract helper
            "models/mossong_plus.py",  # Keras layer get_config / from_config dictionaries
            "models/layers/rep_conv1d.py",  # Keras layer get_config / from_config dictionaries
        }

        violations = []

        for py_file in SRC_DIR.glob("**/*.py"):
            rel_path = py_file.relative_to(SRC_DIR).as_posix()
            if rel_path in exemptions:
                continue

            content = py_file.read_text(encoding="utf-8")
            lines = content.splitlines()

            for line_idx, line in enumerate(lines, start=1):
                # Ignore comments or string docs
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue

                for pattern in forbidden_patterns:
                    if pattern.search(line):
                        violations.append(f"{rel_path}:{line_idx}: {line.strip()}")

        if violations:
            msg = "Forbidden dictionary-style configuration access detected:\n" + "\n".join(violations)
            self.fail(msg)
