import os
import unittest
import numpy as np
import json

class TestBaselineNegative(unittest.TestCase):
    def test_negative_file_discovery_mismatch(self):
        # Load baseline
        baseline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../parity/baselines"))
        with open(os.path.join(baseline_dir, "file_discovery.json"), "r") as f:
            baseline = json.load(f)

        # Intentional mismatch: modify files list
        modified_files = list(baseline["files"])
        if modified_files:
            modified_files[0] = "corrupted_filename_test.wav"

        # Verify that comparing different lists raises AssertionError
        with self.assertRaises(AssertionError):
            self.assertEqual(modified_files, baseline["files"])

    def test_negative_preprocessing_mismatch(self):
        # Load baseline
        baseline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../parity/baselines"))
        baseline = np.load(os.path.join(baseline_dir, "preprocessing_outputs.npz"))

        # Select first key and apply intentional deviation
        keys = list(baseline.keys())
        if keys:
            key = keys[0]
            original = baseline[key]
            corrupted = original + 0.1  # deliberate difference

            # Verify that the parity check fails under the strict tolerances
            with self.assertRaises(AssertionError):
                np.testing.assert_allclose(corrupted, original, rtol=1e-6, atol=1e-7)

if __name__ == "__main__":
    unittest.main()
