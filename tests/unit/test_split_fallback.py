"""Regression test for undersized stratified partitions."""

import unittest
import warnings

import numpy as np

from wingbeat_ml.data.dataset import SupervisedDataset


class TestUndersizedStratifiedPartition(unittest.TestCase):
    def test_falls_back_when_partition_is_smaller_than_class_count(self):
        paths = np.array(
            [
                f"/data/class_{label}/sample_{sample}.wav"
                for label in range(11)
                for sample in range(4)
            ],
            dtype=object,
        )
        labels = np.repeat(np.arange(11, dtype=np.int32), 4)

        dataset = object.__new__(SupervisedDataset)
        dataset.seed = 45

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            train_paths, eval_paths, train_labels, eval_labels = (
                dataset._split_paths(
                    paths,
                    labels,
                    test_size=0.2,
                    split_name="training/evaluation",
                )
            )

        self.assertEqual(len(train_paths), 35)
        self.assertEqual(len(eval_paths), 9)
        self.assertEqual(len(train_labels) + len(eval_labels), 44)
        self.assertTrue(
            any("cannot stratify" in str(item.message) for item in captured)
        )


if __name__ == "__main__":
    unittest.main()
