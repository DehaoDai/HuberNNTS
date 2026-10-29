"""Lightweight tests for the four-task real-data CLI (no network required)."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from experiments.real_data import TASKS, TARGETS, METHODS, metrics, summarize, prepare_task, fred_transform


class RealDataTests(unittest.TestCase):
    def test_only_four_tasks(self):
        self.assertEqual(TASKS, (
            "beijing_aotizhongxin", "beijing_changping", "fred_tb3", "fred_tb6",
        ))
        self.assertEqual(set(TARGETS), {"fred_tb3", "fred_tb6"})
        self.assertEqual(len(METHODS), 4)

    def test_removed_task_is_rejected_before_download(self):
        with self.assertRaisesRegex(ValueError, "Unknown real-data task"):
            prepare_task("fred_uemp", Path("/nonexistent"), 3, False)
        with self.assertRaisesRegex(ValueError, "Unknown real-data task"):
            prepare_task("appliances", Path("/nonexistent"), 3, False)

    def test_metrics_and_grouping(self):
        mae, rmse = metrics(np.array([1., 2.]), np.array([2., 2.]))
        self.assertAlmostEqual(mae, 0.5)
        self.assertAlmostEqual(rmse, 2 ** -0.5)
        rows = []
        for method in METHODS:
            for rep in range(2):
                value = 1.0 if method == "Huber-SPDNN" else 2.0
                rows.append({"task": TASKS[0], "rep": rep, "method": method, "mae": value, "rmse": value * 2})
        result = summarize(rows)
        winners = [x for x in result if x["joint_best"]]
        self.assertEqual([x["method"] for x in winners], ["Huber-SPDNN"])

    def test_fred_tcodes(self):
        x = pd.Series([3., 5., 8.])
        self.assertAlmostEqual(float(fred_transform(x, 2).iloc[2]), 3.)
        self.assertAlmostEqual(float(fred_transform(x, 1).iloc[0]), 3.)


if __name__ == "__main__":
    unittest.main()
