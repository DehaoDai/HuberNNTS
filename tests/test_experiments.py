import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.dgp import (
    DGP, active_coordinates, draw_x_series, exact_sparse_parameter_count,
    generate_training_series, make_exact_sparse_network, regression_function,
)
from experiments.network import ClippedNetwork
from experiments.run import cases_for
from experiments.tables import (
    load_summaries, render_active, render_dimension, render_sample_size,
    render_width,
)


ROOT = Path(__file__).resolve().parents[1]


class ExperimentTests(unittest.TestCase):
    def test_paper_matrix_and_fixed_b_support(self):
        cases = cases_for("paper")
        self.assertEqual(len(cases), 38)
        self.assertEqual(len({case.case_id for case in cases}), 38)
        self.assertEqual(set(active_coordinates("factor_sparse")), set(range(5)))
        self.assertEqual({case.sparsity for case in cases if case.family == "B"}, {4})
        self.assertEqual({case.beta for case in cases if case.family == "C"}, {0.7})

    def test_compact_serial_covariates(self):
        x = draw_x_series(1000, 64, 0.5, np.random.default_rng(123))
        self.assertGreaterEqual(float(x.min()), 0.0)
        self.assertLessEqual(float(x.max()), 1.0)
        self.assertGreater(np.corrcoef(x[:-1, 0], x[1:, 0])[0, 1], 0.25)

    def test_exact_sparse_comparator(self):
        x = draw_x_series(128, 64, 0.5, np.random.default_rng(321))
        for s in (4, 8, 16, 32):
            model = ClippedNetwork(64, 24, 2, 2.0, 2.0, 100.0)
            make_exact_sparse_network(model, s, 1.0)
            with torch.no_grad():
                fitted = model(torch.from_numpy(x)).flatten().numpy()
            truth = regression_function(x, "sparse_relu", s)
            self.assertLess(float(np.max(np.abs(fitted - truth))), 1e-5)
            nonzero = sum(int((p.abs() > 1e-10).sum()) for p in model.parameters())
            self.assertEqual(nonzero, exact_sparse_parameter_count(s))

    def test_heavy_tailed_series_is_generated(self):
        spec = DGP("sparse_relu", n=128, d=64)
        x, y = generate_training_series(spec, np.random.default_rng(99))
        self.assertEqual(x.shape, (128, 64))
        self.assertEqual(y.shape, (128,))
        self.assertTrue(np.isfinite(y).all())

    def test_reference_tables_match_pasted_values(self):
        index = load_summaries(ROOT / "reference/aggregate_summary.csv")
        sample = render_sample_size(index, 100)
        dimension = render_dimension(index, 100)
        active = render_active(index, 100)
        width = render_width(index, 100)
        self.assertIn(r"\textbf{0.0496(0.0015)}", sample)
        self.assertIn(r"\textbf{0.2301(0.0026)}", dimension)
        self.assertIn("0.0631(0.0025)", active)
        self.assertIn("0.1706(0.0022)", width)
        self.assertNotIn("DGP 2}", active)

    def test_table_requires_all_cases(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            render_sample_size({}, 100)


if __name__ == "__main__":
    unittest.main()
