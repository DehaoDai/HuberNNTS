import unittest

from experiments.run import METHODS, cases_for
from experiments.sensitivity_tables import render_dependence, render_tail


class SensitivityTests(unittest.TestCase):
    def test_full_factorial_cases(self):
        dependence = cases_for("dependence")
        self.assertEqual(len(dependence), 9)
        self.assertEqual(
            {(case.rho_x, case.rho_e) for case in dependence},
            {(x, e) for x in (0, 0.5, 0.8) for e in (0, 0.5, 0.8)},
        )
        tail = cases_for("tail")
        self.assertEqual(len(tail), 6)
        self.assertEqual(
            {(case.alpha, case.beta) for case in tail},
            {(alpha, beta) for alpha in (1.35, 1.5, 1.8)
             for beta in (0, 0.7)},
        )
        self.assertEqual({case.tau for case in tail}, {14.0})
        self.assertEqual(len({case.case_id for case in dependence + tail}), 15)

    def test_tables_require_complete_replicated_results(self):
        index = {}
        for case in cases_for("dependence") + cases_for("tail"):
            for i, method in enumerate(METHODS):
                index[(case.case_id, method)] = {
                    "method": method,
                    "reps": "2",
                    "mean_prediction_error": str(0.1 + i * 0.01),
                    "sem_prediction_error": "0.002",
                    "mean_population_target_bias_sq": "0.025",
                }
        self.assertEqual(render_dependence(index, 2).count("0.1000(0.0020)"), 9)
        tail_table = render_tail(index, 2)
        self.assertEqual(tail_table.count("0.0250"), 6)
        self.assertEqual(tail_table.count(r"\textbf{0.1000(0.0020)}"), 6)
        with self.assertRaisesRegex(ValueError, "Only 2 replications"):
            render_tail(index, 3)
        del index[(cases_for("dependence")[0].case_id, "LS-DNN")]
        with self.assertRaisesRegex(ValueError, "Missing LS-DNN"):
            render_dependence(index, 2)


if __name__ == "__main__":
    unittest.main()
