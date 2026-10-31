import unittest

from experiments.reproduce_tables import check_all


class TableReproducibilityTests(unittest.TestCase):
    def test_committed_tables_match_summaries(self):
        check_all()


if __name__ == "__main__":
    unittest.main()
