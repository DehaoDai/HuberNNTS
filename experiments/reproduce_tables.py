"""Regenerate or check all six numerical-study tables from committed CSVs.

Examples:
  python -m experiments.reproduce_tables --check
  python -m experiments.reproduce_tables --output-dir results_tables
"""

import argparse
import csv
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
TABLES = (
    ("experiments.tables", "aggregate_summary.csv", "paper_tables.tex"),
    ("experiments.sensitivity_tables", "sensitivity_aggregate_summary.csv",
     "sensitivity_tables.tex"),
)


def render_all(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for module, csv_name, tex_name in TABLES:
        subprocess.run([
            sys.executable, "-m", module,
            "--input", str(ROOT / "reference" / csv_name),
            "--output", str(output_dir / tex_name),
            "--min-reps", "100",
        ], cwd=ROOT, check=True)


def check_all():
    with TemporaryDirectory(prefix="hubernnts-tables-") as temporary:
        output_dir = Path(temporary)
        render_all(output_dir)
        for _, _, tex_name in TABLES:
            expected = (ROOT / "reference" / tex_name).read_bytes()
            actual = (output_dir / tex_name).read_bytes()
            if actual != expected:
                raise SystemExit(
                    f"{tex_name} differs from the committed reference; "
                    "regenerate it or inspect the source CSV and renderer."
                )
    summaries = []
    for _, csv_name, _ in TABLES:
        with (ROOT / "reference" / csv_name).open(newline="") as handle:
            rows = csv.DictReader(handle)
            summaries.append({
                (row["case_id"], row["method"]): row for row in rows
            })
    shared = summaries[0].keys() & summaries[1].keys()
    if not shared or any(summaries[0][key] != summaries[1][key]
                         for key in shared):
        raise SystemExit("Shared reference cases disagree between CSV files.")
    print("All six tables match the committed TeX files exactly.")
    print(f"{len(shared)} shared summary rows agree across both CSV files.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--output-dir", type=Path)
    options = parser.parse_args()
    if options.check:
        check_all()
    else:
        render_all(options.output_dir.resolve())


if __name__ == "__main__":
    main()
