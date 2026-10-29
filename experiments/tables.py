"""Render the four Numerical Studies tables from replication summaries.

Usage: python -m experiments.tables --input results/aggregate_summary.csv
"""

import argparse
import csv
from pathlib import Path

from .run import Case, METHODS


DISPLAY = {
    "Oracle-support-Huber": "Oracle-Huber",
    **{method: method for method in METHODS if method != "Oracle-support-Huber"},
}
FULL_METHODS = METHODS[:4]


def load_summaries(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    index = {}
    for row in rows:
        key = (row["case_id"], row["method"])
        if key in index:
            raise ValueError(f"Duplicate summary row: {key}")
        index[key] = row
    return index


def cell(row, bold=False):
    value = (f'{float(row["mean_prediction_error"]):.4f}'
             f'({float(row["sem_prediction_error"]):.4f})')
    return rf"\textbf{{{value}}}" if bold else value


def method_cells(index, case, methods, min_reps):
    found = []
    for method in methods:
        key = (case.case_id, method)
        if key not in index:
            raise ValueError(f"Missing {method} for {case.case_id}")
        row = index[key]
        if int(row["reps"]) < min_reps:
            raise ValueError(f"Only {row['reps']} replications for {key}; need {min_reps}")
        found.append(row)
    best = min(
        (float(index[(case.case_id, method)]["mean_prediction_error"]), method)
        for method in FULL_METHODS
    )[1] if set(FULL_METHODS).issubset(methods) else None
    return [cell(row, row["method"] == best) for row in found]


def table(caption, label, columns, header, body):
    lines = [
        r"\begin{table}[!ht]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{adjustbox}{max width=\linewidth}",
        rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}", r"\toprule",
        header + r"\\", r"\midrule",
        *body,
        r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}",
    ]
    return "\n".join(lines)


def render_sample_size(index, min_reps):
    methods = METHODS
    body = []
    for family in "ABC":
        if body:
            body.append(r"\midrule")
        cases = [Case(family, n, 256, beta=0.7 if family == "C" else 0.0)
                 for n in (400, 800, 1600, 3200)]
        for i, case in enumerate(cases):
            prefix = rf"\multirow{{4}}{{*}}{{DGP {ord(family)-64}}}" if i == 0 else ""
            body.append(" & ".join([prefix, str(case.n),
                                    *method_cells(index, case, methods, min_reps)]) + r"\\")
    return table(
        r"Sample-size sweeps for DGPs 1--3 at $d=256$ and "
        r"$(\rho_X,\rho_\varepsilon)=(0.5,0.5)$. "
        r"Boldface marks the best full-input estimator; Oracle-Huber is a benchmark.",
        "tab:sample-size-all", "lrccccc",
        "Data & $n$ & " + " & ".join(DISPLAY[m] for m in methods), body,
    )


def render_dimension(index, min_reps):
    body = []
    for family in "ABC":
        if body:
            body.append(r"\midrule")
        cases = [Case(family, 800, d, beta=0.7 if family == "C" else 0.0)
                 for d in (64, 256, 1024, 2048)]
        for i, case in enumerate(cases):
            prefix = rf"\multirow{{4}}{{*}}{{DGP {ord(family)-64}}}" if i == 0 else ""
            body.append(" & ".join([prefix, str(case.d),
                                    *method_cells(index, case, METHODS, min_reps)]) + r"\\")
    return table(
        r"Dimension sweeps for DGPs 1--3 at $n=800$ and "
        r"$(\rho_X,\rho_\varepsilon)=(0.5,0.5)$. "
        r"Boldface marks the best full-input estimator; Oracle-Huber is a benchmark.",
        "tab:dimension-abc", "lrccccc",
        "Data & $d$ & " + " & ".join(DISPLAY[m] for m in METHODS), body,
    )


def render_active(index, min_reps):
    methods = ("Huber-DNN", "Huber-SPDNN", "Oracle-support-Huber")
    body = []
    for family in "AC":
        if body:
            body.append(r"\midrule")
        cases = [Case(family, n, 1024, sparsity=s,
                      beta=0.7 if family == "C" else 0.0)
                 for n in (800, 1600) for s in (4, 8, 16, 32)]
        for i, case in enumerate(cases):
            prefix = rf"\multirow{{8}}{{*}}{{DGP {ord(family)-64}}}" if i == 0 else ""
            body.append(" & ".join([prefix, str(case.n), str(case.sparsity),
                                    str(5 * case.sparsity // 2),
                                    *method_cells(index, case, methods, min_reps)]) + r"\\")
    return table(
        r"Active-coordinate sweeps for DGP 1 and 3 at $d=1024$. "
        r"The exact sparse representation has $5s/2$ nonzero parameters. "
        r"DGP 2 has five fixed active inputs and no $s$ sweep.",
        "tab:AC-s", "lrrrccc",
        "Data & $n$ & $s$ & $5s/2$ & "
        + " & ".join(DISPLAY[m] for m in methods), body,
    )


def render_width(index, min_reps):
    methods = ("Huber-DNN", "Huber-SPDNN", "Oracle-support-Huber")
    body = []
    for width in (8, 16, 24, 32):
        case = Case("B", 1600, 256, width=width)
        body.append(" & ".join([str(width),
                                *method_cells(index, case, methods, min_reps)]) + r"\\")
    return table(
        r"Width sweep for DGP 2 under $n=1600$, $d=256$. "
        r"Width changes both capacity and fitted parameter count.",
        "tab:B-width", "rccc",
        "Width & " + " & ".join(DISPLAY[m] for m in methods), body,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/aggregate_summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/paper_tables.tex"))
    parser.add_argument("--min-reps", type=int, default=100)
    args = parser.parse_args()
    if args.min_reps < 1:
        parser.error("--min-reps must be positive")
    index = load_summaries(args.input)
    rendered = "\n\n".join([
        render_sample_size(index, args.min_reps),
        render_dimension(index, args.min_reps),
        render_active(index, args.min_reps),
        render_width(index, args.min_reps),
    ]) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
