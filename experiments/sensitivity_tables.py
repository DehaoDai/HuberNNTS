"""Render dependence and stable-tail sensitivity tables from completed runs.

Usage: python -m experiments.sensitivity_tables --input results_sensitivity/aggregate_summary.csv
"""

import argparse
from pathlib import Path

from .run import METHODS, cases_for
from .tables import DISPLAY, load_summaries, method_cells, table


def render_dependence(index, min_reps):
    body = []
    cases = cases_for("dependence")
    for family in ("A", "B", "C"):
        if body:
            body.append(r"\midrule")
        family_cases = [case for case in cases if case.family == family]
        for i, case in enumerate(family_cases):
            data = (rf"\multirow{{9}}{{*}}{{DGP {ord(family) - 64}}}"
                    if i == 0 else "")
            body.append(" & ".join([
                data, f"{case.rho_x:g}", f"{case.rho_e:g}",
                *method_cells(index, case, METHODS, min_reps),
            ]) + r"\\")
    return table(
        r"Dependence-strength sweep for DGPs 1--3 at $n=800$, $d=256$, "
        r"$\alpha=1.5$, and $\tau=8$. DGPs 1 and 2 have symmetric "
        r"stable errors ($\beta=0$); DGP 3 has skewed errors ($\beta=0.7$). "
        r"Entries are prediction MSE (Monte Carlo SE); boldface marks the "
        r"best full-input estimator. Oracle-Huber uses true active inputs.",
        "tab:dependence-strength", "lrrccccc",
        r"Data & $\rho_X$ & $\rho_\varepsilon$ & "
        + " & ".join(DISPLAY[method] for method in METHODS), body,
    )


def render_tail(index, min_reps):
    body = []
    for case in cases_for("tail"):
        row = index.get((case.case_id, "Huber-DNN"))
        if row is None:
            raise ValueError(f"Missing Huber-DNN for {case.case_id}")
        bias_sq = float(row["mean_population_target_bias_sq"])
        body.append(" & ".join([
            f"{case.alpha:g}", f"{case.beta:g}", f"{case.tau:g}",
            f"{bias_sq:.4f}",
            *method_cells(index, case, METHODS, min_reps),
        ]) + r"\\")
    return table(
        r"Stable-tail and skewness sweep at $n=800$, $d=256$, "
        r"$(\rho_X,\rho_\varepsilon)=(0.5,0.5)$. "
        r"$\beta=0$ is symmetric and $\beta=0.7$ is skewed. "
        r"Bias$^2$ is the squared gap between the clipped population Huber "
        r"target and $f_0$, averaged over the test covariates. "
        r"Other entries are prediction MSE (Monte Carlo SE); boldface marks "
        r"the best full-input estimator. The common $\tau=14$ "
        r"satisfies the stated curvature check.",
        "tab:stable-tail-skew", "rrrrccccc",
        r"$\alpha$ & $\beta$ & $\tau$ & Bias$^2$ & "
        + " & ".join(DISPLAY[method] for method in METHODS), body,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("results_sensitivity/aggregate_summary.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results_sensitivity/sensitivity_tables.tex"))
    parser.add_argument("--min-reps", type=int, default=100)
    args = parser.parse_args()
    if args.min_reps < 1:
        parser.error("--min-reps must be positive")
    index = load_summaries(args.input)
    rendered = "\n\n".join((
        render_dependence(index, args.min_reps),
        render_tail(index, args.min_reps),
    )) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
