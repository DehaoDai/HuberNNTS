# HuberNNTS numerical studies

Reproducible code for the four simulation tables in the supplied **Numerical Studies** section. DGP 1 and 3 have a sparse ReLU regression function; DGP 2 has five fixed active coordinates and a sine component. All three have serially dependent covariates and stable AR(1) errors with infinite variance.

## Setup

Use Python 3.9 or newer, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The original numerical run used Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1, and PyTorch 2.8.0. PyTorch results may vary slightly by platform and version.

## Reproduce the tables

First check the 38 distinct settings without fitting models:

```bash
python -m experiments.run --suite paper --dry-run --reps 100 --workers 4
```

Run the full experiment with four **parallel processes**:

```bash
python -m experiments.run --suite paper --reps 100 --workers 4 --output-root results
python -m experiments.tables --input results/aggregate_summary.csv --output results/paper_tables.tex
```

The full run fits all five methods over 38 settings and 100 independent replications; the sparse methods each try seven penalty values, so this is a substantial CPU job. Each process uses one PyTorch/BLAS thread. Completed replications are saved as they finish; rerunning the same command resumes them. Use a new `--output-root` for different tuning settings, or use `--overwrite` to replace an existing run.

A short parallel smoke run is:

```bash
python -m experiments.run --suite pilot --reps 2 --workers 2 --epochs 3 --patience 2 --n-test 128 --output-root results_smoke
```

The table renderer requires at least 100 replications per setting by default. For a smaller completed run, set `--min-reps` explicitly. It validates that every required case and method is present.

## What is fitted

- Covariates: a stationary Gaussian AR(1) factor with loading 1 on coordinate 1, 0.6 on coordinates 2–33, and 0.2 on the rest, transformed by the standard normal CDF into `[0,1]^d`.
- Errors: an independent stationary stable AR(1), with `alpha=1.5`, `rho_epsilon=0.5`, and skewness `beta=0` for DGP 1/2 or `beta=0.7` for DGP 3. Stable scale is calibrated to the IQR of the symmetric alpha 1.5 law with scale 0.8.
- DGP 1: `A_s * (3/s) * sum_k ReLU(x[2k] + x[2k+1] - 1)` using one-based input indices and `s=4,8,16,32`. DGP 3 uses the same signal with skewed errors.
- DGP 2: `0.7 sin(2πx1) + 0.8 ReLU(x2+x3-1) - 0.5 ReLU(x4+x5-1)`. Its active input count remains five; there is no DGP 2 sparsity sweep.
- Fits: LS-DNN, Huber-DNN, LS-SPDNN, Huber-SPDNN, and Oracle-Huber (true active inputs only). The oracle is a diagnostic fitted network, not the true regression function.
- Network: two hidden ReLU layers, width 24 unless swept, clipped output `[-2,2]`, parameter projection `[-2,2]`, and the source Lipschitz projection. Sparse fits use `lambda * sum min(|theta|/0.05,1)` with `lambda` in `{0,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2}`.
- Optimization: full-batch Adam at 0.003, up to 150 epochs, patience 25, one initialization. A separate stationary validation trajectory selects the epoch and penalty.
- Metric: prediction error against the known `f0` on 10,000 independent draws from the stationary covariate marginal. Parentheses show the Monte Carlo standard error of the mean over replications.

`results/<case_id>/replications.csv` stores selected fit details for every method and replication; `summary.csv` stores case-level statistics, and `results/aggregate_summary.csv` combines completed cases. The reference aggregate from the original runs is at `reference/aggregate_summary.csv`. Generate its exact four tables with:

```bash
python -m experiments.tables --input reference/aggregate_summary.csv --output reference/paper_tables.tex
```

To include the generated TeX in a paper, load `booktabs`, `multirow`, and `adjustbox` and use `\input{.../paper_tables.tex}`. The generated table file is a reference rendering of prior results; it does not replace running the experiments.

## Checks

```bash
python -m unittest discover -s tests -v
python -m experiments.run --suite sanity
```

The sanity command checks the exact sparse comparator, compact support of `X`, serial dependence, and the moment condition used for Huber curvature.
