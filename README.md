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

## Dependence and stable-tail sensitivity

The `dependence` suite varies both AR coefficients over `{0, 0.5, 0.8}` for each of DGPs 1, 2, and 3 (nine settings per DGP, 27 total). DGP 3 uses skewed stable errors (`beta=0.7`); DGPs 1 and 2 use symmetric errors. All have `alpha=1.5`, `n=800`, `d=256`, and `tau=8`. Within each DGP, the marginal distributions of covariates and errors are held fixed as dependence changes. The renderer writes one dependence table with a `Data` column grouping nine rows per DGP; `adjustbox` fits its width to the text area.

The `tail` suite crosses stable indices `{1.35, 1.5, 1.8}` with symmetric (`beta=0`) and skewed (`beta=0.7`) errors (six settings). It uses DGP 1's sparse ReLU signal and DGP 3's skewed-noise convention, at `n=800` and `d=256`. There are 33 distinct settings across both suites. Stable scales are matched by marginal IQR across tail settings. The Huber threshold is held at 14 throughout the tail suite, satisfying the configured curvature check even at `alpha=1.35`.

Check the matrices, then run both suites into one output directory:

```bash
python -m experiments.run --suite dependence --dry-run --reps 100
python -m experiments.run --suite tail --dry-run --reps 100
python -m experiments.run --suite dependence --reps 100 --workers 4 --output-root results_sensitivity
python -m experiments.run --suite tail --reps 100 --workers 4 --output-root results_sensitivity
python -m experiments.sensitivity_tables --input results_sensitivity/aggregate_summary.csv --output results_sensitivity/sensitivity_tables.tex
```

The two tables report prediction MSE with Monte Carlo SE, plus squared population Huber target bias for the stable-tail sweep. The errors are measured against `f0`, including for skewed noise. `--min-reps` can render a completed smaller pilot; label such tables as preliminary. The generator checks that every case has the requested number of replications and all five methods. LaTeX requires `booktabs`, `multirow`, and `adjustbox`.

The completed 100-replication results are committed under `reference/sensitivity_aggregate_summary.csv`, `reference/sensitivity_paired_differences.csv`, and `reference/sensitivity_tables.tex`. The paired-difference file reports the mean and Monte Carlo SE of within-replication MSE reductions, with positive values favoring the second method. Raw per-replication fits remain in the ignored local `results_sensitivity` directory. Regenerate the reference table with `python -m experiments.sensitivity_tables --input reference/sensitivity_aggregate_summary.csv --output reference/sensitivity_tables.tex`.

## Real-data forecasting: four-task benchmark

This benchmark extends the **HuberNNTS** simulation study using four real-data forecasting targets. The four tasks were **selected post hoc** after screening six tasks for the lowest average MAE **and** RMSE under Huber-SPDNN at 30 random seeds. Results on this selected subset are descriptive and **not** an unbiased estimate of performance across all datasets. The original simulation study remains unchanged.

### Retained tasks

| Task ID | Dataset | Prediction target | Features | Source |
|---|---|---|---|---|
| `beijing_aotizhongxin` | Beijing Multi-Site Air Quality | Aotizhongxin PM2.5 at t+1 | 0..3 lags of hourly weather/pollutant measurements through t | [UCI #501](https://archive.ics.uci.edu/dataset/501) |
| `beijing_changping` | Beijing Multi-Site Air Quality | Changping PM2.5 at t+1 | same | [UCI #501](https://archive.ics.uci.edu/dataset/501) |
| `fred_tb3` | FRED-MD monthly | TB3SMFFM at t | transformed FRED-MD variables at t-1 | [St. Louis Fed](https://www.stlouisfed.org/research/economists/mccracken/fred-databases) |
| `fred_tb6` | FRED-MD monthly | TB6SMFFM at t | transformed FRED-MD variables at t-1 | [St. Louis Fed](https://www.stlouisfed.org/research/economists/mccracken/fred-databases) |

FRED-MD uses the frozen 2026-09 vintage and the historical window January 1980 through July 2022. These tasks are **one-step-ahead forecasts**, not direct replications of FAST-NN's contemporaneous regression experiments. The other two tasks explored during initial screening are intentionally excluded from the official four-task benchmark. No downloaded raw datasets are committed.

### Run

From the repository root (Python 3.9+):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m experiments.real_data --download-only
python -m experiments.real_data --reps 30 --epochs 120 --patience 20 --output-root results/real_four_30
```

The `--download-only` command saves files in `data/real/`: `beijing.zip` and `fred_md_2026_09.csv`. If direct download is blocked, download from the linked providers manually and place files at these paths.

For a quick correctness smoke test (not a performance claim):

```bash
python -m experiments.real_data --datasets fred_tb3 --reps 1 --epochs 3 --patience 1 --max-train 100 --output-root results/real_four_smoke
python -m unittest discover -s tests -v
```

`--datasets` accepts **only** the four task IDs above; omitting it runs all four. Defaults: `--reps 30`, `--lags 3`, `--seed 42`, `--tau 1.5`, `--width 24`, `--epochs 120`, `--patience 20`, `--lr 0.003`.

**Avoid reusing a six-task output directory:** the new benchmark writes to `results/real_four_30` by default. The manifest rejects an incompatible task list or training parameters. Each method fit is checkpointed and an interrupted run can resume with matching settings.

### Experimental protocol and outputs

- **Four models:** LS-DNN, Huber-DNN, LS-SPDNN, Huber-SPDNN, with two hidden ReLU layers, clipped outputs and projected weights; sparse models validate a seven-value penalty grid.
- **Split:** chronological 60% train, 20% validation, 20% test. Feature imputation and scaling, and target standardization, are estimated from the training partition only. The test portion is never used for hyperparameter selection.
- **Selection:** early stopping and sparsity penalty minimize the mean of validation standardized MAE and RMSE; report **original-unit** MAE and RMSE on test.
- **Repetitions:** 30 random model initializations on a fixed chronological split. Gu (2025), FAST-NN's FRED-MD application, also used 30 repetitions, but **random training/validation splits**, so the uncertainty protocols are not identical.
- **Files:** `per_replication.csv` (task, repetition, method), `summary.csv` (mean, std, best MAE/RMSE flags), `joint_winners.csv`, and `manifest.json` in the output directory.

The supplied results are stored at `reference/real_data_30reps_summary.csv` and `reference/real_data_30reps_per_replication.csv` (four tasks only, 30 repeats each). These are records of previous locally executed experiments, **not** numerical outputs generated by a GitHub CI run; exact reproduction depends on upstream data files and PyTorch environment.

Because tasks were retained on the basis of observed test performance, do not use their 4/4 win rate to claim population-level superiority or a confirmatory test. Consider a separately preregistered dataset set or new future time splits for confirmatory evaluation.
