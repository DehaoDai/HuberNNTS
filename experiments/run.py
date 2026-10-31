"""Run the Numerical Studies DGP matrix with parallel replications.

Examples:
  python -m experiments.run --suite sanity
  python -m experiments.run --suite pilot --reps 20 --workers 4
  python -m experiments.run --suite paper --reps 100 --workers 4

Every worker uses one PyTorch/BLAS thread. Completed replications are
appended to CSV as they finish, and identical runs resume automatically.
"""

import os

for _variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
from dataclasses import asdict, dataclass
import json
import math
import multiprocessing as mp
from pathlib import Path
import shutil

import numpy as np
import torch

from .network import ClippedNetwork
from .dgp import (
    DEFAULT_P, DGP, SPARSITIES, absolute_moment, active_coordinates,
    calibrate_sparse_amplitudes, curvature_min_tau, draw_x_marginal,
    exact_sparse_parameter_count, generate_training_series,
    make_exact_sparse_network, population_huber_shift,
    regression_function, stable_scale,
)


METHODS = (
    "LS-DNN", "Huber-DNN", "LS-SPDNN", "Huber-SPDNN",
    "Oracle-support-Huber",
)
LAMBDA_GRID = (0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
SUITES = (
    "sanity", "pilot", "sample_size", "dimension", "sparsity",
    "dependence", "approximation", "bias", "bias_sample_size",
    "dimension_bc", "sparsity_c", "tail", "paper", "all",
)


@dataclass(frozen=True)
class Case:
    family: str
    n: int
    d: int
    sparsity: int = 4
    rho_x: float = 0.5
    rho_e: float = 0.5
    width: int = 24
    tau: float = 8.0
    beta: float = 0.0
    alpha: float = 1.5

    @property
    def signal_kind(self):
        return "factor_sparse" if self.family == "B" else "sparse_relu"

    @property
    def case_id(self):
        def tag(value):
            return f"{value:g}".replace(".", "p")

        return (
            f"{self.family}_n{self.n}_d{self.d}_s{self.sparsity}"
            f"_rx{tag(self.rho_x)}_re{tag(self.rho_e)}"
            f"_w{self.width}_tau{tag(self.tau)}"
            f"_a{tag(self.alpha)}_b{tag(self.beta)}"
        )


@dataclass(frozen=True)
class RunConfig:
    reps: int
    n_test: int
    epochs: int
    patience: int
    restarts: int
    lr: float
    omega: float
    output_bound: float
    parameter_bound: float
    lip_bound: float
    seed: int
    workers: int


def cases_for(suite):
    if suite == "pilot":
        return [Case("A", 800, 64), Case("A", 800, 1024)]
    if suite == "sample_size":
        return [Case("A", n, 256) for n in (400, 800, 1600, 3200)]
    if suite == "dimension":
        return [Case("A", 800, d) for d in (64, 256, 1024, 2048)]
    if suite == "dimension_bc":
        return [
            Case(family, 800, d, tau=8.0, beta=0.7 if family == "C" else 0.0)
            for family in ("B", "C") for d in (64, 256, 1024, 2048)
        ]
    if suite == "sparsity":
        return [
            Case("A", n, 1024, sparsity=s)
            for n in (800, 1600) for s in SPARSITIES
        ]
    if suite == "sparsity_c":
        return [
            Case("C", n, 1024, sparsity=s, tau=8.0, beta=0.7)
            for n in (800, 1600) for s in SPARSITIES
        ]
    if suite == "dependence":
        return [
            Case(family, 800, 256, rho_x=rx, rho_e=re,
                 beta=0.7 if family == "C" else 0.0)
            for family in ("A", "B", "C")
            for rx in (0.0, 0.5, 0.8)
            for re in (0.0, 0.5, 0.8)
        ]
    if suite == "approximation":
        cases = [Case("B", n, 256) for n in (400, 800, 1600, 3200)]
        cases.extend(Case("B", 1600, 256, width=w) for w in (8, 16, 32))
        return cases
    if suite == "bias":
        return [
            Case("C", 800, 256, tau=tau, beta=0.7)
            for tau in (8.0, 12.0, 16.0, 32.0)
        ]
    if suite == "bias_sample_size":
        return [
            Case("C", n, 256, tau=8.0, beta=0.7)
            for n in (400, 800, 1600, 3200)
        ]
    if suite == "tail":
        return [
            Case(family, 800, 256, alpha=alpha,
                 beta=0.7 if family == "C" else 0.0, tau=14.0)
            for family in ("A", "B", "C")
            for alpha in (1.35, 1.5, 1.8)
        ]
    if suite == "paper":
        # The four tables in the Numerical Studies manuscript.
        seen = {}
        for name in (
            "sample_size", "dimension", "dimension_bc", "sparsity",
            "sparsity_c", "approximation", "bias_sample_size",
        ):
            for case in cases_for(name):
                seen[case.case_id] = case
        return list(seen.values())
    if suite == "all":
        seen = {}
        for name in (
            "sample_size", "dimension", "sparsity", "dependence",
            "approximation", "bias", "bias_sample_size", "dimension_bc",
            "sparsity_c", "tail",
        ):
            for case in cases_for(name):
                seen[case.case_id] = case
        return list(seen.values())
    raise ValueError(f"Unknown suite: {suite}")


def worker_initializer():
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def loss_value(prediction, target, method, tau):
    if method.startswith("Huber") or method == "Oracle-support-Huber":
        return torch.nn.functional.huber_loss(
            prediction, target, delta=tau, reduction="mean"
        )
    return 0.5 * torch.mean((prediction - target)**2)


def train_one(case, config, method, lam, restart, rep, x_train, y_train,
              x_valid, y_valid, x_test, f_test, target_test):
    oracle_input = method == "Oracle-support-Huber"
    if oracle_input:
        columns = active_coordinates(case.signal_kind, case.sparsity)
        x_train = x_train[:, columns].copy()
        x_valid = x_valid[:, columns].copy()
        x_test = x_test[:, columns].copy()
    input_dimension = x_train.shape[1]
    torch.manual_seed(config.seed + rep + 10_000 + 100_000 * restart)
    model = ClippedNetwork(
        input_dimension, case.width, 2, config.output_bound,
        config.parameter_bound, config.lip_bound,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    xt = torch.from_numpy(x_train)
    yt = torch.from_numpy(y_train)
    xv = torch.from_numpy(x_valid)
    yv = torch.from_numpy(y_valid)
    xq = torch.from_numpy(x_test)
    best_valid = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()
        objective = loss_value(model(xt).flatten(), yt, method, case.tau)
        if lam:
            objective = objective + lam * model.clipped_penalty(config.omega)
        if not torch.isfinite(objective):
            raise FloatingPointError(
                f"Nonfinite objective in {case.case_id}, rep={rep}, {method}"
            )
        objective.backward()
        optimizer.step()
        model.project_parameters()
        with torch.no_grad():
            validation = float(loss_value(
                model(xv).flatten(), yv, method, case.tau
            ))
        if not math.isfinite(validation):
            raise FloatingPointError(
                f"Nonfinite validation in {case.case_id}, rep={rep}, {method}"
            )
        if validation < best_valid - 1e-8:
            best_valid = validation
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if epoch - best_epoch >= config.patience:
            break
    model.load_state_dict(best_state)
    with torch.no_grad():
        pred_test = model(xq).flatten().numpy()
        train_objective = float(loss_value(
            model(xt).flatten(), yt, method, case.tau
        ))
        penalty_norm = float(model.clipped_penalty(config.omega))
        train_objective += lam * penalty_norm
        active_count = model.active_count(config.omega)
    return {
        "method": method, "lambda": lam, "restart": restart,
        "validation_loss": best_valid, "train_objective": train_objective,
        "best_epoch": best_epoch, "clipped_parameter_norm": penalty_norm,
        "active_parameters": active_count,
        "prediction_error": float(np.mean((pred_test - f_test)**2)),
        "target_prediction_error": float(
            np.mean((pred_test - target_test)**2)
        ),
    }


def true_parameter_objectives(case, config, amplitude, x_train, y_train):
    if case.family == "B":
        return None
    model = ClippedNetwork(
        case.d, case.width, 2, config.output_bound,
        config.parameter_bound, config.lip_bound,
    )
    make_exact_sparse_network(model, case.sparsity, amplitude)
    with torch.no_grad():
        prediction = model(torch.from_numpy(x_train)).flatten()
        target = torch.from_numpy(y_train)
        norms = float(model.clipped_penalty(config.omega))
        return {
            "Huber": float(loss_value(
                prediction, target, "Huber-DNN", case.tau
            )),
            "LS": float(loss_value(
                prediction, target, "LS-DNN", case.tau
            )),
            "norm": norms,
        }


def run_replication(case, config, rep, amplitude, target_shift):
    worker_initializer()
    rng = np.random.default_rng(config.seed + rep)
    spec = DGP(
        kind=case.signal_kind, n=case.n, d=case.d,
        rho_x=case.rho_x, rho_e=case.rho_e,
        sparsity=case.sparsity, alpha=case.alpha,
        beta=case.beta, amplitude=amplitude,
    )
    x_train, y_train = generate_training_series(spec, rng)
    x_valid, y_valid = generate_training_series(spec, rng)
    x_test = draw_x_marginal(config.n_test, case.d, rng)
    f_test = regression_function(
        x_test, case.signal_kind, case.sparsity, amplitude
    )
    target_test = np.clip(
        f_test.astype(np.float64) + target_shift,
        -config.output_bound, config.output_bound,
    ).astype(np.float32)
    references = true_parameter_objectives(
        case, config, amplitude, x_train, y_train
    )
    rows = []
    for method in METHODS:
        candidates = (
            LAMBDA_GRID if method.endswith("SPDNN") else (0.0,)
        )
        fits = [
            train_one(
                case, config, method, lam, restart, rep,
                x_train, y_train, x_valid, y_valid, x_test,
                f_test, target_test,
            )
            for lam in candidates
            for restart in range(config.restarts)
        ]
        selected = min(fits, key=lambda fit: fit["validation_loss"])
        selected.update({
            "case_id": case.case_id, "rep": rep,
            "n": case.n, "d": case.d, "sparsity": case.sparsity,
            "family": case.family, "rho_x": case.rho_x,
            "rho_e": case.rho_e, "width": case.width,
            "tau": case.tau, "alpha": case.alpha, "beta": case.beta,
            "signal_amplitude": amplitude,
            "zero_predictor_error": float(np.mean(f_test**2)),
            "best_constant_error": float(np.var(f_test)),
            "population_target_bias_sq": float(
                np.mean((target_test - f_test)**2)
            ),
            "true_parameter_count": (
                exact_sparse_parameter_count(case.sparsity)
                if references is not None else ""
            ),
            "train_objective_minus_true_parameter": "",
        })
        if references is not None and method != "Oracle-support-Huber":
            family = "Huber" if method.startswith("Huber") else "LS"
            comparator = (
                references[family] + selected["lambda"] * references["norm"]
            )
            selected["train_objective_minus_true_parameter"] = (
                selected["train_objective"] - comparator
            )
        rows.append(selected)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def mean_mcse(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values.mean()), float("nan")
    return (
        float(values.mean()),
        float(values.std(ddof=1) / math.sqrt(len(values))),
    )


def summarize_case(case, root, expected_reps):
    rows = read_csv(root / "replications.csv")
    complete = defaultdict(dict)
    for row in rows:
        complete[int(row["rep"])][row["method"]] = row
    if len(complete) != expected_reps or any(
        set(per_method) != set(METHODS) for per_method in complete.values()
    ):
        raise ValueError(f"Incomplete replications for {case.case_id}")
    summary = []
    for method in METHODS:
        selected = [complete[rep][method] for rep in sorted(complete)]
        pe, pe_mcse = mean_mcse(
            [float(row["prediction_error"]) for row in selected]
        )
        target_pe, target_mcse = mean_mcse(
            [float(row["target_prediction_error"]) for row in selected]
        )
        objective_gaps = [
            float(row["train_objective_minus_true_parameter"])
            for row in selected
            if row["train_objective_minus_true_parameter"] != ""
        ]
        summary.append({
            **asdict(case), "case_id": case.case_id, "method": method,
            "reps": expected_reps,
            "mean_prediction_error": pe,
            "sem_prediction_error": pe_mcse,
            "mean_target_prediction_error": target_pe,
            "sem_target_prediction_error": target_mcse,
            "mean_clipped_parameter_norm": float(np.mean([
                float(row["clipped_parameter_norm"]) for row in selected
            ])),
            "mean_active_parameters": float(np.mean([
                float(row["active_parameters"]) for row in selected
            ])),
            "mean_zero_predictor_error": float(np.mean([
                float(row["zero_predictor_error"]) for row in selected
            ])),
            "mean_best_constant_error": float(np.mean([
                float(row["best_constant_error"]) for row in selected
            ])),
            "mean_population_target_bias_sq": float(np.mean([
                float(row["population_target_bias_sq"]) for row in selected
            ])),
            "mean_selected_lambda": float(np.mean([
                float(row["lambda"]) for row in selected
            ])),
            "lambda_zero_fraction": float(np.mean([
                float(row["lambda"]) == 0.0 for row in selected
            ])),
            "mean_train_objective_minus_true_parameter": (
                float(np.mean(objective_gaps)) if objective_gaps else ""
            ),
            "fraction_objective_above_true_parameter": (
                float(np.mean(np.asarray(objective_gaps) > 0))
                if objective_gaps else ""
            ),
        })
    write_csv(root / "summary.csv", summary)
    comparisons = (
        ("LS-DNN", "Huber-DNN"),
        ("Huber-DNN", "Huber-SPDNN"),
        ("LS-DNN", "LS-SPDNN"),
        ("Huber-DNN", "Oracle-support-Huber"),
    )
    paired = []
    for first, second in comparisons:
        diff = [
            float(complete[rep][first]["prediction_error"])
            - float(complete[rep][second]["prediction_error"])
            for rep in sorted(complete)
        ]
        gain, gain_mcse = mean_mcse(diff)
        paired.append({
            "case_id": case.case_id, "first_method": first,
            "second_method": second,
            "mean_error_reduction": gain,
            "sem_error_reduction": gain_mcse,
            "reps": expected_reps,
        })
    write_csv(root / "paired_differences.csv", paired)
    return summary


def case_manifest(case, config, amplitude, target_shift):
    vp = absolute_moment(
        case.alpha, case.beta, 0.8, DEFAULT_P
    )
    tau_min = curvature_min_tau(vp, DEFAULT_P, config.output_bound)
    if case.tau + 1e-10 < tau_min:
        raise ValueError(
            f"{case.case_id}: tau={case.tau} below curvature minimum {tau_min}"
        )
    if case.family != "B" and case.width < case.sparsity // 2:
        raise ValueError("Width cannot represent the sparse oracle.")
    parameter_count = (
        case.d * case.width + case.width**2 + 3 * case.width + 1
    )
    w = max(case.d, case.width, 1)
    sensitivity_constant = (
        3.0 * ((w + 1.0) * config.parameter_bound)**3
    )
    eta_proxy = 1.0 / case.n
    entropy_upper_proxy = parameter_count * math.log1p(
        2.0 * config.parameter_bound * sensitivity_constant / eta_proxy
    )
    r_upper_proxy = (
        (entropy_upper_proxy + 1.0) * math.log(case.n)**2 / case.n
    )
    return {
        "case": asdict(case),
        "config": {
            key: value for key, value in asdict(config).items()
            if key not in ("reps", "workers")
        },
        "amplitude": amplitude,
        "stable_scale": stable_scale(case.alpha, case.beta, 0.8),
        "p": DEFAULT_P,
        "v_p": vp,
        "curvature_min_tau": tau_min,
        "curvature_left": 2.0**DEFAULT_P * vp / case.tau**DEFAULT_P,
        "target_shift": target_shift,
        "lambda_grid": list(LAMBDA_GRID),
        "network_parameter_count": parameter_count,
        "entropy_upper_proxy_eta": eta_proxy,
        "entropy_upper_proxy": entropy_upper_proxy,
        "r_upper_proxy": r_upper_proxy,
        "r_upper_proxy_below_one": r_upper_proxy < 1.0,
        "omega_times_sensitivity_constant": (
            config.omega * sensitivity_constant
        ),
        "eta_proxy_exceeds_omega_C": (
            eta_proxy > config.omega * sensitivity_constant
        ),
        "entropy_proxy_note": (
            "Crude upper bound only; r_upper_proxy >= 1 does not "
            "establish that the unknown exact r_n >= 1."
        ),
    }


def run_case(case, config, output_root, amplitudes, overwrite=False):
    amplitude = amplitudes[case.sparsity] if case.family != "B" else 1.0
    target_shift = population_huber_shift(
        case.alpha, case.beta, 0.8, case.tau
    )
    manifest = case_manifest(case, config, amplitude, target_shift)
    root = output_root / case.case_id
    if overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise ValueError(
                f"{case.case_id} has different settings. Use a new "
                "output root or --overwrite."
            )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    csv_path = root / "replications.csv"
    existing_rows = read_csv(csv_path)
    groups = defaultdict(list)
    for row in existing_rows:
        groups[int(row["rep"])].append(row)
    complete_reps = {
        rep for rep, rows in groups.items()
        if len(rows) == len(METHODS)
        and {row["method"] for row in rows} == set(METHODS)
    }
    complete_reps = {rep for rep in complete_reps if 1 <= rep <= config.reps}
    clean_rows = [
        row for row in existing_rows if int(row["rep"]) in complete_reps
    ]
    if clean_rows and len(clean_rows) != len(existing_rows):
        write_csv(csv_path, clean_rows)
    pending = [
        rep for rep in range(1, config.reps + 1)
        if rep not in complete_reps
    ]
    print(
        f"{case.case_id}: {len(complete_reps)}/{config.reps} complete, "
        f"{len(pending)} pending, workers={config.workers}",
        flush=True,
    )
    if pending:
        pool = ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=mp.get_context("spawn"),
            initializer=worker_initializer,
        )
        try:
            futures = {
                pool.submit(
                    run_replication, case, config, rep,
                    amplitude, target_shift,
                ): rep for rep in pending
            }
            for future in as_completed(futures):
                rows = future.result()
                if csv_path.exists():
                    with csv_path.open("a", newline="") as file:
                        writer = csv.DictWriter(
                            file, fieldnames=list(rows[0])
                        )
                        writer.writerows(rows)
                else:
                    write_csv(csv_path, rows)
                complete_reps.add(futures[future])
                print(
                    f"{case.case_id}: {len(complete_reps)}/{config.reps}",
                    flush=True,
                )
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    return summarize_case(case, root, config.reps)


def run_sanity(amplitudes):
    from scipy.stats import spearmanr

    print("Calibrated amplitudes:", amplitudes)
    for sparsity in SPARSITIES:
        case = Case("A", 800, 64, sparsity=sparsity)
        x = draw_x_marginal(1024, case.d, np.random.default_rng(sparsity))
        f0 = regression_function(
            x, "sparse_relu", sparsity, amplitudes[sparsity]
        )
        model = ClippedNetwork(64, 24, 2, 2.0, 2.0, 100.0)
        make_exact_sparse_network(model, sparsity, amplitudes[sparsity])
        with torch.no_grad():
            prediction = model(torch.from_numpy(x)).flatten().numpy()
        difference = float(np.max(np.abs(prediction - f0)))
        count = sum(int((parameter.abs() > 1e-10).sum())
                    for parameter in model.parameters())
        expected = exact_sparse_parameter_count(sparsity)
        if difference > 1e-5 or count != expected:
            raise AssertionError(
                f"Oracle mismatch at s={sparsity}: {difference}, {count}"
            )
        print(
            f"s={sparsity}: exact-network max error={difference:.3g}, "
            f"nonzeros={count}, bound={1.5*amplitudes[sparsity]:.3g}"
        )
    for rho in (0.0, 0.5, 0.8):
        rng = np.random.default_rng(20261028)
        spec = DGP("sparse_relu", 10_000, 64, rho, rho)
        x, y = generate_training_series(spec, rng)
        f0 = regression_function(x, "sparse_relu", 4)
        e = y - f0
        if x.min() < 0 or x.max() > 1:
            raise AssertionError("Covariates left [0,1]^d.")
        ranks = [
            float(spearmanr(z[:-1], z[1:]).statistic)
            for z in (x[:, 0], e, y)
        ]
        x_quartiles = np.quantile(x[:, 0], (0.25, 0.75))
        e_quartiles = np.quantile(e, (0.25, 0.75))
        print(
            f"rho={rho:g}: lag-one Spearman(X1,e,Y)="
            f"{tuple(round(v,3) for v in ranks)}, "
            f"X1 quartiles={tuple(round(float(v),3) for v in x_quartiles)}, "
            f"e quartiles={tuple(round(float(v),3) for v in e_quartiles)}"
        )
    vp = absolute_moment()
    print(
        f"p={DEFAULT_P}, v_p={vp:.6f}, tau_min="
        f"{curvature_min_tau(vp):.3f}"
    )


def aggregate_results(output_root):
    """Collect all completed case summaries; table output is built separately."""
    all_rows = []
    for path in sorted(output_root.glob("*/summary.csv")):
        all_rows.extend(read_csv(path))
    if all_rows:
        write_csv(output_root / "aggregate_summary.csv", all_rows)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--workers", type=int,
                        default=min(4, os.cpu_count() or 1))
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--n-test", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--omega", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--output-root", type=Path,
                        default=Path("results"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    reps = options.reps if options.reps is not None else (
        20 if options.suite == "pilot" else 100
    )
    if min(
        reps, options.workers, options.restarts, options.n_test,
        options.epochs, options.patience
    ) < 1 or options.lr <= 0 or options.omega <= 0:
        parser.error("All run sizes, workers, restarts, and tuning values must be positive.")
    config = RunConfig(
        reps=reps, n_test=options.n_test, epochs=options.epochs,
        patience=options.patience, restarts=options.restarts,
        lr=options.lr, omega=options.omega,
        output_bound=2.0, parameter_bound=2.0, lip_bound=100.0,
        seed=options.seed, workers=options.workers,
    )
    amplitudes, variances = calibrate_sparse_amplitudes()
    if options.suite == "sanity":
        worker_initializer()
        run_sanity(amplitudes)
        return
    cases = cases_for(options.suite)
    print(
        f"{len(cases)} cases, {config.reps} reps/case, "
        f"{config.workers} parallel workers, {config.restarts} restarts",
        flush=True,
    )
    for case in cases:
        amplitude = amplitudes[case.sparsity] if case.family != "B" else 1.0
        case_manifest(
            case, config, amplitude,
            population_huber_shift(case.alpha, case.beta, 0.8, case.tau),
        )
        print(case.case_id, flush=True)
    if options.dry_run:
        return
    options.output_root.mkdir(parents=True, exist_ok=True)
    (options.output_root / "signal_calibration.json").write_text(json.dumps(
        {"amplitudes": amplitudes, "unit_variances": variances,
         "reference_seed": 20261028, "reference_n": 2**18},
        indent=2, sort_keys=True,
    ))
    for case in cases:
        run_case(
            case, config, options.output_root, amplitudes,
            overwrite=options.overwrite,
        )
        aggregate_results(options.output_root)
    print(f"Results saved in {options.output_root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
