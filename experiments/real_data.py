"""Real-data HuberNNTS benchmark: one-step-ahead forecasts, four full-input models.

Four pre-specified reporting tasks: two Beijing stations and two FRED-MD spreads. See REAL_DATA.md.
Data are downloaded from original public providers on the local machine only.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch

from .network import ClippedNetwork

SOURCES = {
    "fred": (
        "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/2026-09-md.csv",
        "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/2026-09-md.csv",
    ),
    "beijing": (
        "https://archive.ics.uci.edu/static/public/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata.zip",
    ),
}
FILES = {"fred": "fred_md_2026_09.csv", "beijing": "beijing.zip"}
TASKS = ("beijing_aotizhongxin", "beijing_changping", "fred_tb3", "fred_tb6")
TARGETS = {"fred_tb3": "TB3SMFFM", "fred_tb6": "TB6SMFFM"}
METHODS = ("LS-DNN", "Huber-DNN", "LS-SPDNN", "Huber-SPDNN")
LAMBDA_GRID = (0., 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def download(source: str, root: Path, allow_download: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    dest = root / FILES[source]
    if dest.exists() and dest.stat().st_size > 100:
        return dest
    if not allow_download:
        raise FileNotFoundError(f"Missing {dest}. Run with --download-only, or manually place dataset there.")
    failures = []
    for url in SOURCES[source]:
        try:
            print(f"Downloading {source} from {url}", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research data download)"})
            with urllib.request.urlopen(req, timeout=100) as response, dest.with_suffix(dest.suffix + ".part").open("wb") as out:
                import shutil
                shutil.copyfileobj(response, out)
            tmp = dest.with_suffix(dest.suffix + ".part")
            if tmp.stat().st_size < 100:
                raise RuntimeError("Unexpectedly small downloaded file")
            if source != "fred" and not zipfile.is_zipfile(tmp):
                raise ValueError("Expected ZIP from UCI; server returned another format")
            tmp.replace(dest)
            return dest
        except Exception as exc:
            failures.append(str(exc))
            dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)
    raise RuntimeError(f"Unable to download {source}. URLs: {SOURCES[source]}. Errors: {failures}")


def fred_transform(values: pd.Series, code: int) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").astype(float)
    if code in (4, 5, 6):
        x = np.log(x.where(x > 0))
        return {4: x, 5: x.diff(), 6: x.diff().diff()}[code]
    if code == 2:
        return x.diff()
    if code == 3:
        return x.diff().diff()
    if code == 7:
        return x.pct_change(fill_method=None).diff()
    return x  # tcode 1 / unavailable


def load_fred(file: Path, task: str) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(file, low_memory=False)
    date_col = next((c for c in frame if c.lower() == "sasdate"), None)
    if date_col is None:
        raise ValueError("Expected FRED-MD CSV with 'sasdate' column")
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    valid = dates.notna()
    tcodes = frame.loc[~valid].head(1)
    raw = frame.loc[valid].drop(columns=date_col).reset_index(drop=True)
    timestamp = dates.loc[valid].reset_index(drop=True)
    transformed = pd.DataFrame(index=raw.index)
    for col in raw.columns:
        code = 1
        if not tcodes.empty:
            value = pd.to_numeric(tcodes.iloc[0].get(col), errors="coerce")
            if pd.notna(value) and int(value) in range(1, 8):
                code = int(value)
        transformed[col] = fred_transform(raw[col], code)
    mask = (timestamp >= pd.Timestamp("1980-01-01")) & (timestamp <= pd.Timestamp("2022-07-01"))
    transformed = transformed.loc[mask].reset_index(drop=True)
    target = TARGETS[task]
    if target not in transformed.columns:
        raise ValueError(f"FRED-MD column {target!r} missing; available={list(transformed.columns)[:20]}")
    # Genuine one-step ahead: predict transformed Y[t] using transformed variables at t-1.
    features = transformed.shift(1).add_prefix("lag1_")
    response = transformed[target]
    return features, response


def first_member_with_name(archive: zipfile.ZipFile, needle: str) -> bytes:
    needle = needle.lower()
    for name in archive.namelist():
        if name.lower().endswith(".csv") and needle in name.lower():
            return archive.read(name)
    for name in archive.namelist():
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(archive.read(name))) as child:
                try:
                    return first_member_with_name(child, needle)
                except FileNotFoundError:
                    pass
    raise FileNotFoundError(f"Could not find CSV containing {needle!r} in {archive.namelist()[:12]}")


def build_lagged(frame: pd.DataFrame, target: str, lags: int) -> tuple[pd.DataFrame, pd.Series]:
    if target not in frame:
        raise ValueError(f"Missing target {target}")
    base = frame.select_dtypes(include=["number"]).copy()
    base = base.drop(columns=[x for x in ("No", "year", "month", "day", "hour") if x in base], errors="ignore")
    # X[t] includes current plus previous `lags` observations; y[t] is next sample.
    features = pd.concat([base.shift(lag).add_prefix(f"lag{lag}_") for lag in range(lags + 1)], axis=1)
    response = pd.to_numeric(frame[target], errors="coerce").shift(-1)
    return features, response


def load_beijing(file: Path, station: str, lags: int) -> tuple[pd.DataFrame, pd.Series]:
    with zipfile.ZipFile(file) as z:
        blob = first_member_with_name(z, f"prsa_data_{station}")
    data = pd.read_csv(io.BytesIO(blob))
    data = data.sort_values(["year", "month", "day", "hour"]).reset_index(drop=True)
    # No interpolation from the future: remaining missing covariates filled using train medians.
    return build_lagged(data, "PM2.5", lags)


def prepare_task(task: str, root: Path, lags: int, allow_download: bool):
    if task not in TASKS:
        raise ValueError(f"Unknown real-data task {task!r}; available: {TASKS}")
    kind = "fred" if task.startswith("fred_") else "beijing"
    file = download(kind, root, allow_download)
    if kind == "fred":
        x, y = load_fred(file, task)
    else:
        x, y = load_beijing(file, task.removeprefix("beijing_"), lags)
    mask = y.notna() & np.isfinite(y.to_numpy(dtype=float))
    x = x.loc[mask].reset_index(drop=True).replace([np.inf, -np.inf], np.nan)
    y = y.loc[mask].reset_index(drop=True).to_numpy(dtype=float)
    if len(y) < 100:
        raise ValueError(f"Too few observations in {task}: {len(y)}")
    return x, y, {"task": task, "source": str(file), "n": len(y), "raw_features": x.shape[1]}


def _subsample_evenly(index: np.ndarray, max_rows: int | None):
    if max_rows is None or len(index) <= max_rows:
        return index
    return index[np.unique(np.linspace(0, len(index) - 1, max_rows).astype(int))]


def split_scale(xdf: pd.DataFrame, y: np.ndarray, max_train: int | None):
    # No shuffling or look-ahead: 60/20/20 chronological split.
    n = len(y)
    cut1, cut2 = int(0.6*n), int(0.8*n)
    train_idx = _subsample_evenly(np.arange(cut1), max_train)
    valid_idx, test_idx = np.arange(cut1, cut2), np.arange(cut2, n)
    # Remove all-empty columns based ONLY on training data.
    keep = xdf.iloc[train_idx].notna().any(axis=0)
    xdf = xdf.loc[:, keep].astype(float)
    med = xdf.iloc[train_idx].median().fillna(0.)
    xdf = xdf.fillna(med)
    # Train-only 1%/99% quantile scaling, clipped to theoretical [0,1] input range.
    a = xdf.iloc[train_idx].quantile(.01)
    b = xdf.iloc[train_idx].quantile(.99)
    denom = (b - a).where((b-a).abs() > 1e-8, 1.)
    x = ((xdf-a)/denom).clip(0., 1.).to_numpy(dtype=np.float32)
    mu = float(y[train_idx].mean())
    sigma = float(y[train_idx].std())
    if sigma < 1e-12:
        raise ValueError("Constant training target")
    yn = ((y-mu)/sigma).astype(np.float32)
    return x, yn, y, train_idx, valid_idx, test_idx, mu, sigma


def loss(pred, truth, method, tau):
    if method.startswith("Huber"):
        return torch.nn.functional.huber_loss(pred, truth, delta=tau)
    return torch.mean((pred-truth)**2) * 0.5


def metrics(pred: np.ndarray, truth: np.ndarray):
    err = np.asarray(pred, dtype=float)-np.asarray(truth, dtype=float)
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err*err)))


def train_candidate(x, yn, splits, method, lam, seed, options):
    ti, vi, qi = splits
    torch.manual_seed(seed)
    model = ClippedNetwork(x.shape[1], options.width, 2, options.output_bound, 2., 100.)
    opt = torch.optim.Adam(model.parameters(), lr=options.lr)
    xt = torch.from_numpy(x[ti]); yt = torch.from_numpy(yn[ti]); xv = torch.from_numpy(x[vi]); yv = torch.from_numpy(yn[vi])
    best, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(1, options.epochs+1):
        model.train(); opt.zero_grad()
        objective = loss(model(xt).flatten(), yt, method, options.tau)
        if lam:
            objective = objective + lam*model.clipped_penalty(.05)
        if not torch.isfinite(objective):
            raise FloatingPointError(f"Non-finite objective for {method}")
        objective.backward(); opt.step(); model.project_parameters()
        model.eval()
        with torch.no_grad():
            vp = model(xv).flatten().numpy()
        va, vr = metrics(vp, yn[vi]); score = (va+vr)/2
        if score < best-1e-7:
            best, best_state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch
        if epoch-best_epoch >= options.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        test_pred = model(torch.from_numpy(x[qi])).flatten().numpy()
        actives = model.active_count(.05)
    return best, test_pred, best_epoch, actives


def write_csv(filename: Path, rows: list[dict]):
    if not rows:
        return
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize(rows: list[dict]):
    summary = []
    for task in sorted(set(row["task"] for row in rows)):
        block = [r for r in rows if r["task"] == task]
        for method in METHODS:
            m = [r for r in block if r["method"] == method]
            summary.append({"task": task, "method": method, "reps": len(m),
                            "mean_mae": float(np.mean([float(r["mae"]) for r in m])),
                            "mean_rmse": float(np.mean([float(r["rmse"]) for r in m])),
                            "std_mae": float(np.std([float(r["mae"]) for r in m], ddof=1)) if len(m)>1 else float("nan"),
                            "std_rmse": float(np.std([float(r["rmse"]) for r in m], ddof=1)) if len(m)>1 else float("nan")})
        relevant = summary[-len(METHODS):]
        mae_min = min(r["mean_mae"] for r in relevant)
        rmse_min = min(r["mean_rmse"] for r in relevant)
        for item in relevant:
            item["best_mae"] = abs(item["mean_mae"]-mae_min) < 1e-10
            item["best_rmse"] = abs(item["mean_rmse"]-rmse_min) < 1e-10
            item["joint_best"] = item["best_mae"] and item["best_rmse"]
    return summary


def checkpoint_csv(filename: Path, rows: list[dict]):
    """Write a complete CSV snapshot and atomically replace the previous checkpoint."""
    if not rows:
        return
    filename.parent.mkdir(parents=True, exist_ok=True)
    tmp = filename.with_suffix(filename.suffix + ".tmp")
    write_csv(tmp, rows)
    tmp.replace(filename)


def save_partial_summary(rows: list[dict], output_root: Path):
    """Aggregate only complete replications, never partial four-model groups."""
    groups = {}
    for r in rows:
        groups.setdefault((r["task"], int(r["rep"])), []).append(r)
    complete = [r for members in groups.values()
                if {x["method"] for x in members} == set(METHODS)
                for r in members]
    if not complete:
        return
    report = summarize(complete)
    checkpoint_csv(output_root / "summary.csv", report)
    winners = [r for r in report if r["joint_best"]]
    if winners:
        checkpoint_csv(output_root / "joint_winners.csv", winners)
    else:
        (output_root / "joint_winners.csv").write_text(
            "task,method,reps,mean_mae,mean_rmse,std_mae,std_rmse,best_mae,best_rmse,joint_best\n"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=TASKS, default=list(TASKS))
    p.add_argument("--data-root", type=Path, default=Path("data/real"))
    p.add_argument("--output-root", type=Path, default=Path("results/real_four_30"))
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--assume-legacy-settings", action="store_true", help="Explicitly allow resuming an older 3-rep output whose manifest did not record every hyperparameter")
    p.add_argument("--lags", type=int, default=3, help="Extra history steps for Beijing")
    p.add_argument("--max-train", type=int, default=None, help="Optional size cap for smoke tests")
    p.add_argument("--reps", type=int, default=30, help="30 replications, matching the FAST-NN FRED-MD experiment count")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--tau", type=float, default=1.5, help="Huber delta in train-standardized target units")
    p.add_argument("--lr", type=float, default=.003)
    p.add_argument("--output-bound", type=float, default=10., help="Bound in standardized target units")
    p.add_argument("--seed", type=int, default=42)
    opt = p.parse_args()
    if min(opt.reps,opt.epochs,opt.patience,opt.width) < 1 or opt.lags < 0 or opt.tau <= 0 or opt.lr <= 0:
        p.error("Invalid hyperparameters")
    torch.set_num_threads(1)
    if opt.download_only:
        for source in sorted({"fred" if t.startswith("fred") else "beijing" for t in opt.datasets}):
            path = download(source, opt.data_root, not opt.no_download)
            print("Downloaded:", path, path.stat().st_size, "bytes")
        return
    # Checkpoint after each fitted method and resume by (task, rep, method).
    # The fixed chronological split and seed+rep initialization preserve the first
    # three fits exactly when extending an existing 3-rep run to 30 reps.
    opt.output_root.mkdir(parents=True, exist_ok=True)
    per_rep_file = opt.output_root / "per_replication.csv"
    manifest_file = opt.output_root / "manifest.json"
    manifest = {
        "tasks": opt.datasets, "source_urls": SOURCES, "data_root": str(opt.data_root),
        "reps": opt.reps, "epochs": opt.epochs, "tau": opt.tau, "seed": opt.seed,
        "lags": opt.lags, "max_train": opt.max_train, "patience": opt.patience,
        "width": opt.width, "lr": opt.lr, "output_bound": opt.output_bound,
        "split": "chronological 60/20/20",
        "selection": "best validation mean(MAE,RMSE)/2; no test tuning",
        "label": "one-step ahead", "frozen_fred_vintage": "2026-09",
        "note": "Not reported thesis results; independently defined forecasting tasks",
        "replication_note": "30 initialization seeds at a fixed chronological split; FAST-NN used 30 repeated random train/validation splits",
    }
    if per_rep_file.exists() and not manifest_file.exists():
        p.error("Found existing per_replication.csv but no manifest.json; use a different --output-root")
    if manifest_file.exists():
        old_manifest = json.loads(manifest_file.read_text())
        comparable = ["tasks", "source_urls", "data_root", "epochs", "tau", "seed",
                      "lags", "max_train", "patience", "width", "lr", "output_bound",
                      "split", "selection", "label", "frozen_fred_vintage"]
        bad = {key: (old_manifest[key], manifest[key]) for key in comparable
               if key in old_manifest and old_manifest[key] != json.loads(json.dumps(manifest[key]))}
        if bad:
            p.error(f"Cannot resume with modified data/training settings: {bad}. Use a new --output-root")
        missing = [key for key in ("lags", "max_train", "patience", "width", "lr", "output_bound")
                   if key not in old_manifest]
        if per_rep_file.exists() and missing and not opt.assume_legacy_settings:
            p.error("Existing legacy run does not record settings " + str(missing) +
                    ". Pass --assume-legacy-settings ONLY if the old run used the same settings, "
                    "or choose a new --output-root")
        if int(old_manifest.get("reps", 0)) > opt.reps:
            p.error("Existing run requested more reps than this invocation; choose another --output-root")
    rows = []
    if per_rep_file.exists():
        with per_rep_file.open(newline="") as f:
            rows = list(csv.DictReader(f))
        required = {"task", "rep", "method", "mae", "rmse", "val_score",
                    "selected_lambda", "best_epoch", "active_parameters", "n", "d"}
        if rows and not required.issubset(rows[0]):
            p.error("Existing per_replication.csv has incompatible columns")
        keys = [(r["task"], int(r["rep"]), r["method"]) for r in rows]
        if len(set(keys)) != len(keys):
            p.error("Existing per_replication.csv contains duplicate task/rep/method rows")
        if any(task not in opt.datasets or rep not in range(opt.reps) or method not in METHODS
               for task, rep, method in keys):
            p.error("Existing per_replication.csv contains incompatible task, rep or method")
        print(f"Resuming from {len(rows)} completed model fits", flush=True)
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    completed = {(r["task"], int(r["rep"]), r["method"]) for r in rows}
    for task in opt.datasets:
        df, y, metadata = prepare_task(task, opt.data_root, opt.lags, not opt.no_download)
        x, yn, raw_y, tr, va, te, mean, sd = split_scale(df, y, opt.max_train)
        print(f"{task}: n={len(raw_y)}, d={x.shape[1]}, train={len(tr)}, valid={len(va)}, test={len(te)}", flush=True)
        for rep in range(opt.reps):
            for method in METHODS:
                key = (task, rep, method)
                if key in completed:
                    continue
                selected = None
                for lam in (LAMBDA_GRID if method.endswith("SPDNN") else (0.,)):
                    val, pred, epoch, actives = train_candidate(
                        x, yn, (tr, va, te), method, lam, opt.seed + rep, opt
                    )
                    if selected is None or val < selected[0]:
                        selected = (val, pred, epoch, actives, lam)
                val, pred, epoch, actives, lam = selected
                mae, rmse = metrics(pred.astype(float) * sd + mean, raw_y[te])
                rows.append({"task": task, "rep": rep, "method": method,
                             "mae": mae, "rmse": rmse, "val_score": val,
                             "selected_lambda": lam, "best_epoch": epoch,
                             "active_parameters": actives, "n": len(raw_y), "d": x.shape[1]})
                completed.add(key)
                checkpoint_csv(per_rep_file, rows)
                print(f"  rep {rep+1}/{opt.reps} {method}: MAE={mae:.5g} RMSE={rmse:.5g}", flush=True)
            # Aggregate only complete four-method replications, even after an interrupt.
            save_partial_summary(rows, opt.output_root)
    if len(completed) != len(opt.datasets) * opt.reps * len(METHODS):
        raise RuntimeError("Missing model replications after run")
    save_partial_summary(rows, opt.output_root)
    all_rows = rows
    print("\nCOMPARISON (two metrics jointly minimal):")
    for item in summarize(all_rows):
        mark = " [JOINT BEST]" if item["joint_best"] else ""
        print(f'{item["task"]:24s} {item["method"]:14s} MAE={item["mean_mae"]:.5g} RMSE={item["mean_rmse"]:.5g}{mark}')
    good = [r["task"] for r in summarize(all_rows) if r["method"] == "Huber-SPDNN" and r["joint_best"]]
    print("Huber-SPDNN wins BOTH metrics on:", good if good else "none in this run")
    print("Output:", (opt.output_root/"summary.csv").resolve())


if __name__ == "__main__":
    main()