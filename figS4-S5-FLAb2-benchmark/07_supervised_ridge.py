#!/usr/bin/env python
"""The supervised head behind S5 Fig: a ridge regression on the frozen embeddings.

This is the probe of Kenlay et al. 2024 (PLOS Comput Biol 20:e1012646, the IgBert / IgT5 paper): a linear
least squares fit with L2 regularisation on mean-pooled embeddings, the regularisation strength chosen by
5-fold cross-validation inside the training part, then refitted on the whole training part. Datasets,
embeddings, split and input standardisation are those of 05_few-shot_mlp.py, so the two heads are fit on
exactly the same rows and their correlations can be compared directly:
  datasets   the 49 FLAb2 datasets with at least 100 antibodies (data/fewshot_rows.csv)
  split      80 / 10 / 10, stratified on 10 quantile bins of the label, seed 42; the validation part is unused
  inputs     standardised with the mean and standard deviation of the training rows
  selection  5-fold cross-validation inside the training part, highest mean R2
  score      Spearman correlation between prediction and label on the test part

Two deliberate differences to Kenlay et al. The outer scheme there is 10-fold cross-validation; here it is
the FLAb2 split, for the comparability described above. And their grid stops at 1 (1, 1e-1, ..., 1e-6, and 0),
which on these standardised embeddings was pressed against its ceiling: the cross-validation chose that top
value in 665 of the 735 fits. The grid is therefore extended upwards to 1e6, where it settles at 1e3 to 1e4 in
most fits. Their range is kept for comparison in the columns spearman_narrow / alpha_narrow; it gives a lower
correlation throughout (0.394 against 0.463 overall) and the same conclusion.

All arithmetic is in float64 and solved in closed form, by singular value decomposition or, for the large
datasets, the eigen-decomposition of the Gram matrix. The result equals sklearn's RidgeCV(alphas, cv=5),
which --check verifies on the grid used for the figure.

Reads  data/fewshot_rows.csv, results/embeddings/<model>.npy, models.yaml
Writes results/ridge_results.csv   model, size, tier, property, source_file, n, n_train, n_test,
       spearman, alpha, spearman_narrow, alpha_narrow, seconds
       The file is appended to and tasks already in it are skipped, so a run can be resumed.

Usage: python 07_supervised_ridge.py --workers 6        (--only 8M_full, --check 40 to compare with sklearn)
"""
import argparse, os, time, warnings
from importlib.machinery import SourceFileLoader
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.join(HERE, "results", "embeddings")
OUT = os.path.join(HERE, "results", "ridge_results.csv")
SEED, INNER = 42, 5
GRIDS = {"main": [float(f"1e{k}") for k in range(6, -7, -1)],       # the grid used for S5 Fig
         "narrow": [float(f"1e{k}") for k in range(0, -7, -1)]}     # the range of Kenlay et al.
ALPHAS = sorted({a for g in GRIDS.values() for a in g}, reverse=True)
COLS = ["model", "size", "tier", "property", "source_file", "n", "n_train", "n_test",
        "spearman", "alpha", "spearman_narrow", "alpha_narrow", "seconds"]
CHECK = False

stratified_80_10_10 = SourceFileLoader(
    "_mlp", os.path.join(HERE, "05_few-shot_mlp.py")).load_module().stratified_80_10_10


def _solver(Xc, yc):
    """alpha -> argmin_w |Xc w - yc|^2 + alpha |w|^2, for centred Xc and yc."""
    n, d = Xc.shape
    if n <= d or n <= 3000:
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        Uty = U.T @ yc
        return lambda a: Vt.T @ ((s / (s * s + a)) * Uty)
    lam, V = np.linalg.eigh(Xc.T @ Xc)
    lam = np.clip(lam, 0.0, None)
    Vtb = V.T @ (Xc.T @ yc)
    return lambda a: V @ (Vtb / (lam + a))


def ridge_cv(Xtr, ytr, Xte):
    """{grid: (chosen alpha, test predictions)}; folds and selection as GridSearchCV(Ridge(), cv=5)."""
    n = len(ytr)
    sizes = np.full(INNER, n // INNER); sizes[: n % INNER] += 1          # KFold without shuffling
    b = np.concatenate([[0], np.cumsum(sizes)])
    r2 = np.zeros((INNER, len(ALPHAS)))
    for k in range(INNER):
        va = np.arange(b[k], b[k + 1]); tr = np.concatenate([np.arange(0, b[k]), np.arange(b[k + 1], n)])
        xm, ym = Xtr[tr].mean(0), ytr[tr].mean()
        w_of = _solver(Xtr[tr] - xm, ytr[tr] - ym)
        Xv, yv = Xtr[va] - xm, ytr[va]
        sst = ((yv - yv.mean()) ** 2).sum()
        for j, a in enumerate(ALPHAS):
            sse = ((yv - (Xv @ w_of(a) + ym)) ** 2).sum()
            r2[k, j] = 1 - sse / sst if sst > 0 else (1.0 if sse == 0 else 0.0)
    mean_r2 = r2.mean(0)
    xm, ym = Xtr.mean(0), ytr.mean()
    w_of = _solver(Xtr - xm, ytr - ym)
    out = {}
    for name, grid in GRIDS.items():
        idx = [ALPHAS.index(a) for a in grid]                 # grid is in descending order: ties -> larger alpha
        best = grid[int(np.argmax(mean_r2[idx]))]
        out[name] = (best, (Xte - xm) @ w_of(best) + ym)
    return out


def _rho(y, p):
    r = spearmanr(y, p).correlation
    return float(r) if np.isfinite(r) else np.nan


def fit_one(task):
    model, size, tier, prop, src, seq_index, y = task
    t0 = time.time()
    distinct, loc = np.unique(seq_index, return_inverse=True)
    Xd = np.load(os.path.join(EMB, f"{model}.npy"), mmap_mode="r")[distinct].astype(np.float64)
    y = np.asarray(y, dtype=float)
    tr, va, te = stratified_80_10_10(y, SEED)
    w = np.bincount(loc[tr], minlength=len(Xd)).astype(np.float64); w /= w.sum()
    mu = (Xd * w[:, None]).sum(0); sd = np.sqrt((((Xd - mu) ** 2) * w[:, None]).sum(0)) + 1e-8
    Xn = (Xd - mu) / sd                                        # float64 throughout
    del Xd
    Xtr, Xte = Xn[loc[tr]], Xn[loc[te]]
    res = ridge_cv(Xtr, y[tr], Xte)
    row = dict(model=model, size=size, tier=tier, property=prop, source_file=src, n=len(y),
               n_train=len(tr), n_test=len(te),
               spearman=_rho(y[te], res["main"][1]), alpha=res["main"][0],
               spearman_narrow=_rho(y[te], res["narrow"][1]), alpha_narrow=res["narrow"][0])
    if CHECK:                                                  # sklearn on the same arrays, same grid
        from sklearn.linear_model import RidgeCV
        m = RidgeCV(alphas=GRIDS["main"], cv=INNER).fit(Xtr, y[tr])
        row["seconds"] = _rho(y[te], m.predict(Xte))           # column "seconds" carries the sklearn value
        row["alpha_narrow"] = float(m.alpha_)                  # and "alpha_narrow" sklearn's alpha
    else:
        row["seconds"] = round(time.time() - t0, 1)
    return row


def main():
    global CHECK
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check", type=int, default=0, help="compare with sklearn RidgeCV on this many tasks (n <= 6000)")
    a = ap.parse_args()
    CHECK = a.check > 0

    cfg = yaml.safe_load(open(os.path.join(HERE, "models.yaml")))["models"]
    rows = pd.read_csv(os.path.join(HERE, "data", "fewshot_rows.csv"))
    done = set()
    if os.path.exists(a.out):
        d = pd.read_csv(a.out); done = set(zip(d.model, d.property, d.source_file))
    tasks = []
    for model in ([m for m in a.only.split(",") if m] or list(cfg)):
        assert os.path.exists(os.path.join(EMB, f"{model}.npy")), f"no embeddings for {model}"
        for (prop, src), g in rows.groupby(["property", "source_file"], sort=True):
            if (model, prop, src) in done:
                continue
            tasks.append((model, cfg[model]["size"], cfg[model]["tier"], prop, src,
                          g.seq_index.values, g.fitness.values))
    if CHECK:
        tasks = [t for t in tasks if len(t[6]) <= 6000]
        tasks = [tasks[i] for i in np.random.default_rng(0).permutation(len(tasks))[: a.check]]
    tasks.sort(key=lambda t: -len(t[6]))                       # longest first
    print(f"{len(tasks)} tasks ({len(done)} already done), {a.workers} workers", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if not os.path.exists(a.out):
        pd.DataFrame(columns=COLS).to_csv(a.out, index=False)
    t0 = time.time()
    with Pool(a.workers, maxtasksperchild=4) as pool:
        for k, r in enumerate(pool.imap_unordered(fit_one, tasks), 1):
            pd.DataFrame([r], columns=COLS).to_csv(a.out, mode="a", header=False, index=False)
            if k % 50 == 0 or k == len(tasks):
                print(f"{k}/{len(tasks)} done, {(time.time() - t0) / 60:.1f} min", flush=True)
    print("done ->", a.out, flush=True)


if __name__ == "__main__":
    main()
