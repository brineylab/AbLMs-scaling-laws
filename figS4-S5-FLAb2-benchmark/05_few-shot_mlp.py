#!/usr/bin/env python
"""Step 2 of the FLAb2 few-shot evaluation: a small regression head on the frozen embeddings.

Settings are those of FLAb2's own few-shot code (models/ft_*_score.py `train_predictor` and
ft_scoring_*.py in the FLAb repository):
  split      80 / 10 / 10 (train / validation / test), stratified on 10 quantile bins of the label, seed 42
  inputs     standardised with the mean and standard deviation of the training rows
  head       Linear(d, 32) -> ReLU -> Dropout(0.2) -> Linear(32, 1)
  training   Adam (learning rate 1e-4, epsilon 1e-7), mean squared error, batch size 8, 240 epochs,
             no early stopping (the validation part is not used), labels not rescaled
  score      Spearman correlation between prediction and label on the test part
One difference to FLAb2 is in step 1, not here: the two chains are joined by two <cls> tokens, as in the
pretraining of these models (FLAb2 joins them with a glycine-serine linker for ESM-2).

One (model, dataset) pair is one task; tasks run in parallel, one thread each. Rows with the same sequence share
one embedding row, so the large datasets fit in memory; every row still counts in the split, in the
standardisation and in training.

Reads  data/fewshot_rows.csv, results/embeddings/<model>.npy, models.yaml
Writes results/fewshot_results.csv   model, size, tier, property, source_file, n, n_train, n_test, spearman, seconds
       The file is appended to and tasks already in it are skipped, so a run can be resumed.

Usage: python 05_few-shot_mlp.py --workers 56          (--only "8M_full" and --max_n for tests)
"""
import argparse, os, time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.join(HERE, "results", "embeddings")
OUT = os.path.join(HERE, "results", "fewshot_results.csv")
EPOCHS, BATCH, LR, DENSE, DROPOUT, SEED, EPS = 240, 8, 1e-4, 32, 0.2, 42, 1e-7
CHECK = False
COLS = ["model", "size", "tier", "property", "source_file", "n", "n_train", "n_test", "spearman", "seconds"]


class EmbPredictor(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, DENSE), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(DENSE, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _split(idx, test_size, strat, seed):
    try:
        return train_test_split(idx, test_size=test_size, stratify=strat, random_state=seed)
    except ValueError:                      # a bin with a single member -> plain random split
        return train_test_split(idx, test_size=test_size, random_state=seed)


def stratified_80_10_10(y, seed):
    bins = pd.qcut(pd.Series(y), q=10, duplicates="drop").astype(str)
    idx = np.arange(len(y))
    tr, tmp = _split(idx, 0.2, bins, seed)
    te, va = _split(tmp, 0.5, bins.iloc[tmp], seed)
    return tr, va, te


def _prepare(Xd, loc, y, seed):
    """Split, input standardisation and the freshly initialised head: shared by both implementations."""
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = stratified_80_10_10(y, seed)
    w = np.bincount(loc[tr], minlength=len(Xd)).astype(np.float64); w /= w.sum()      # training rows, each counted
    mu = (Xd * w[:, None]).sum(0); sd = np.sqrt((((Xd - mu) ** 2) * w[:, None]).sum(0)) + 1e-8
    Xn = ((Xd - mu) / sd).astype(np.float32)
    return tr, te, Xn, EmbPredictor(Xd.shape[1])


def train_eval_torch(Xd, loc, y, seed):
    """Reference implementation, straight PyTorch autograd."""
    tr, te, Xn, m = _prepare(Xd, loc, y, seed)
    Xn = torch.tensor(Xn); ltr = torch.tensor(loc[tr]); ytr = torch.tensor(y[tr], dtype=torch.float32)
    opt = torch.optim.Adam(m.parameters(), lr=LR, eps=EPS)
    lossf = nn.MSELoss()
    n = len(tr)
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        m.train()
        for s in range(0, n, BATCH):
            b = perm[s:s + BATCH]
            opt.zero_grad()
            loss = lossf(m(Xn[ltr[b]]), ytr[b])
            loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(Xn[torch.tensor(loc[te])]).numpy()
    rho = spearmanr(y[te], pred).correlation
    return (float(rho) if np.isfinite(rho) else np.nan), len(tr), len(te)


def train_eval(Xd, loc, y, seed):
    """The same training written out in numpy (about ten times faster for these tiny batches).

    Same split, same initial weights, same batch order and the same dropout masks as the PyTorch version: the
    initial weights, the permutations and the masks are drawn from PyTorch's generator in the same order.
    `--check` runs both versions and compares the results.
    """
    tr, te, Xn, m = _prepare(Xd, loc, y, seed)
    W1 = m.net[0].weight.detach().numpy().copy(); b1 = m.net[0].bias.detach().numpy().copy()
    w2 = m.net[3].weight.detach().numpy().copy()[0]; b2 = m.net[3].bias.detach().numpy().copy()
    params = [W1, b1, w2, b2]; mom = [np.zeros_like(p) for p in params]; var = [np.zeros_like(p) for p in params]
    ltr = loc[tr]; ytr = y[tr].astype(np.float32); keep = np.float32(1 - DROPOUT)
    beta1, beta2, t, n = 0.9, 0.999, 0, len(tr)
    for _ in range(EPOCHS):
        perm = torch.randperm(n).numpy()
        for s in range(0, n, BATCH):
            b = perm[s:s + BATCH]; x = Xn[ltr[b]]; yb = ytr[b]; B = np.float32(len(b))
            mask = torch.empty(len(b), DENSE).bernoulli_(1 - DROPOUT).numpy() / keep
            h = x @ W1.T + b1; a = np.maximum(h, 0) * mask
            out = a @ w2 + b2[0]
            g_out = (2 / B) * (out - yb)                                   # d(mean squared error) / d(out)
            g_a = np.outer(g_out, w2) * mask * (h > 0)
            grads = [g_a.T @ x, g_a.sum(0), g_out @ a, np.array([g_out.sum()], dtype=np.float32)]
            t += 1; c1 = 1 - beta1 ** t; c2 = np.sqrt(1 - beta2 ** t)
            for p, g, mo, va_ in zip(params, grads, mom, var):
                mo *= beta1; mo += (1 - beta1) * g
                va_ *= beta2; va_ += (1 - beta2) * g * g
                p -= (LR / c1) * mo / (np.sqrt(va_) / c2 + EPS)
    xt = Xn[loc[te]]
    pred = np.maximum(xt @ W1.T + b1, 0) @ w2 + b2[0]
    rho = spearmanr(y[te], pred).correlation
    return (float(rho) if np.isfinite(rho) else np.nan), len(tr), len(te)


def fit_one(task):
    torch.set_num_threads(1)
    model, size, tier, prop, src, seq_index, y = task
    t0 = time.time()
    distinct, loc = np.unique(seq_index, return_inverse=True)
    Xd = np.load(os.path.join(EMB, f"{model}.npy"), mmap_mode="r")[distinct].astype(np.float64)
    y = np.asarray(y, dtype=float)
    if CHECK:                                   # run the PyTorch reference as well and report both
        ref = train_eval_torch(Xd, loc, y, SEED)[0]
        rho, ntr, nte = train_eval(Xd, loc, y, SEED)
        return dict(model=model, size=size, tier=tier, property=prop, source_file=src, n=len(y), n_train=ntr, n_test=nte,
                    spearman=rho, seconds=ref)       # column "seconds" carries the reference value in check mode
    rho, ntr, nte = train_eval(Xd, loc, y, SEED)
    return dict(model=model, size=size, tier=tier, property=prop, source_file=src, n=len(y), n_train=ntr, n_test=nte,
                spearman=rho, seconds=round(time.time() - t0, 1))


def main():
    global CHECK, SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=56); ap.add_argument("--only", default=""); ap.add_argument("--max_n", type=int, default=0)
    ap.add_argument("--out", default=OUT); ap.add_argument("--check", action="store_true", help="also run the PyTorch reference")
    ap.add_argument("--seed", type=int, default=SEED, help="seed of the split and of the head (42 = the FLAb2 setting, used for the table)")
    a = ap.parse_args()
    CHECK = a.check; SEED = a.seed
    cfg = yaml.safe_load(open(os.path.join(HERE, "models.yaml")))["models"]
    rows = pd.read_csv(os.path.join(HERE, "data", "fewshot_rows.csv"))
    done = set()
    if os.path.exists(a.out):
        d = pd.read_csv(a.out); done = set(zip(d.model, d.property, d.source_file))
    tasks = []
    for model in ([m for m in a.only.split(",") if m] or list(cfg)):
        assert os.path.exists(os.path.join(EMB, f"{model}.npy")), f"no embeddings for {model}"
        for (prop, src), g in rows.groupby(["property", "source_file"], sort=True):
            if (model, prop, src) in done or (a.max_n and len(g) > a.max_n):
                continue
            tasks.append((model, cfg[model]["size"], cfg[model]["tier"], prop, src, g.seq_index.values, g.fitness.values))
    tasks.sort(key=lambda t: -len(t[6]))                      # longest first
    print(f"{len(tasks)} tasks ({len(done)} already done), {a.workers} workers", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if not os.path.exists(a.out):
        pd.DataFrame(columns=COLS).to_csv(a.out, index=False)
    t0 = time.time()
    with Pool(a.workers, maxtasksperchild=4) as pool:
        for k, r in enumerate(pool.imap_unordered(fit_one, tasks), 1):
            pd.DataFrame([r])[COLS].to_csv(a.out, mode="a", header=False, index=False)
            if k % 50 == 0 or k == len(tasks):
                print(f"{k}/{len(tasks)} done, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
