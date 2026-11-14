#!/usr/bin/env python
"""Turn the raw pseudo-perplexities into the zero-shot table behind S4 Fig.

One row per (property, model, dataset): 141 datasets x 15 models = 2,115 rows, with the columns
    property, model, source_file, n, rho, absrho, size, tier
For each property and model, the shards written by 02_zero-shot_pseudo-perplexity.py
(results/ppl/<property>/<model>[_shardKofN]_pseudo-ppl.csv) are concatenated, rows whose
pseudo_ppl or fitness is missing are dropped, and for every source_file (one FLAb2 dataset) with
at least 10 sequences we take rho = spearmanr(pseudo_ppl, fitness), absrho = |rho| and
n = the number of sequences. size and tier come from models.yaml.

Usage:
    python 03_zero-shot_spearman.py
"""
import os, re, glob, argparse

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
PPL = os.path.join(HERE, "results", "ppl")
PROPS = ["binding", "expression", "thermostability", "aggregation",
         "pharmacokinetics", "polyreactivity", "immunogenicity"]
MIN_N = 10
COLS = ["property", "model", "source_file", "n", "rho", "absrho", "size", "tier"]


def shard_files(prop, model):
    """The result files of one (property, model): one unsharded file, or all N of shard0..N-1."""
    d = os.path.join(PPL, prop)
    files = sorted(glob.glob(os.path.join(d, f"{model}_pseudo-ppl.csv")) +
                   glob.glob(os.path.join(d, f"{model}_shard*of*_pseudo-ppl.csv")))
    tags = [re.search(r"_shard(\d+)of(\d+)_pseudo-ppl\.csv$", f) for f in files]
    if not files:
        raise SystemExit(f"{prop}/{model}: no result file in {d}")
    if any(t is None for t in tags):
        complete = len(files) == 1
    else:
        n = {int(t.group(2)) for t in tags}
        complete = len(n) == 1 and sorted(int(t.group(1)) for t in tags) == list(range(n.pop()))
    if not complete:
        raise SystemExit(f"{prop}/{model}: incomplete or mixed shard set ({len(files)} files)")
    return files


def load(prop, model, files):
    """Merge the shards and restore the row order of data/flab2_<prop>_full.csv (seq_id)."""
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sort_values("seq_id", kind="stable").reset_index(drop=True)
    if not ((df["model"] == model).all() and (df["property"] == prop).all()):
        raise SystemExit(f"{prop}/{model}: model/property columns disagree with the file names")
    if not np.array_equal(df["seq_id"].to_numpy(), np.arange(len(df))):
        raise SystemExit(f"{prop}/{model}: seq_id is not 0..{len(df) - 1} (rows missing or repeated)")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "per_dataset_spearman.csv"),
                    help="output CSV (default: results/per_dataset_spearman.csv)")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing an existing output")
    a = ap.parse_args()
    if os.path.exists(a.out) and not a.overwrite:
        ap.error(f"{a.out} already exists; use --out <other file>, or --overwrite to replace it")
    models = yaml.safe_load(open(os.path.join(HERE, "models.yaml")))["models"]

    rows, n_files, n_seqs = [], 0, 0
    for prop in PROPS:
        used = set()
        for model, m in models.items():
            files = shard_files(prop, model)
            used.update(files)
            df = load(prop, model, files)
            n_seqs += len(df)
            df = df.dropna(subset=["pseudo_ppl", "fitness"])
            for src, d in df.groupby("source_file"):          # sorted by source_file
                if len(d) < MIN_N:
                    continue
                rho = float(spearmanr(d["pseudo_ppl"], d["fitness"])[0])
                rows.append(dict(property=prop, model=model, source_file=src, n=len(d),
                                 rho=rho, absrho=abs(rho), size=m["size"], tier=m["tier"]))
        extra = set(glob.glob(os.path.join(PPL, prop, "*.csv"))) - used
        if extra:
            raise SystemExit(f"{prop}: {len(extra)} file(s) of a model that is not in "
                             f"models.yaml, e.g. {sorted(extra)[0]}")
        n_files += len(used)

    table = pd.DataFrame(rows, columns=COLS)
    per_ds = table.groupby(["property", "source_file"])["n"].agg(["nunique", "size"])
    if (per_ds["nunique"] != 1).any() or (per_ds["size"] != len(models)).any():
        raise SystemExit("the models do not all have the same datasets with the same n: "
                         "results/ppl/ is incomplete")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    table.to_csv(a.out, index=False)
    print(f"read {n_files} files, {n_seqs:,} sequence scores ({len(models)} models)")
    print(f"wrote {len(table)} rows ({len(per_ds)} datasets x {len(models)} models) -> {a.out}")


if __name__ == "__main__":
    main()
