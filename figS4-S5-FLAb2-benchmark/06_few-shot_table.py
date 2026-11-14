#!/usr/bin/env python
"""Step 3: sort results/fewshot_results.csv and check that it is complete.

735 rows = 15 models x 49 datasets (every FLAb2 dataset of the zero-shot table with at least 100
antibodies). Row order: property, then model (order of models.yaml), then dataset.

Usage: python 06_few-shot_table.py
"""
import os

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROPS = ["binding", "polyreactivity", "aggregation", "thermostability",
         "pharmacokinetics", "expression", "immunogenicity"]
OUT = os.path.join(HERE, "results", "fewshot_results.csv")

models = list(yaml.safe_load(open(os.path.join(HERE, "models.yaml")))["models"])
d = pd.read_csv(OUT)
assert not d.duplicated(["model", "property", "source_file"]).any(), "a task was run twice"
assert set(d.model) == set(models), sorted(set(models) - set(d.model))

zs = pd.read_csv(os.path.join(HERE, "results", "per_dataset_spearman.csv"))
key = zs[zs.n >= 100].drop_duplicates(["property", "source_file"]).set_index(["property", "source_file"]).n
got = d.drop_duplicates(["property", "source_file"]).set_index(["property", "source_file"]).n
assert key.sort_index().equals(got.sort_index()), "datasets or their sizes differ from the zero-shot table"
assert len(d) == len(models) * len(key), (len(d), len(models), len(key))

d["property"] = pd.Categorical(d.property, PROPS, ordered=True)
d["model"] = pd.Categorical(d.model, models, ordered=True)
d = d.sort_values(["property", "model", "source_file"]).reset_index(drop=True)
d.to_csv(OUT, index=False)
print(f"{OUT}: {len(d)} rows, {len(key)} datasets x {len(models)} models, "
      f"{d.spearman.isna().sum()} without a correlation (constant prediction)")
