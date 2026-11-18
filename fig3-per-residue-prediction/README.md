### Replicating Figure 3 Analysis

Per-position inference was run using the following repo: https://github.com/smburbach/AbLM-Eval. However, this package is still under development and subject to major changes, therefore a simplified version is also present [here](./ablm_eval/).

The two input datasets are `Per-residue_pred.zip` on [Zenodo](https://doi.org/10.5281/zenodo.15447078);
unzip it inside `data/`, which puts the mutated and unmutated donor tables where
`01_run-inference.ipynb` looks for them. That notebook loads the five Dataset-F models from the
Hugging Face hub. Then run `02_process-results.ipynb` and `03_plots.ipynb`, in that order.
