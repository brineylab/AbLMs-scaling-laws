### FLAb2 fitness benchmark (S4 Fig and S5 Fig)

All fifteen models were evaluated on the [FLAb](https://github.com/Graylab/FLAb) antibody fitness
benchmark, zero-shot (pseudo-perplexity, **S4 Fig**) and with a supervised probe trained on frozen
embeddings (**S5 Fig**). Clone the benchmark next to these scripts, or pass `--flab_repo`:

```bash
git clone https://github.com/Graylab/FLAb.git FLAb_repo
```

### Order

```bash
python 01_build-datasets.py                                     # data/flab2_<property>_full.csv
python 02_zero-shot_pseudo-perplexity.py --model 350M_full --dataset binding   # per model x property
python 03_zero-shot_spearman.py                                 # results/per_dataset_spearman.csv  -> S4 Fig
python 04_few-shot_embed.py --gpu 0                             # results/embeddings/<model>.npy
python 07_supervised_ridge.py --workers 6                       # results/ridge_results.csv         -> S5 Fig
```

Step 1 keeps every FLAb2 file that has both a heavy and a light chain, a measured property and at
least 10 antibodies: **141 datasets, 563,844 measurements, seven properties**. Step 2 is the slow
one — it masks every residue of every sequence one at a time — and takes `--nshards`/`--shard` to
spread one (model, property) over several GPUs; `03_zero-shot_spearman.py` merges the shards. The
supervised steps use the 49 of those datasets with at least 100 antibodies, which gives
**15 x 49 = 735** test-set correlations.

### The supervised probe (S5 Fig)

`07_supervised_ridge.py` fits a ridge regression on the standardised mean-pooled embeddings of the
frozen model: the probe of [Kenlay et al. 2024](https://doi.org/10.1371/journal.pcbi.1012646), a
linear least squares fit with L2 regularisation whose strength is chosen by 5-fold cross-validation
inside the training part. The split is FLAb2's own, 80/10/10 stratified on ten quantile bins of the
label with seed 42, and the score is the Spearman correlation on the held-out test part.

Two things differ from Kenlay et al. on purpose. Their outer scheme is 10-fold cross-validation;
here it is the FLAb2 split, so that these numbers sit on the same rows as the two-layer head below
and the two can be compared directly. And their grid stops at 1 (1, 1e-1, ..., 1e-6, and 0), which
on these standardised embeddings turned out to be pressed against its ceiling — the inner
cross-validation picked that top value in 665 of the 735 fits. The grid runs to 1e6 here, where it
settles at 1e3 to 1e4 in most fits. Their range is still reported, in the columns `spearman_narrow`
and `alpha_narrow`; it gives a lower correlation throughout (0.394 against 0.463 overall) and leads
to the same conclusion.

The fit is solved in closed form in float64 and matches `sklearn.linear_model.RidgeCV(alphas, cv=5)`
exactly; `--check N` runs sklearn on N random datasets and reports both.

### FLAb2's own two-layer head (not in the paper)

`05_few-shot_mlp.py` and `06_few-shot_table.py` train the head FLAb2 uses in its own supervised task,
on the same embeddings and the same splits: `Linear(d, 32) -> ReLU -> Dropout(0.2) -> Linear(32, 1)`,
Adam at 1e-4, batch size 8, 240 epochs, no early stopping. They are kept here because the comparison
is worth having — that head reaches 0.311 overall against 0.463 for the ridge probe, with the same
ranking of models and the same conclusion — but S5 Fig is the ridge probe.

```bash
python 05_few-shot_mlp.py --workers 32                          # results/fewshot_results.csv
python 06_few-shot_table.py                                     # sort and check that table
```

That script trains the head in numpy, which is about ten times faster than autograd for batches this
small. `--check` runs the PyTorch version as well; the two draw the same initial weights, batch order
and dropout masks and give the same correlation.

### A note on the input format

These models were pretrained on `heavy<cls><cls>light`, so an antibody is scored here as one sequence
in that format. FLAb2 itself scores the two chains separately, and joins them with a (G4S)3 linker for
its supervised task.
