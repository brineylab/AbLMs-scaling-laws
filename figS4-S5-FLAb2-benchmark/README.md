### FLAb2 fitness benchmark (S4 Fig and S5 Fig)

All fifteen models were evaluated on the [FLAb](https://github.com/Graylab/FLAb) antibody fitness
benchmark, zero-shot (pseudo-perplexity, **S4 Fig**) and few-shot (a supervised head on frozen
embeddings, **S5 Fig**). Clone the benchmark next to these scripts, or pass `--flab_repo`:

```bash
git clone https://github.com/Graylab/FLAb.git FLAb_repo
```

### Order

```bash
python 01_build-datasets.py                                     # data/flab2_<property>_full.csv
python 02_zero-shot_pseudo-perplexity.py --model 350M_full --dataset binding   # per model x property
python 03_zero-shot_spearman.py                                 # results/per_dataset_spearman.csv
python 04_few-shot_embed.py --gpu 0                             # results/embeddings/<model>.npy
python 05_few-shot_mlp.py --workers 32                          # results/fewshot_results.csv
python 06_few-shot_table.py                                     # sort and check that table
```

Step 1 keeps every FLAb2 file that has both a heavy and a light chain, a measured property and at
least 10 antibodies: **141 datasets, 563,844 measurements, seven properties**. Step 2 is the slow
one — it masks every residue of every sequence one at a time — and takes `--nshards`/`--shard` to
spread one (model, property) over several GPUs; `03_zero-shot_spearman.py` merges the shards. The
few-shot steps use the 49 of those datasets with at least 100 antibodies, which gives
**15 x 49 = 735** test-set correlations.

### A note on the input format

These models were pretrained on `heavy<cls><cls>light`, so an antibody is scored here as one
sequence in that format. FLAb2 itself scores the two chains separately, and joins them with a
(G4S)3 linker for its supervised task. Everything else follows FLAb2: the 80/10/10 split
stratified on ten quantile bins of the label, mean-pooled frozen embeddings, a two-layer head
(32 units, dropout 0.2), Adam at 1e-4, batch size 8, 240 epochs, and the Spearman correlation on
the held-out test part.

`05_few-shot_mlp.py` trains that head in numpy, which is about ten times faster than autograd for
batches this small. `--check` runs the PyTorch version as well; the two draw the same initial
weights, batch order and dropout masks and give the same correlation.
