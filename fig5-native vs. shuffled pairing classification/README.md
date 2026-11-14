### Replicating Figure 5 Analysis
This pair classification task was introduced in [Ng and Briney 2025](https://doi.org/10.1016/j.patter.2025.101239).

The fine-tuning script `02_pair_classification-5itr.py` can be run with its associated config file (`pair-class_train-config.yaml`) using DeepSpeed, as described [here](https://github.com/brineylab/deepspeed/tree/main).

### Data

The five folds are `Native_paired.zip` on [Zenodo](https://zenodo.org/records/16938681); unzip it
so that the CSVs sit at `data/train-test_splits/native-0_shuffled-1_{train,test}{i}.csv`, which is
the `data_path` in the config. They were made by `01_kfold-split.ipynb` from the shuffled dataset,
so that notebook only has to be run if you build the pairs yourself.

`04_plots.ipynb` also uses the annotated repertoires the pairs were drawn from
(`data/all_combined.csv` and the two `data/HDfluJOY_SAHD_*_AbLM.csv` tables), which carry the
V-gene mutation counts behind Fig 5c. Those are not part of the Zenodo archive.

### Order

1. `02_pair_classification-5itr.py` — one run per model; metrics go to `results/`, the fine-tuned
   heads to `paired_clssification_models/`.
2. `03_inference.ipynb` — per-sequence predictions of the test folds, into `results/`.
3. `04_plots.ipynb` — the panels of Fig 5.
