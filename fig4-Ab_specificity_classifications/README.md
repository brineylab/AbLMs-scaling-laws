Antibody specificity classification (Fig 4, S6 Table).

The fine-tuning scripts `02_HD_vs_CoV.py` (healthy donor vs CoV) and `02_HD_vs_CoV_vs_Flu.py`
(healthy donor vs CoV vs influenza) can be run using DeepSpeed, as described
[here](https://github.com/brineylab/deepspeed/tree/main). Both train only the classification head;
the base model stays frozen.

## Data Setup

Both datasets are on [Zenodo](https://zenodo.org/records/16938681). Unzip them under `data/`,
then make the 5-fold splits with `01_stratified-kfold-split.ipynb`:

| Zenodo archive | put it at | folds written by the notebook |
|---|---|---|
| `HD_vs_COV.csv.zip` | `data/classification/HD_vs_COV.csv` | `data/classification/5_folded/hd-0_cov-1_{train,test}{i}.csv` |
| `hd-0_CoV-1_flu-2.csv.zip` | `data/HDvsFluvsCov/hd-0_CoV-1_flu-2.csv` | `data/HDvsFluvsCov/5_folded/hd-0_CoV-1_flu-2_{train,test}{i}.csv` |

The CSVs have the columns `name`, `h_sequence`, `l_sequence` and `label`. **The labels are
0 = healthy donor, 1 = CoV and 2 = influenza**, which is what the file name says.

The scripts expect to be run from the root of the repository, so that `./data/...` resolves.

## Running the Script

Use `accelerate launch` with the script name and provide the `--model` (path to the pre-trained
model checkpoint, local or a Hugging Face repo id) and `--model_name` (used for logging and in the
result file names) arguments. For example:

```bash
accelerate launch fig4-Ab_specificity_classifications/02_HD_vs_CoV.py \
    --model brineylab/350M_full_checkpoint-500000 \
    --model_name esm_350M_full
```

The binary task trains for 3 epochs at an effective batch size of 128, the three-way task for
5 epochs at an effective batch size of 32 (both with 4 GPUs; the per-device batch sizes in
`def_training_args` assume that). Results land in `results_HDvsCoV/` and `results_HD-Flu-CoV/`,
and `03_plots_HD-CoV.ipynb` / `03_plots_HD-CoV-Flu.ipynb` turn them into the panels of Fig 4.

## A note on the separator

Both classifiers join the heavy and light chain with `<sep>`. That token is not part of the
ESM-2 vocabulary, so it is tokenized as a single `<unk>`. Pretraining and the pairing classifier
(`fig5-...`) use `<cls><cls>` instead. We kept `<sep>` here because it is what the published
classifiers were trained with, and a classifier only has to see the same separator during
training and inference.
