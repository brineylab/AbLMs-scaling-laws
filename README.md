[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://jupyter.org/try)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16938681.svg)](https://doi.org/10.5281/zenodo.16938681)

# Data-optimal scaling of paired antibody language models

Scaling laws for large language models in natural language domains are typically derived under the assumption that performance is primarily compute-constrained. In contrast, antibody language models (AbLMs) trained on paired sequences are primarily data-limited, thus requiring different considerations. To explore how model size and data scale affect AbLM performance, we trained 15 AbLMs across all pairwise combinations of five model sizes and three training data sizes. From these experiments, we derive an AbLM-specific scaling law and estimate that training a data-optimal AbLM equivalent of the highly performant 650M-parameter ESM-2 protein language model would require ~5.2 million paired antibody sequences. Evaluation on multiple downstream classification tasks revealed that significant performance gains emerged only with sufficiently large model size, suggesting that in data-limited domains, improved performance depends jointly on both model scale and data volume.

## What is in this repository

| folder | what it makes |
|---|---|
| [`01-pre-train-AbLMs`](./01-pre-train-AbLMs/) | the fifteen pretrained AbLMs |
| [`fig2-CE loss curve plots`](./fig2-CE%20loss%20curve%20plots/) | cross-entropy loss per model, the FixedData profiles and the scaling law (Fig 2, S3 Table) |
| [`fig3-per-residue-prediction`](./fig3-per-residue-prediction/) | per-position amino-acid prediction (Fig 3) |
| [`fig4-Ab_specificity_classifications`](./fig4-Ab_specificity_classifications/) | antibody specificity classification (Fig 4, S6 Table) |
| [`fig5-native vs. shuffled pairing classification`](./fig5-native%20vs.%20shuffled%20pairing%20classification/) | native vs. shuffled heavy/light pairing (Fig 5, S7 Table) |
| [`figS4-S5-FLAb2-benchmark`](./figS4-S5-FLAb2-benchmark/) | the FLAb2 fitness benchmark, zero-shot and few-shot (S4 Fig, S5 Fig) |

Each folder has its own README with the order to run things in and where its data comes from.
`requirements.txt` pins the versions these results were produced with.

## Models

All fifteen models are on the [Hugging Face hub](https://huggingface.co/collections/brineylab/ablms-scaling-laws)
and can be loaded by name, for example:

```python
from transformers import EsmTokenizer, EsmForMaskedLM

model = EsmForMaskedLM.from_pretrained("brineylab/350M_full_checkpoint-500000")
tokenizer = EsmTokenizer.from_pretrained("brineylab/350M_full_checkpoint-500000")

# a paired antibody is one string: the two chains joined by two <cls> tokens
inputs = tokenizer(heavy + "<cls><cls>" + light, return_tensors="pt")
```

The checkpoint of each model is the step at which the evaluation loss began diverging from the
training loss (S1 Table); it is part of the repository name:

| | Dataset-Q | Dataset-H | Dataset-F |
|---|---|---|---|
| **8M** | `8M_quarter_checkpoint-425000` | `8M_half_checkpoint-435000` | `8M_full_checkpoint-500000` |
| **35M** | `35M_quarter_checkpoint-240000` | `35M_half_checkpoint-430000` | `35M_full_checkpoint-500000` |
| **150M** | `150M_quarter_checkpoint-165000` | `150M_half_checkpoint-330000` | `150M_full_checkpoint-500000` |
| **350M** | `350M_quarter_checkpoint-155000` | `350M_half_checkpoint-300000` | `350M_full_checkpoint-500000` |
| **650M** | `650M_quarter_checkpoint-130000` | `650M_half_checkpoint-330000` | `650M_full_checkpoint-395000` |

A note on the separator: pretraining, the per-residue inference, the pairing classifier and the
FLAb2 evaluation all join the two chains with `<cls><cls>`. The two specificity classifiers of
Fig 4 use `<sep>`, which is not part of the ESM-2 vocabulary and is tokenized as a single
`<unk>`; that is what those classifiers were trained with, and their README says so.

## Data

Every dataset is on [Zenodo](https://zenodo.org/records/16938681). Each archive belongs to one
part of the analysis:

| archive | used by |
|---|---|
| `train-eval-test.zip` | pretraining (`01-pre-train-AbLMs`) |
| `Fixed_data_profiles.zip` | the ten donor test sets of the loss evaluation (`fig2-...`) |
| `Per-residue_pred.zip` | per-residue prediction (`fig3-...`) |
| `HD_vs_COV.csv.zip`, `hd-0_CoV-1_flu-2.csv.zip` | specificity classification (`fig4-...`) |
| `Native_paired.zip` | pairing classification (`fig5-...`) |

The FLAb2 benchmark is not ours and stays where it is published, at
[Graylab/FLAb](https://github.com/Graylab/FLAb).

## Pretraining AbLMs

Models can be trained from scratch by running `ModelTrainer.py` with an associated `train_config_<size>.yaml`, as described [here](https://github.com/brineylab/deepspeed/tree/main).

Configuration for the training process is managed through YAML files. We provide five of them, one for each of the supported model parameter sizes: ***8M, 35M, 150M, 350M, and 650M***. Within each of these model-specific YAML files, you can select the dataset size for training (Quarter, Half, or Full).

## Citation

The Scaling Law  paper has been published as a [preprint on biorxiv](https://www.biorxiv.org/content/10.1101/2025.09.02.673765v1), and can be cited as: 
```
Neyestanak MS, Burbach SM, Ng K, Gangavarapu P, Hurtado J, Magura J, Ismail N, Muema D, 
Ndung'u T, Ward A, Briney B. Data-optimal scaling of paired antibody language models. 
bioRxiv. 2025:2025-09.
```


The current version of the datasets used for pre-training and classifier head fine-tuning (v2025.08.26) can be cited as:

```
Shafiei Neyestanak, M., & Briney, B. (2025). Data-optimal scaling of paired antibody language 
models (v2025.08.26) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.16938681
``` 
