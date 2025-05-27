[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://jupyter.org/try)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15447079.svg)](https://doi.org/10.5281/zenodo.15447079)

# Data-optimal scaling laws for natively paired antibody language models

**Motivation**: Antibody language models (AbLMs) play a critical role in exploring the extensive sequence diversity of antibody repertoires, significantly enhancing therapeutic discovery. However, the optimal strategy for scaling these models, particularly concerning the interplay between model size and data availability, remains underexplored, especially in contrast to natural language processing where data is abundant. This study aims to systematically investigate scaling laws in AbLMs to define optimal scaling thresholds and maximize their potential in antibody engineering and discovery.

**Results**: This study pretrained ESM-2 architecture models across five distinct parameterizations (8 million to 650 million weights) and three training data scales (Quarter, Half, and Full datasets, with the full set comprising ~1.6 million paired antibody sequences). Performance was evaluated using cross-entropy loss and downstream tasks, including per-position amino acid identity prediction, antibody specificity classification, and native heavy-light chain pairing recognition. Findings reveal that increasing model size does not monotonically improve performance; for instance, with the full dataset, loss began to increase beyond ~163M parameters. The 350M parameter model trained on the full dataset (350M-F) often demonstrated optimal or near-optimal performance in downstream tasks, such as achieving the highest accuracy in predicting mutated CDRH3 regions. 

**Conclusion**: These results underscore that in data-constrained domains like antibody sequences, strategically balancing model capacity with dataset size is crucial, as simply increasing model parameters without a proportional increase in diverse training data can lead to diminishing returns or even impaired generalization

## Pre-training all AbLMs
This repository employs a unified `ModelTrainer.py` script for the pre-training of all Antibody Language Models (AbLMs) developed in this study.

Configuration for the training process is managed through YAML files. We provide five main `config.yaml` files, one for each of the supported model parameter sizes: ***8M, 35M, 150M, 350M, and 650M*** in this folder: 01-pre-train-AbLMs.

Within each of these model-specific YAML files, you can select the dataset size for training (Quarter, Half, or Full). The paths to these different dataset sizes can be changed in `train_file` parameter. To use a specific dataset size, simply uncomment the corresponding line in the config.yaml file for your chosen model size saved in your specific directory.

Base models can be trained from scratch by running `ModelTrainer.py` with an associated `train-config_<modelname>.yaml`, as described [here](https://github.com/brineylab/deepspeed/tree/main).

Training datasets can be downloaded from Zenodo and weights for the pre-trained model checkpoints used in the paper can also be downloaded from [Zenodo](https://zenodo.org/records/15447079) and model weights for CurrAb are avaliable on [Hugging Face](https://huggingface.co/collections/brineylab/ablms-scaling-laws-6824e4beaabf4b16107cac4f)


## Citation


The current version of the datasets used for pre-training and classifier head fine-tuning (v2025.05.10) can be cited as:

```
Shafiei Neyestanak, M., & Briney, B. (2025). Scaling laws in antibody language models reveal data-constrained optima (v2025.05.16) [Data set].
Zenodo. https://doi.org/10.5281/zenodo.15447079
``` 
