[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://jupyter.org/try)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15447079.svg)](https://doi.org/10.5281/zenodo.15447079)

# Data-optimal scaling laws for natively paired antibody language models

**Motivation**: Antibody language models (AbLMs) play a critical role in exploring the extensive sequence diversity of antibody repertoires, significantly enhancing therapeutic discovery. However, the optimal strategy for scaling these models, particularly concerning the interplay between model size and data availability, remains underexplored, especially in contrast to natural language processing where data is abundant. This study aims to systematically investigate scaling laws in AbLMs to define optimal scaling thresholds and maximize their potential in antibody engineering and discovery.

**Results**: This study pretrained ESM-2 architecture models across five distinct model sizes (8 million to 650 million parameters) and three training data scales (Quarter, Half, and Full datasets, with the full set comprising ~1.6 million paired antibody sequences). Performance was evaluated using cross-entropy loss and downstream tasks, including per-position amino acid prediction, antibody specificity classification, and native heavy-light chain pairing recognition. Findings reveal that increasing model size does not monotonically improve performance. For instance, with the full dataset, the optimal model size is predicted to be around ~152M parameters. The 350M parameter model trained on the full dataset (350M-F) often demonstrated optimal or near-optimal performance in downstream tasks, such as achieving the highest accuracy in predicting mutated CDRH3 regions. 

**Conclusion**: These results underscore that in data-constrained domains like paired AbLMs, strategically balancing model capacity with dataset size is crucial, as simply increasing model parameters without a proportional increase in diverse training data can lead to diminishing returns or even impaired generalization. Our findings also underscore the importance of generating additional high-quality, paired antibody sequence data to improve AbLM performance.

## Pretraining AbLMs
This repository contains all the code required for pre-training the Antibody Language Models (AbLMs) developed in this study in this folder: [01-pre-train-AbLMs](./01-pre-train-AbLMs/).

Models can be trained from scratch by running `ModelTrainer.py` with an associated `train-config_<modelname>.yaml`, as described [here](https://github.com/brineylab/deepspeed/tree/main).

Configuration for the training process is managed through YAML files. We provide five main `config.yaml` files, one for each of the supported model parameter sizes: ***8M, 35M, 150M, 350M, and 650M***. Within each of these model-specific YAML files, you can select the dataset size for training (Quarter, Half, or Full).

Training datasets and weights for the pre-trained models can be downloaded from [Zenodo](https://zenodo.org/records/16938681). All model weights are also available on [Hugging Face](https://huggingface.co/collections/brineylab/ablms-scaling-laws-6824e4beaabf4b16107cac4f).

## Citation

The current version of the datasets used for pre-training and classifier head fine-tuning (v2025.08.26) can be cited as:

```
Shafiei Neyestanak, M., & Briney, B. (2025). Data-optimal scaling of paired antibody language models (v2025.08.26) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.16938681
``` 
