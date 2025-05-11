# Scaling laws in Antibody Language Models: optimizing pretraining model size given data availability

Antibody Language Models (AbLMs) are essential for exploring the vast landscape of antibody sequences, holding significant promise for therapeutic discovery. However, a key challenge lies in understanding how to effectively scale these models. Unlike natural language processing where data is abundant, antibody sequence data is relatively limited, and the optimal balance between increasing model complexity and the availability of training data for AbLMs remains largely unexplored. This knowledge gap hinders the development of more performant and generalizable antibody models.

This work addresses this challenge by systematically investigating ***AbLMs scaling laws***. We trained 15 different Ab-specific large language models of varying sizes on different scales of antibody sequence data. Our findings demonstrate that simply increasing model size does not guarantee improved performance; instead, there is a critical interplay between model capacity and data availability. The results highlight that optimal performance requires a strategic balance, indicating that scaling strategies for AbLMs in data-constrained settings must differ from those in data-rich domains. This study provides crucial insights to guide the future design and training of more effective AbLMs for antibody engineering and discovery.

## Pre-training all AbLMs
Base models can be trained by running `ModelTrainer.py` with a matching `config.yaml` file, as detailed in [here](https://github.com/brineylab/deepspeed/tree/main).
Weights for the pre-trained model checkpoints used in the paper can also be downloaded from [Zenodo] ().


## Citation


The current version of the datasets used for pre-training and classifier head fine-tuning (v2025.05.10) can be cited as:


