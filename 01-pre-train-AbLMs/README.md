This repository employs a unified `ModelTrainer.py` script for the pre-training of all Antibody Language Models (AbLMs) developed in this research.

Configuration for the training process is managed through YAML files. We provide five main `config.yaml` files, one for each of the supported model parameter sizes: ***8M, 35M, 150M, 350M, and 650M***.

Within each of these model-specific YAML files, you can select the dataset size for training (Quarter, Half, or Full). The paths to these different dataset sizes are included as commented-out lines under the `train_file` parameter. To use a specific dataset size, simply uncomment the corresponding line in the config.yaml file for your chosen model size.

Base models can be trained from scratch by running `ModelTrainer.py` with an associated `train-config_<modelname>.yaml`, as described [here](https://github.com/brineylab/deepspeed/tree/main).

Training datasets can be downloaded from Zenodo, and model weights are also available on Hugging Face and Zenedo.
