### AbLMs Pre-training Overview

This repo uses a unified script, `ModelTrainer.py`, to pre-train all AbLMs.

Training is configured via five `train_config_<size>.yaml` files, one for each model size: 8M, 35M, 150M, 350M, and 650M.

Each config supports three dataset sizes: Quarter, Half, and Full. The paths to the dataset sizes can be changed using the `train_file` parameter. The three training sets, the validation set and the test set are `train-eval-test.zip` on [Zenodo](https://doi.org/10.5281/zenodo.15447078).

`batch_size` is per device. Every model was trained with an effective batch size of 128, which is
the number the paper reports, so the value here has to match the number of GPUs you use (32 per
device on four GPUs).
