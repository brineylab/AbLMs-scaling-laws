## how to run binary classification (CoV specific Abs vs. Healthy donor Abs)

ise 

The fine-tuning script `HD_vs_CoV.py` can be run with its associated config file (CoV-classification_train-config.yaml) using DeepSpeed, as described here.

accelerate launch HD_vs_CoV.py \
    --model /path/to/your/model/checkpoints/ \
    --model_name <model_name>
