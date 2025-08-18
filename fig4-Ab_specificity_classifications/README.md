The fine-tuning scripts: `02_HD_vs_CoV.py` and `02_HD_vs_CoV_vs_Flu.py`  can be run using DeepSpeed, as described [here](https://github.com/brineylab/deepspeed/tree/main).

## Data Setup

* The script expects 5 sets of train/test CSV files for cross-validation. Split the data using `01_stratified-kfold-split.ipynb`.
* **Crucially, data paths are hardcoded.** You MUST modify these paths directly in the `main()` function of the script:
    
        * `'/home/jovyan/shared/mahdi/1_projects/model_optimization/02classification/data/5_folded/hd-0_cov-1_train{i}.csv'`
        * `'/home/jovyan/shared/mahdi/1_projects/model_optimization/02classification/data/5_folded/hd-0_cov-1_test{i}.csv'` (where `{i}` is 0 through 4)
  
* CSVs need "h_sequence", "l_sequence", and a label column. 

## Running the Script

Use `accelerate launch` with the script name and provide the `--model` (path to pre-trained model checkpoint) and `--model_name` (for logging) arguments. For example:

```bash
accelerate launch 02_HD_vs_CoV.py \
    --model PATH_TO_ESM_MODEL_OR_CHECKPOINT \
    --model_name YOUR_DESCRIPTIVE_MODEL_NAME
```