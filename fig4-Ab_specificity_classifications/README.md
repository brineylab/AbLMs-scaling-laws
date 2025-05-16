

The fine-tuning scripts: `HD_vs_CoV.py` and `HD_vs_CoV_vs_Flu.py`  can be run using DeepSpeed, as described [here](https://github.com/brineylab/deepspeed/tree/main). Descriptions bellow can be followed for creating environment to run the scripts.

## Prerequisites

1.  **Environment:** Python 3.8+ (virtual environment recommended).
2.  **Dependencies:**  install packages in requirements.txt
3.  **split the data using stratified-kfold-split.ipynb***
    
3.  **Configuration:**
    * Configure Hugging Face Accelerate: `accelerate config`
    * Login to Weights & Biases: `wandb login`

## Data Setup

* The script expects 5 sets of train/test CSV files for cross-validation.
* **Crucially, data paths are hardcoded.** You MUST modify these paths directly in the `main()` function of the script:
    
        * `'/home/jovyan/shared/mahdi/1_projects/model_optimization/02classification/data/5_folded/hd-0_cov-1_train{i}.csv'`
        * `'/home/jovyan/shared/mahdi/1_projects/model_optimization/02classification/data/5_folded/hd-0_cov-1_test{i}.csv'` (where `{i}` is 0 through 4)
  
* CSVs need "h_sequence", "l_sequence", and a label column. 

## Running the Script

Use `accelerate launch` with the script name and provide the `--model` (path to pre-trained model checkpoint) and `--model_name` (for logging) arguments.

**Command Template:**

```bash
accelerate launch YOUR_SCRIPT_NAME.py \
    --model PATH_TO_ESM_MODEL_OR_CHECKPOINT \
    --model_name YOUR_DESCRIPTIVE_MODEL_NAME