#!/bin/bash

# #############################################################################
# Bash Script to Run HD_vs_CoV.py for Ab binary classification and three way classification (HD vs CoV vs Flu)
# #############################################################################

# --- Configuration ---
# !!! IMPORTANT: Users MUST modify these variables below !!!

# 1. PYTHON_SCRIPT_NAME: The name of your Python script.
#    HD_vs_CoV.py OR HD_vs_CoV_vs_Flu.py
PYTHON_SCRIPT_NAME="YOUR_PYTHON_SCRIPT_NAME.py"

# 2. MODEL_PATH: Full path to the pre-trained AbLM checkpoint directory
#    or a Hugging Face model hub identifier.
#    Example (local path): "/path/to/your/model_checkpoint"

MODEL_PATH="PATH_TO_YOUR_MODEL_CHECKPOINT"

# 3. MODEL_NAME_TAG: A descriptive name/tag for your model run.
#    This will be used for naming WandB runs and the output CSV file.
#    Example: "150M_full_b128"
MODEL_NAME_TAG="YOUR_DESCRIPTIVE_MODEL_RUN_NAME"

# --- (Optional) Advanced Configuration ---
# If you need to set specific environment variables for WandB or other tools,
# you can uncomment and set them here. The Python script already sets
# WANDB_PROJECT, WANDB_RUN_GROUP, and WANDB_JOB_TYPE.


# --- Prerequisites Check (User Reminder) ---
echo "------------------------------------------------------------------------"
echo "Reminder: Before running this script, ensure you have:"
echo "1. Activated your Python virtual environment with all dependencies installed (see requirements.txt)."
echo "2. Configured Hugging Face Accelerate ('accelerate config')."
echo "3. Logged into Weights & Biases ('wandb login')."
echo "4. Correctly set up your data paths inside the Python script: ${PYTHON_SCRIPT_NAME}"
echo "   (Search for 'data_files = DatasetDict({' in the script)."
echo "5. Ensured the output directory for results is writable:"
echo "   (Search for 'results.to_csv(' in the script)."
echo "------------------------------------------------------------------------"
echo ""

# --- Validation (Basic) ---
if [ "$PYTHON_SCRIPT_NAME" == "YOUR_PYTHON_SCRIPT_NAME.py" ]; then
    echo "ERROR: Please set the PYTHON_SCRIPT_NAME variable in this script."
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT_NAME" ]; then
    echo "ERROR: Python script '$PYTHON_SCRIPT_NAME' not found. Please check the name and path."
    exit 1
fi

if [ "$MODEL_PATH" == "PATH_TO_YOUR_ESM_MODEL_OR_CHECKPOINT" ]; then
    echo "ERROR: Please set the MODEL_PATH variable in this script."
    exit 1
fi

if [ "$MODEL_NAME_TAG" == "YOUR_DESCRIPTIVE_MODEL_RUN_NAME" ]; then
    echo "ERROR: Please set the MODEL_NAME_TAG variable in this script."
    exit 1
fi

# --- Execution ---
echo "Starting the fine-tuning process..."
echo "Python Script: $PYTHON_SCRIPT_NAME"
echo "Model Path/ID: $MODEL_PATH"
echo "Model Name Tag: $MODEL_NAME_TAG"
echo ""

accelerate launch "$PYTHON_SCRIPT_NAME" \
    --model "$MODEL_PATH" \
    --model_name "$MODEL_NAME_TAG"

# --- Completion ---
echo ""
echo "------------------------------------------------------------------------"
echo "Script execution finished."
echo "Check Weights & Biases for logs and your specified output directory for results."
echo "------------------------------------------------------------------------"

# To make this script executable:
# chmod +x classification_finetuning.sh
#
# To run it:
# ./classification_finetuning.sh
