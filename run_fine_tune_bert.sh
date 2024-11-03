#!/bin/bash

#SBATCH --job-name=log_anomaly_detection_sft_bert
#SBATCH --mail-type=ALL
#SBATCH --mail-user=giorgio24@ru.is
#SBATCH --partition=gpu-1xA100
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00         
#SBATCH --hint=nomultithread    
#SBATCH --output=/hpchome/giorgio24/Log-Anomaly-Detection-via-LLMs/logs/bert_fine_tune_output.out
#SBATCH --error=/hpchome/giorgio24/Log-Anomaly-Detection-via-LLMs/logs/bert_fine_tune_errors.err
#SBATCH --chdir=/hpchome/giorgio24/Log-Anomaly-Detection-via-LLMs

# Activate virtual environment if it doesn't exist
VENV_PATH="/hpchome/giorgio24/Log-Anomaly-Detection-via-LLMs/venv"
# Add the current directory to PYTHONPATH

source $VENV_PATH/bin/activate

# Set environment variables
export PYTHONPATH=/hpchome/giorgio24/Log-Anomaly-Detection-via-LLMs

# Run the fine-tune script
python scripts/fine_tune_bert.py
