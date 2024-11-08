#!/bin/bash

#SBATCH --job-name=log_anomaly_detection_sft_llama
#SBATCH --mail-type=ALL
#SBATCH --mail-user=giorgio24@ru.is
#SBATCH --partition=gpu-8xA100
#SBATCH --ntasks-per-node=48
#SBATCH --mem=64G
#SBATCH --time=0-20:00:00         
#SBATCH --hint=nomultithread    
#SBATCH --output=/hpchome/giorgio24/Log-Anomaly-Detection-via-LLMs/logs/llama_fine_tune_output.out
#SBATCH --error=/hpchome/giorgio24/Log-Anomaly-Detection-via-LLMs/logs/llama_fine_tune_errors.err
#SBATCH --chdir=/hpchome/giorgio24/Log-Anomaly-Detection-via-LLMs

# Loading necessary modules
ml load Python
ml use /opt/ohpc/pub/modulefiles
ml load nvidia/nvhpc/22.3
ml load NVHPC/23.7-CUDA-12.1.1
ml load Anaconda3/2023.09-0

# Activate the Conda environment
source activate /hpchome/giorgio24/Log-Anomaly-Detection-via-LLMs/conda_venv

# Verify the environment is activated
echo "Environment activated: $CONDA_PREFIX"

# Set environment variables
export PYTHONPATH=/hpchome/giorgio24/Log-Anomaly-Detection-via-LLMs

# Run the fine-tun script
# torchrun --nproc_per_node=8 scripts/fine_tune_llama.py
python scripts/fine_tune_llama.py

