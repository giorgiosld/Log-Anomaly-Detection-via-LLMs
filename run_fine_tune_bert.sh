#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=giorgio24@ru.is
#SBATCH --partition=48cpu_192mem  # request node from a specific partition
#SBATCH --nodes=2                 # number of nodes
#SBATCH --ntasks-per-node=48      # 48 cores per node (96 in total)
#SBATCH --mem-per-cpu=3900        # MB RAM per cpu core
#SBATCH --time=0-04:00:00         # run for 4 hours maximum (DD-HH:MM:SS)
#SBATCH --hint=nomultithread      # Suppress multithread
#SBATCH --output=bert_fine_tune_output.log
#SBATCH --error=bert_fine_tune_errors.log

# Activate the virtual environment
source $(pwd)/venv/bin/activate

# Add the current directory to PYTHONPATH
export PYTHONPATH=$(pwd)

# Run the fine-tune script
python scripts/fine_tune_bert.py
