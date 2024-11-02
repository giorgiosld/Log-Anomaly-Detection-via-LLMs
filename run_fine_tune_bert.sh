#!/bin/bash

# Add the current directory to PYTHONPATH
export PYTHONPATH=$(pwd)

# Run the fine-tune script
python scripts/fine_tune_bert.py
