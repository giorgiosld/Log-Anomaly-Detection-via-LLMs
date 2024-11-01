# Dataset
This project was executed on the Icelandic HPC, so the dataset itself is not stored within this repository. However, a 
script named download_dataset.sh is provided to download the dataset. Running this script will create a directory named 
HDFSv1/ where the dataset will be stored.

## Dataset Overview
The dataset comprises three main components:
- `HDFS.log`: the original log file.
- `preprocessed/`: a directory containing the preprocessed log file.
- `README.md`: a README file with information about the dataset.

For simplicity and streamlined data handling, the preprocessed data in the preprocessed/ directory is primarily used.

## Files in the `preprocessed/` Directory
The `preprocessed/` directory contains the following files for efficient analysis:
- `anomaly_label.csv`: a CSV file containing the anomaly labels for each log entry.
- `Event_traces.csv`: a CSV file containing the event traces extracted
- `Event_occurrence_matrix.csv`: a CSV file containing the event occurrence matrix
- `HDFS.log_templates.csv`: a CSV file containing the log templates
- `HDFS.npz`: a NumPy file containing the event occurrence matrix in a compressed format

## Analysis and Utilities
The repository includes the following Python scripts to work with the dataset:
- `analyze.py`: a Python script to analyze the dataset
- `occurencies.py`: a Python script for calculating event occurrences within the dataset
