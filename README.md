# Log-Anomaly-Detection-via-LLMs
This repository showcases an end-to-end workflow for anomaly detection using large language models (LLMs) such as BERT and LLAMA. 
The project was developed as part of my final coursework for the T-725-MALV Natural Language Processing course, 
conducted in the Fall semester of 2024 at Reykjavik University. The code was run and tested on Elja, a High Performance 
Computing (HPC) environment located in Iceland.

## Project Overview
The primary goal of this project is to apply NLP techniques to the field of log anomaly detection. By leveraging modern 
transformer-based models, this project focuses on detecting anomalies in system logs, which is a crucial task in fields 
like cybersecurity and systems reliability. The project demonstrates how LLMs can be fine-tuned to detect 
irregularities in log files, providing a powerful tool for monitoring and safeguarding complex infrastructures.

The dataset used is from HDFS, a well-known distributed file system, which contains both normal and anomalous log traces.
Using BERT and LLAMA, alongside a model-agnostic modular design, this repository serves as an exploration of advanced 
anomaly detection techniques that are particularly relevant in the domains of AI, ML, and cybersecurity.

## Project Structure
The repository is organized as follows:
- `dataset/`: Contains resources related to the dataset, including scripts for downloading, analyzing, and preprocessing the dataset. It also contains the raw dataset files used for training and evaluation.
- `deployment/`: Contains script for deploying the trained model for presentation.
- `models/`: Holds model-specific code. Currently, it includes subdirectories for BERT model. Each subdirectory contains code for data loading, model initialization, and training.
- `result/`: Contains the results of the experiments, including model evaluation metrics and visualizations.
- `scripts/`: Contains main scripts used to fine-tune models. These scripts serve as entry points for training models.
- `LICENSE`: Contains license information for the project.
- `requirements.txt`: Lists the dependencies required to run the code.
- `run_fine_tune_bert.sh`: A shell script to execute the fine-tuning of the BERT model in the HPC environment. It contains the necessary SBATCH configurations to run the training process efficiently on a cluster.
- `run_fine_tune_llama.sh`: A shell script to execute the fine-tuning of the LLAMA model in the HPC environment. It contains the necessary SBATCH configurations to run the training process efficiently on a cluster.

## Motivation
This project highlights my interest in the intersection of Artificial Intelligence, Cybersecurity, and Natural Language 
Processing. Given the importance of anomaly detection in ensuring the safety of modern digital infrastructure, the 
project focuses on the use of state-of-the-art language models to tackle the challenge of log-based anomaly detection. 
This work demonstrates how advanced NLP techniques, including the use of large language models (LLMs), can be adapted 
to solve practical cybersecurity challenges in large, distributed systems.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- HPC Iceland for the computational resources that made large-scale training feasible.
