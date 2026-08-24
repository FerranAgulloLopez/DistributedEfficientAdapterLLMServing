#!/bin/bash
#SBATCH --job-name=svr
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --output=benchmarks/lora/definitive_results/finding_maximum/digital_twin_simplification/training/qwen-2.5-7b-instruct/class/svr/log_%j.out
#SBATCH --error=benchmarks/lora/definitive_results/finding_maximum/digital_twin_simplification/training/qwen-2.5-7b-instruct/class/svr/log_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --time=12:55:00
module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate MLTrain
PYTHONPATH=/gpfs/home/bsc/bsc098069/llm_benchmarking/IPDPSPaper python3 /gpfs/home/bsc/bsc098069/llm_benchmarking/IPDPSPaper/benchmarks/lora/train_ml.py --output-path benchmarks/lora/definitive_results/finding_maximum/digital_twin_simplification/training/qwen-2.5-7b-instruct/class/svr --train-dataset-path '/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/data/dataset_qwen-25-7b-instruct.csv' --test-dataset-path '/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/data/dataset_qwen-25-7b-instruct_test.csv' --import-statement 'from sklearn.svm import SVC' --class-name 'SVC' --parameters-to-test '{"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0], "kernel": ["linear", "poly", "rbf", "sigmoid"], "degree": [2, 3, 4, 5], "gamma": ["scale", "auto", 0.01, 0.1, 1, 10], "coef0": [0.0, 0.1, 0.5, 1.0]}' --predict-classification-features