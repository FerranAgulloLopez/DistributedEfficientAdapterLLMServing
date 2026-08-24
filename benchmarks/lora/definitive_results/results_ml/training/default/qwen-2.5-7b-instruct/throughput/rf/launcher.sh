#!/bin/bash
#SBATCH --job-name=rf
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --output=benchmarks/lora/definitive_results/finding_maximum/digital_twin_simplification/training/qwen-2.5-7b-instruct/reg/rf/log_%j.out
#SBATCH --error=benchmarks/lora/definitive_results/finding_maximum/digital_twin_simplification/training/qwen-2.5-7b-instruct/reg/rf/log_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --time=1:55:00
module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate MLTrain
PYTHONPATH=/gpfs/home/bsc/bsc098069/llm_benchmarking/IPDPSPaper python3 /gpfs/home/bsc/bsc098069/llm_benchmarking/IPDPSPaper/benchmarks/lora/train_ml.py --output-path benchmarks/lora/definitive_results/finding_maximum/digital_twin_simplification/training/qwen-2.5-7b-instruct/reg/rf --train-dataset-path '/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/data/dataset_qwen-25-7b-instruct.csv' --test-dataset-path '/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/data/dataset_qwen-25-7b-instruct_test.csv' --import-statement 'from sklearn.ensemble import RandomForestRegressor' --class-name 'RandomForestRegressor' --parameters-to-test '{"n_jobs": [-1], "n_estimators": [32, 128, 256], "max_depth": [null, 5, 10, 20], "min_samples_split": [2, 5, 10, 20], "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"], "min_samples_leaf": [1, 2, 5, 10, 32, 128], "max_features": [null, "sqrt", "log2"]}' --extract-rule-number
