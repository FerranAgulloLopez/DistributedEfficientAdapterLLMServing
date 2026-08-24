#!/bin/bash
#SBATCH --job-name=rf_simpler
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --output=benchmarks/lora/definitive_results/finding_maximum/digital_twin_simplification/training/llama-3.1-8b-instruct/explainability/reg/rf_simpler/log_%j.out
#SBATCH --error=benchmarks/lora/definitive_results/finding_maximum/digital_twin_simplification/training/llama-3.1-8b-instruct/explainability/reg/rf_simpler/log_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --time=00:25:00
module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate MLTrain
PYTHONPATH=/gpfs/home/bsc/bsc098069/llm_benchmarking/IPDPSPaper python3 /gpfs/home/bsc/bsc098069/llm_benchmarking/IPDPSPaper/benchmarks/lora/train_ml_explainability.py --output-path benchmarks/lora/definitive_results/finding_maximum/digital_twin_simplification/training/llama-3.1-8b-instruct/explainability/reg/rf_simpler --train-dataset-path '/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/data/dataset_llama-31-8b-instruct.csv' --test-dataset-path '/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/data/dataset_llama-31-8b-instruct_test.csv' --import-statement 'from sklearn.ensemble import RandomForestRegressor' --class-name 'RandomForestRegressor' --parameters-to-test '{"n_jobs": [-1], "n_estimators": [1], "max_depth": [2, 5], "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"], "min_samples_leaf": [10, 32, 128], "max_features": ["auto", "sqrt", "log2"]}'