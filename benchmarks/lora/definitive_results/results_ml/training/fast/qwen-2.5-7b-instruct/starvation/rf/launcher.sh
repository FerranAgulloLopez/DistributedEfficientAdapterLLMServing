#!/bin/bash
#SBATCH --job-name=rf_simpler
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --output=benchmarks/lora/definitive_results/finding_maximum/digital_twin_simplification/training/qwen-2.5-7b-instruct/explainability/class/rf_simpler/log_%j.out
#SBATCH --error=benchmarks/lora/definitive_results/finding_maximum/digital_twin_simplification/training/qwen-2.5-7b-instruct/explainability/class/rf_simpler/log_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --time=00:25:00
module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate MLTrain
PYTHONPATH=/gpfs/home/bsc/bsc098069/llm_benchmarking/IPDPSPaper python3 /gpfs/home/bsc/bsc098069/llm_benchmarking/IPDPSPaper/benchmarks/lora/train_ml_explainability.py --output-path benchmarks/lora/definitive_results/finding_maximum/digital_twin_simplification/training/qwen-2.5-7b-instruct/explainability/class/rf_simpler --train-dataset-path '/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/data/dataset_qwen-25-7b-instruct.csv' --test-dataset-path '/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/data/dataset_qwen-25-7b-instruct_test.csv' --import-statement 'from sklearn.ensemble import RandomForestClassifier' --class-name 'RandomForestClassifier' --parameters-to-test '{"n_jobs": [-1], "n_estimators": [1], "max_depth": [2, 4], "criterion": ["gini", "entropy", "log_loss"], "min_samples_leaf": [10, 32, 128], "max_features": [null, "sqrt", "log2"]}' --predict-classification-features