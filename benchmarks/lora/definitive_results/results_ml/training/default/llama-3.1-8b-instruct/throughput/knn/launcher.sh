#!/bin/bash
#SBATCH --job-name=knn
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --output=benchmarks/lora/definitive_results/finding_maximum/digital_twin_simplification/training/llama-3.1-8b-instruct/reg/knn/log_%j.out
#SBATCH --error=benchmarks/lora/definitive_results/finding_maximum/digital_twin_simplification/training/llama-3.1-8b-instruct/reg/knn/log_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --time=24:55:00
module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate MLTrain
PYTHONPATH=/gpfs/home/bsc/bsc098069/llm_benchmarking/IPDPSPaper python3 /gpfs/home/bsc/bsc098069/llm_benchmarking/IPDPSPaper/benchmarks/lora/train_ml.py --output-path benchmarks/lora/definitive_results/finding_maximum/digital_twin_simplification/training/llama-3.1-8b-instruct/reg/knn --train-dataset-path '/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/data/dataset_llama-31-8b-instruct.csv' --test-dataset-path '/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/data/dataset_llama-31-8b-instruct_test.csv' --import-statement 'from sklearn.neighbors import KNeighborsRegressor' --class-name 'KNeighborsRegressor' --parameters-to-test '{"n_jobs": [-1], "n_neighbors": [1], "leaf_size": [8], "p": [1, 2], "weights": ["uniform"], "algorithm": ["kd_tree"]}'