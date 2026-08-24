# Data-Driven Optimization of GPU efficiency for Distributed LLM–Adapter Serving
_This repository was created through a fork of the [vLLM repository](https://github.com/vllm-project/vllm) of version 0.8.5 as a result of the journal manuscript [Data-Driven Optimization of GPU efficiency for Distributed LLM–Adapter Serving](MISSING)_

### Introduction
This repository includes the code modifications made to the original vLLM repository to support the experiments presented in the mentioned manuscript. It also contains all the results and code used to generate the manuscript's tables and figures.

The sections below detail:
- The specific code changes
- The required steps for setting up the environment
- The required steps to run the experiments
- The results of the manuscript and how to create the tables and charts

For Figure A.13 of the manuscript that was done over S-LoRA, please refer to the [counterpart repository](https://github.com/FerranAgulloLopez/GPULLMAdapterOptimizationSLoRA).

### Code changes

This repository includes modifications to both the benchmarking and server components of the vLLM serving system. The main changes are described below.

* **Benchmark component**

  * **`benchmarks/benchmark_serving.py`**: An updated version of the original online benchmark script with the following additions:

    * **Adapter management:** Collects the adapters deployed on the server and sends requests to them.
    * **Metrics logging:** Collects metrics from the vLLM Prometheus endpoint. This functionality is implemented in `benchmarks/concurrent_metrics_checker.py` and can be disabled using the `--disable-log-stats` flag.
    * **Integrated server launching:** Automatically launches the server using the `--launch-server` flag. Server arguments can be specified through `--server-args`, allowing a benchmark to be executed from a single entry point.
  * **`benchmarks/benchmark_serving_by_time.py`**: An extension of the previous benchmark script with the following changes:

    * **Fixed time window:** Reproduces a client-server workload over a fixed time window rather than sending a fixed number of requests. This avoids the unrealistic workload pattern that can occur at the end of a benchmark when all remaining requests are processed.
    * **Extended adapter management:** Provides greater control over the rates and sizes of the adapters being deployed and served.
    * **Unpredictable arrivals:** The analogous `benchmarks/benchmark_serving_by_time_unpredictable.py` script generates unpredictable request arrivals to evaluate the system under more challenging workload patterns.
  * **`benchmarks/benchmark_serving_by_time_multiple_servers_with_placement_with_traces.py`**: Extends the time-based benchmark to support multiple server instances and production traces:

    * **Multiple instances:** Runs multiple vLLM instances on the desired GPUs within a single node.
    * **Real traces:** Generates request arrivals from production traces. Azure trace data is currently supported, and additional traces can be incorporated in `benchmarks/traces.py`.
    * **Adapter placement:** Distributes adapters and their requests across the available instances according to the output of a placement algorithm. Implemented algorithms are available in `benchmarks/placement_algorithm/`, and additional algorithms can be implemented by following the same interface.
    * **Warmup:** The corresponding `benchmarks/benchmark_serving_by_time_multiple_servers_with_placement_with_traces_warmup.py` script executes the placement algorithm without running the actual benchmark. This is useful for debugging placement algorithms and extracting timing metrics.

* **Server component**

  * **HPC/Slurm deployment:** Includes the required scripts and configuration to deploy and run the server and benchmarks with Slurm in HPC environments using Singularity images. These resources are available under `benchmarks/deployment/slurm/`.
  * **Batch execution:** The `benchmarks/deployment/slurm/launcher_XXX` scripts provide launchers for running multiple experiments with different configurations through Slurm. A dedicated launcher is provided for each execution type, including benchmarks, Digital Twin simulations, and ML training.
  * **Additional logging:** The server has been extended with additional logging capabilities, including per-step scheduler time and adapter loading time.
  * **Simplified usage:** New server parameters have been added to facilitate experimentation. For example, `--dummy-lora-modules` can be used to serve a specified set of adapter replicas.

* **Digital Twin**

  * The proposed Digital Twin is implemented under `digital_twin_dynamic/`.
  * The predictive performance models are available under `digital_twin_dynamic/estimators/`.
  * An existing benchmark can be reproduced through `benchmarks/simulation_pipeline_from_scratch.py`, or a new scenario can be simulated using the same script. The corresponding `*_multiple` variant allows multiple simulations to be executed concurrently.

* **ML**

  * The ML models proposed in the manuscript are trained using `benchmarks/train_ml.py`.
  * The refinement phase is implemented in `benchmarks/train_ml_explainabily.py`.
  * Simplified Random Forest implementations are available in `benchmarks/fast_random_forest.py` and `benchmarks/fast_random_forest_numba.py`, including the corresponding Numba-based implementation.

We also made minor updates to the `.gitignore` file, adding rules to prevent newly generated output files from being tracked while removing specific rules to allow selected result files to be included in the repository.

### How to set up
We show how to reproduce the experiments that appear in the paper, where we use Singularity and Slurm, which must be installed prior to execution. Nevertheless, as the original vLLM code, everything can also be run with docker or plainly with Python. Follow the required steps:
- Create the base Docker image as the foundation for the Singularity image: `VLLM_INSTALL_PUNICA_KERNELS=1 docker build -f docker/Dockerfile --target vllm-openai -t vllm .`
- If having problems with the wheel size, you can reduce the amount of supported through build arg `torch_cuda_arch_list` or disable the wheel check through build arg `RUN_WHEEL_CHECK` (or just comment it in the Dockerfile directly)
- Create the Singularity image. If willing to use both online and offline modes, it is recommended to use the definition file found in the _benchmarks_simple_ deployment directory as appears in the following commands: `sudo singularity build vllm.sif benchmarks_simple/deployment/SingularityBenchmark.def`
- If having problems with memory errors, you can include swap space like following:
```
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### How to run
Once the Singularity image has been built, you can easily reproduce the experiments from the paper using the commands provided in the .txt configuration files available under _benchmarks/lora/definitive_results/_. For every run set of experiments we include a config*.txt file with the command that was used ot run it (it is created automatically). These commands invoke the launcher that does all effort in running the desired experiments in the Slurm cluster transparent to the user.

For instance, the directory _benchmarks/lora/definitive_results/performance_analysis/memory_overhead/llama-2-7b/mean_dataset/_ contains the file _config-50824.txt_ with the ready-to-run command that was used to run some of the performance analysis experiments of Section 5.1.1. Simply execute it from the root directory of the repository, and the corresponding experiments will be submitted to the Slurm queue for execution.

In this example we have the following:
```
PYTHONPATH=. python3 benchmarks/lora/deployment/slurm/launcher.py 
--user bsc98 
--queue acc_bsccs 
--max-duration 00:35:00 
--results-path benchmarks/lora/definitive_results/theory/S/llama-2-7b/mean_dataset 
--default-server-args "{'--disable-log-requests': '', '--model': '/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-2-7b', '--enable-lora': '', '--max-num-seqs': '4096'}" 
--default-benchmark-args "{'--disable-loras': '', '--backend': 'openai', '--disable-tqdm': '', '--dataset-path': '/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/data/dummy_dataset_mean.json', '--endpoint': '/v1/completions', '--model': '/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-2-7b', '--num-prompts': '1500', '--save-result': ''}" 
--test-server-args "{'--max-loras': ['1', '128', '256', '384', '512', '640', '768', '896', '784', '800', '816', '832', '848', '864', '880'], '--max-lora-rank': ['8']}" 
--test-benchmark-args "{}"
```

While the full list of arguments can be found in the corresponding launcher script, here is a summary of the key components defined in this example:

- `PYTHONPATH`: Set to the root of the repository to define the Python working directory.
- `--user`: Specifies the Slurm user account.
- `--queue`: Specifies the Slurm queue or partition.
- `--max-duration`: Sets the maximum allowed runtime for the Slurm job.
- `--results-path`: Path where all experiment results will be saved.
- `--default-server-args`: Arguments passed to the vLLM server for all experiments.
- `--default-benchmark-args`: Common benchmark arguments used across all experiments.
- `--test-server-args`: Server arguments that will vary across different experiments.
- `--test-benchmark-args`: Benchmark arguments that will vary across different experiments.

The launcher will automatically initiate an experiment for every combination of server and benchmark arguments defined via the `--test-server-args` and `--test-benchmark-args` flags. Each of these experiments will also include the default arguments specified by the `--default-server-args` and `--default-benchmark-args` flags. Every experiment is executed through the appropriate benchmark script, which also handles launching of the vLLM server.

Take into account that these commands use input arguments for defining the locations of the datasets and models between others. They will need to be change for them to work properly.

### Manuscript results
All results of the manuscript appear in the directory _benchmarks/lora/definitive_results_ divided by figures or tables from the manuscript. In every subfolder there appear the outcome of the run experiments with Slurm, along a chart.py python script or similar that reads these experiment outcomes and produces the corresponding figure or table.

Specifically, we have the following:
- motivation: Figure 1
- performance_analysis
  - memory_overhead: Figure 4
  - compute_overhead: Figure 5
  - loading_overhead: Figure 6
  - scheduler_overhead: Figure 7
- results_digital_twin
  - predictable: Table 1 (left part), Table 2 and Figure 8
  - unpredictable: Table 1 (right part) and Figure 9
- results_ml
  - dataset_creation: Table 3
  - training
    - default: Table 4
    - fast: Table 5
- results_caching
  - single_gpu: Figure 10
  - multi_gpu: Figure 11 and Table 6
  - lat_oriented: Figure 12

Some of the results are compressed with ZIP and uploaded with GIT LFS, they need to be uncompressed before running the python scripts that create the tables and/or figures

### How to cite
If using these code modifications please cite this paper:
```
MISSING
```
