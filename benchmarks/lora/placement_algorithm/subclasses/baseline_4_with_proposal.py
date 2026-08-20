from benchmarks.lora.placement_algorithm.subclasses.baseline_4 import PlacementAlgorithmBASELINE4
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from benchmarks.lora.predict_digital_twin import predict_digital_twin


ML_MODEL_PATH_STARVATION: Dict[str, str] = {
    'llama-3.1-8b-instruct': '/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/models/trained_ml/llama-3.1-8b-instruct/class/rf',
    'qwen-2.5-7b-instruct': '/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/models/trained_ml/qwen-2.5-7b-instruct/class/rf',
}
ML_MODEL_PATH_THROUGHPUT: Dict[str, str] = {
    'llama-3.1-8b-instruct': '/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/models/trained_ml/llama-3.1-8b-instruct/reg/rf',
    'qwen-2.5-7b-instruct': '/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/models/trained_ml/qwen-2.5-7b-instruct/reg/knn',
}

GPU_MEMORY_AVAILABILITY: Dict[str, Dict[int, int]] = {
    'qwen-2.5-7b-instruct': {8: 384, 16: 384, 32: 224},
    'llama-3.1-8b-instruct': {8: 384, 32: 96, 16: 224}
}

class PlacementAlgorithmBASELINE4WithProposal(PlacementAlgorithmBASELINE4):

    def __init__(self):
        return

    def define_placement(
            self,
            model: str,
            servers: List[str],
            adapters: List[str],
            adapters_ranks: List[int],
            adapters_rates: List[float],
            mean_input_length: float,
            mean_output_length: float,
            ml_model_path_throughput: Optional[str] = None,
            ml_model_path_starvation: Optional[str] = None,
    ) -> Tuple[List[int], List[str]]:
        """
            Determines the placement of adapters across available servers with a greedy minimization approach, where each adapter is assigned to the server with the lowest aggregated rate per second, along proposed ML to check if starvation.

            Parameters:
            - servers (List[str]): A list of server string IDs.
            - adapters (List[str]): A list of adapter string IDs.
            - adapters_ranks (List[int]): Rank associated with each adapter, linked by position with adapters input parameter.
            - adapters_rates (List[float]): Rate associated with each adapter, linked by position with adapters input parameter.
            - mean_input_length (float): Average length of input tokens per request.
            - mean_output_length (float): Average length of output tokens per request.

            Returns:
            - Tuple[List[int], List[str]]:
                - First list: Adapter slot associated with each server, linked by position with servers input parameter.
                - Second list: Server associated with each adapter, linked by position with adapters input parameter.
        """
        global ML_MODEL_PATH_STARVATION, ML_MODEL_PATH_THROUGHPUT, GPU_MEMORY_AVAILABILITY

        if ml_model_path_throughput is None:
            ml_model_path_throughput = ML_MODEL_PATH_THROUGHPUT
        if ml_model_path_starvation is None:
            ml_model_path_starvation = ML_MODEL_PATH_STARVATION

        servers_adapter_slots, adapters_servers = super().define_placement(
            model,
            servers,
            adapters,
            adapters_ranks,
            adapters_rates,
            mean_input_length,
            mean_output_length,
        )

        per_server_slots: Dict[str, int] = {}
        for index_server, server in enumerate(servers):
            per_server_slots[server] = servers_adapter_slots[index_server]

        per_server_workload: Dict[str, Dict[str, Any]] = {}
        for index_adapter, adapter_server in enumerate(adapters_servers):
            adapter_rate: float = adapters_rates[index_adapter]
            adapter_rank: int = adapters_ranks[index_adapter]

            if adapter_server not in per_server_workload:
                per_server_workload[adapter_server] = {
                    'rates': [],
                    'ranks': [],
                    'served_adapters': 0,
                }
            per_server_workload[adapter_server]['rates'].append(adapter_rate)
            per_server_workload[adapter_server]['ranks'].append(adapter_rank)
            per_server_workload[adapter_server]['served_adapters'] += 1

        total_throughput: float = 0
        for server, server_workload in per_server_workload.items():
            rates = np.asarray(server_workload['rates'])
            sizes = np.asarray(server_workload['ranks'])
            input_ml_data: Dict[str, Any] = {
                'sum_rate': np.sum(rates),
                'std_rate': np.std(rates),
                'max_size': np.max(sizes),
                'mean_size': np.mean(sizes),
                'std_size': np.std(sizes),
                'served_adapters': server_workload['served_adapters'],
                'adapter_slots': per_server_slots[server],
            }

            output_ml_data: Dict[str, Any] = predict_digital_twin(
                regression=True,
                model_path=ml_model_path_throughput[model],
                x_features=input_ml_data,
                y_features_to_predict=['total_throughput'],
            )
            total_throughput += float(output_ml_data['total_throughput'][0])

            adapter_slots = input_ml_data['adapter_slots']
            max_size = input_ml_data['max_size']
            assert max_size in GPU_MEMORY_AVAILABILITY[model]
            memory_error = False if adapter_slots <= GPU_MEMORY_AVAILABILITY[model][max_size] else True
            if memory_error:
                raise Exception('Not enough servers for input workload')

        ideal_throughput: float = (mean_input_length + mean_output_length) * sum(adapters_rates)
        total_throughput += 400 # add constant for small differences
        if 2*(ideal_throughput - total_throughput) / (ideal_throughput + total_throughput) > 0.1:
            raise Exception('Not enough servers for input workload')

        return servers_adapter_slots, adapters_servers
