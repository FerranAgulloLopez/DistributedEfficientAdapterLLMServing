import math
import random

from benchmarks.lora.placement_algorithm.interface_dynamic_routing import PlacementAlgorithmInterfaceDynamicRouting
from typing import List, Tuple, Deque, Dict, Any, Optional, Set
from enum import Enum
from collections import deque
import numpy as np
from benchmarks.lora.deployment.slurm.launcher_simulator_from_scratch import get_gpu_memory_availability
from benchmarks.lora.placement_algorithm.subclasses.proposal_starvation_3 import GPU_MEMORY_AVAILABILITY

"""
Corresponds to paper dLoRAProactive algorithm
"""

GPU_MEMORY_AVAILABILITY_EXPS: Dict[str, Dict[int, Dict[int, float]]] = {  # from check_available_gpu_memory exps
    'llama-3.1-8b-instruct': {
        8: {8: 305104, 16: 302000, 32: 296352},
        16: {8: 301520, 16: 296000, 32: 284944},
        32: {8: 295232, 16: 284176, 32: 258176},
        64: {8: 282640, 16: 256640, 32: 215104},
        96: {8: 268384, 16: 235584, 32: 168064},
        128: {8: 253568, 16: 212032},
        160: {8: 243536, 16: 187840},
        192: {8: 230976, 16: 163456},
        224: {8: 217488, 16: 138272},
        256: {8: 205888},
        288: {8: 191936},
        320: {8: 180160},
        352: {8: 166016},
        384: {8: 154240}
    },
    'qwen-2.5-7b-instruct': {
        8: {8: 700816, 16: 694848, 32: 683472},
        16: {8: 694848, 16: 683472, 32: 657616},
        32: {8: 683472, 16: 657616, 32: 601664},
        64: {8: 657616, 16: 601664, 32: 526992},
        96: {8: 632384, 16: 565744, 32: 430432},
        128: {8: 601664, 16: 526992, 32: 346768},
        160: {8: 587328, 16: 477168, 32: 250432},
        192: {8: 565744, 16: 430432, 32: 165664},
        224: {8: 541024, 16: 383632, 32: 67136},
        256: {8: 526992, 16: 346768},
        288: {8: 501680, 16: 296512},
        320: {8: 477168, 16: 250432},
        352: {8: 455376, 16: 202528},
        384: {8: 430432, 16: 165664}}
}

KV_CACHE_SIZE: Dict[str, int] = {  # from extract_max_achievable_throughput exps
    'llama-3.1-8b-instruct': 310160,
    'qwen-2.5-7b-instruct': 711200,
}

def create_lora_size_dict() -> Dict[str, Dict[int, int]]:  # for not doing these operations doing dLoRA placement time
    global GPU_MEMORY_AVAILABILITY_EXPS, KV_CACHE_SIZE
    output: Dict[str, Dict[int, int]] = {}

    # get obtained lora sizes by model and rank
    tmp_output: Dict[str, Dict[int, List[float]]] = {}
    for model, adapter_slots_exps in GPU_MEMORY_AVAILABILITY_EXPS.items():
        tmp_output[model] = {}
        for adapter_slots, rank_exps in adapter_slots_exps.items():
            for rank, left_memory_size in rank_exps.items():
                if rank not in tmp_output[model]:
                    tmp_output[model][rank] = []
                tmp_output[model][rank].append(
                    (KV_CACHE_SIZE[model] - left_memory_size) / adapter_slots
                )

    # get max (prioritizing avoiding starvation)
    for model, rank_results in tmp_output.items():
        output[model] = {}
        for rank, found_sizes in rank_results.items():
            output[model][rank] = round(max(found_sizes))

    return output

LORA_SIZE: Dict[str, Dict[int, int]] = create_lora_size_dict()


class MigrationType(Enum):
    DISPATCH_ONLY = 1
    DISPATCH_MIG = 2
    PERIOD_MIG = 3


class PlacementAlgorithmDLoRAProactiveMechanism(PlacementAlgorithmInterfaceDynamicRouting):

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
    ) -> Tuple[List[int], List[List[str]]]:
        global KV_CACHE_SIZE, LORA_SIZE
        """
            Determines the placement of adapters across available servers, it implements the dLoRA proactive mechanism.

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
                - Second list: Servers associated with each adapter, linked by position with adapters input parameter.
        """
        # initialize dLoRA class variables
        self.num_models: int = len(adapters)
        self.num_groups: int =  len(servers)  # num gpus per node
        self.engine_model_mapping: Dict[int, List] = {i: [] for i in range(len(servers))}
        self.model_engine_mapping: Dict[int, List] = {i: [] for i in range(len(adapters))}
        self.available_gpu_memorys: Dict[int, int] = {i: KV_CACHE_SIZE[model] for i in range(len(servers))}
        max_adapter_rank: int = max(adapters_ranks)
        self.lora_weight_sizes: Dict[int, int] = {i: LORA_SIZE[model][max_adapter_rank] for i in range(len(servers))}
        self.migration_type: MigrationType = MigrationType.DISPATCH_MIG  # proactive mechanism

        # define expected lora distribution
        # dLoRA updates this following array with a +1 for each arrival, we do the same with the input poisson supposing one hour of execution
        expected_lora_distribution: List[float] = [adapters_rates[i] * 3600 for i in range(len(adapters))]

        # run original dLoRA proactive mechanism code
        self.find_best_lora_weight_schedule(is_init=True, expected_lora_distribution=expected_lora_distribution)

        # transform output to our output
        servers_adapter_slots = [len(self.engine_model_mapping[i]) for i in range(len(servers))]  # adapters slots in dLoRA is set to active adapters
        adapters_servers = [[servers[j] for j in self.model_engine_mapping[i]] for i in range(len(adapters))]

        return servers_adapter_slots, adapters_servers

    def find_best_lora_weight_schedule(self, is_init: bool, expected_lora_distribution: List[float],
                                       current_lora_distribution: List[int] = None,
                                       engine_lora_capacity: List[int] = None):
        """Find the best lora weight schedule."""
        if current_lora_distribution is None:
            current_lora_distribution = [0 for _ in range(self.num_models)]
        num_lora_replicas = 0
        best_bt = 0
        update_flag = True

        engine_ids = [i for i in range(self.num_groups)]
        models_not_allocated = [i for i in range(self.num_models) if len(self.model_engine_mapping[i]) == 0]

        while update_flag:
            update_flag = False

            next_lora_type = 0
            for i in range(self.num_models):
                if current_lora_distribution[i] / (num_lora_replicas + 1e-7) - expected_lora_distribution[i] < \
                        current_lora_distribution[next_lora_type] / (num_lora_replicas + 1e-7) - \
                        expected_lora_distribution[next_lora_type]:
                    next_lora_type = i

                if next_lora_type not in models_not_allocated and len(models_not_allocated) > 0:
                    next_lora_type = models_not_allocated[0]

            # sort engine ids by number of lora weights on it
            if engine_lora_capacity is None:
                engine_ids = sorted(engine_ids, key=lambda engine_id: len(self.engine_model_mapping[engine_id]))
            else:
                engine_ids = sorted(engine_ids, key=lambda engine_id: engine_lora_capacity[engine_id] - len(
                    self.engine_model_mapping[engine_id]), reverse=True)
                engine_ids = [id for id in engine_ids if engine_lora_capacity[id] > len(self.engine_model_mapping[id])]
            for engine_id in engine_ids:
                if next_lora_type in self.engine_model_mapping[engine_id]:
                    continue
                self.engine_model_mapping[engine_id].append(next_lora_type)
                self.available_gpu_memorys[engine_id] -= self.lora_weight_sizes[engine_id]
                new_bt = self.calc_min_bt(expected_lora_distribution)
                if new_bt >= best_bt:
                    current_lora_distribution[next_lora_type] += 1
                    num_lora_replicas += 1
                    best_bt = new_bt
                    update_flag = True
                    if next_lora_type in models_not_allocated:
                        models_not_allocated.remove(next_lora_type)
                    break
                else:
                    self.engine_model_mapping[engine_id].remove(next_lora_type)
                    self.available_gpu_memorys[engine_id] += self.lora_weight_sizes[engine_id]
                    if models_not_allocated:
                        update_flag = True

        if is_init:
            min_replicas = self.num_groups + self.num_models - 1
            if self.migration_type == MigrationType.PERIOD_MIG:
                next_engine_id = 0
                next_lora_type = random.randint(0, self.num_models - 1)  # random int
                while num_lora_replicas < min_replicas:
                    while len(self.engine_model_mapping[next_engine_id]) >= self.num_models or next_lora_type in \
                            self.engine_model_mapping[next_engine_id]:
                        next_engine_id = (next_engine_id + 1) % self.num_groups
                    self.engine_model_mapping[next_engine_id].append(next_lora_type)
                    self.available_gpu_memorys[next_engine_id] -= self.lora_weight_sizes[next_engine_id]
                    next_engine_id = (next_engine_id + 1) % self.num_groups
                    current_lora_distribution[next_lora_type] += 1
                    num_lora_replicas += 1
            else:
                next_engine_id = 0
                next_lora_type = 0
                while num_lora_replicas < min_replicas:
                    while len(self.engine_model_mapping[next_engine_id]) >= self.num_models:
                        next_engine_id = (next_engine_id + 1) % self.num_groups
                    while next_lora_type in self.engine_model_mapping[next_engine_id]:
                        next_lora_type = (next_lora_type + 1) % self.num_models
                    self.engine_model_mapping[next_engine_id].append(next_lora_type)
                    self.available_gpu_memorys[next_engine_id] -= self.lora_weight_sizes[next_engine_id]
                    next_engine_id = (next_engine_id + 1) % self.num_groups
                    next_lora_type = (next_lora_type + 1) % self.num_models
                    current_lora_distribution[next_lora_type] += 1
                    num_lora_replicas += 1

        self.model_engine_mapping = {i: [] for i in range(self.num_models)}
        for engine_id, model_ids in self.engine_model_mapping.items():
            for model_id in model_ids:
                self.model_engine_mapping[model_id].append(engine_id)

    def calc_min_bt(self, expected_lora_distribution: List[float]):
        min_bt = math.inf

        for lora_type in range(self.num_models):
            total_throughput = 0
            for engine_id in range(self.num_groups):
                if lora_type in self.engine_model_mapping[engine_id]:
                    max_throughput_on_this_replica = self.available_gpu_memorys[
                        engine_id]  # since partitioned kvcache size is the same, calc max_tput is  to calc available gpu memory
                    total_throughput += max_throughput_on_this_replica
            min_bt = min(min_bt, total_throughput / expected_lora_distribution[lora_type])

        return min_bt
