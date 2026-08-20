import copy
import random
from typing import List, Optional, Tuple

import numpy as np

from benchmarks.lora.benchmark_serving_by_time import assign_rates_to_adapters, get_tokenizer, sample_sharegpt_requests

MAX_GPU_TOKENS: int = 82528
MODEL: str = 'llama-2-7b'
DATASET_PATH: str = '/home/ferran/Documents/bsc/polling/data/ShareGPT_V3_unfiltered_cleaned_split.json'


def run_dynamic_simulator(
        total_time: Optional[int] = 30,  # total simulation time in seconds
        adapter_slots: Optional[int] = 64,  # GPU slots for adapters
        max_adapter_size: Optional[int] = 32,  # maximum adapter size in tokens permitted
        number_served_adapters: Optional[int] = 32,  # number of adapters to serve
        adapter_ranks_to_use: Optional[List[int]] = [8, 16, 32],  # adapter ranks to distribute upon served adapters
        adapter_rates_to_use: Optional[List[float]] = [0.2, 0.1, 0.05],  # adapter rates to distribute upon served adapters
        include_computation_overhead: Optional[bool] = False,  # include computation overhead in engine estimation
        include_loading_overhead: Optional[bool] = True,  # include adapter loading overhead in estimation
        random_seed: Optional[int] = 0,
):
    # define random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)

    # create adapters ids
    served_adapters: List[int] = list(range(number_served_adapters))

    # distribute rates between adapters
    served_adapters, adapters_rates = assign_rates_to_adapters(served_adapters, adapter_rates_to_use)

    # determine total prompts
    adapters_prompts_size: List[int] = []
    for rate in adapters_rates:
        adapters_prompts_size.append(max(1, round(total_time * rate)) * 3)  # *3 in case we need more due to random arrival intervals
    print("Adapter prompts.", adapters_prompts_size)

    # retrieve prompts
    total_prompts_size: int = sum(adapters_prompts_size)
    tokenizer = get_tokenizer(MODEL)
    total_prompts: List[Tuple[str, int, int]] = sample_sharegpt_requests(
        dataset_path=DATASET_PATH,
        num_requests=total_prompts_size,
        tokenizer=tokenizer,
        fixed_output_len=False,
    )
    # if required duplicate results until obtaining desired number of requests
    initial_length: int = len(total_prompts)
    while len(total_prompts) < total_prompts_size:
        index_to_duplicate: int = random.randint(0, initial_length - 1)  # always extract from initial set
        total_prompts.append(copy.deepcopy(total_prompts[index_to_duplicate]))
    random.shuffle(total_prompts)
    total_inputs_size: int = sum([input_tokens for _, input_tokens, _ in total_prompts])
    total_outputs_size: int = sum([output_tokens for _, _, output_tokens in total_prompts])
    print("Prompts retrieved:", len(total_prompts), ". Total input tokens:", total_inputs_size, ". Total output tokens:", total_outputs_size)

    # distribute prompts over adapters
    adapters_prompts: List[List[Tuple[str, int, int]]] = []
    global_index = 0
    for adapter_prompt_size in adapters_prompts_size:
        index = global_index
        grouped_prompts = []
        while index < len(total_prompts) and index < (global_index + adapter_prompt_size):
            grouped_prompts.append(total_prompts[index])
            index += 1
        global_index = index
        adapters_prompts.append(grouped_prompts)
    assert total_prompts_size == sum([len(adapter_prompts) for adapter_prompts in adapters_prompts])
    assert all([adapters_prompts_size[index] == len(adapters_prompts[index]) for index in range(len(adapters))])
    print("Prompts distributed")


if __name__ == '__main__':
    run_dynamic_simulator()
