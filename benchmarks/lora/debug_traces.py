import copy
import shlex
import argparse
import asyncio
import json
import os
import csv
import psutil
import random
import time
import warnings
import requests
import subprocess
from subprocess import Popen
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Tuple, Dict, Set

import numpy as np
from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from transformers import PreTrainedTokenizerBase


from benchmarks.lora.concurrent_metrics_checker import ConcurrentMetricsChecker
from benchmarks.lora.placement_algorithm.factory import \
    check_subclass, \
    check_subclass_dynamic_routing, \
    check_subclass_dynamic_routing_with_probabilities, \
    get_subclass, \
    get_subclass_dynamic_routing, \
    get_subclass_dynamic_routing_with_probabilities
from benchmarks.lora.traces import main as generate_trace


def measure(func, *args, **kwargs):
    process = psutil.Process(os.getpid())

    # CPU times (user + system)
    cpu_before = process.cpu_times()
    mem_before = process.memory_info().rss  # bytes

    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()

    cpu_after = process.cpu_times()
    mem_after = process.memory_info().rss

    cpu_used = (
        (cpu_after.user + cpu_after.system)
        - (cpu_before.user + cpu_before.system)
    )
    mem_delta_mb = (mem_after - mem_before) / (1024 ** 2)

    return {
        "result": result,
        "cpu_seconds": cpu_used,
        "wall_seconds": end - start,
        "memory_delta_mb": mem_delta_mb,
    }


def assign_ranks_to_adapters(
        adapters: List[str],
        values_to_use: List[str]
) -> Tuple[List[int], List[str]]:
    assert len(adapters) >= len(values_to_use)
    random.shuffle(values_to_use)

    # extract rank from each adapter
    values_to_use_ranks: List[int] = []
    for path in values_to_use:
        values_to_use_ranks.append(int(path))

    # distribute
    adapters_ranks: List[int] = []
    adapters_paths: List[str] = []
    index = 0
    while len(adapters_ranks) < len(adapters):
        adapters_ranks.append(values_to_use_ranks[index])
        adapters_paths.append(values_to_use[index])
        index += 1
        if index >= len(values_to_use):
            index = 0

    # shuffle
    aux_shuffled_list = list(zip(adapters_ranks, adapters_paths))
    random.shuffle(aux_shuffled_list)
    adapters_ranks, adapters_paths = zip(*aux_shuffled_list)

    return adapters_ranks, adapters_paths


def main(args: argparse.Namespace, placement_type: str):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    servers_processes = []
    open_server_processes = []
    concurrent_metrics_checkers = []
    try:
        # generate trace
        trace_arrivals = generate_trace(
            in_path=args.trace_source,
            out_path=args.result_dir,
            source=args.trace_type,
            num_adapters=args.num_adapters_total,
            num_adapters_subset=args.num_adapters,
            weights=args.trace_adapter_weights,
            arrival_multiplier=args.trace_arrival_multiplier,
            arrival_jitter_eps=args.trace_arrival_jitter_eps,
            gentd26_adapter_mode="single",
            multi_replace=False,
            cut_per_day=args.trace_cut_per_day,
            cut_per_hour_start=args.trace_cut_per_hour_start,
            cut_per_hour_end=2,
            bin_seconds=60,
            seed=args.seed,
        )

        # remove not needed columns and transform to dicts and lists (assert arrivals are sorted)
        adapters: List[str] = []
        adapters_arrivals_dict: Dict[str, List[float]] = {}
        for index, row in trace_arrivals.iterrows():
            adapter_id: str = f'adapter_{row["adapter_id"]}'
            adapter_arrival: float = float(row['t_sec'])
            if adapter_id not in adapters_arrivals_dict:
                adapters_arrivals_dict[adapter_id] = []
                adapters.append(adapter_id)
            adapters_arrivals_dict[adapter_id].append(adapter_arrival)
        random.shuffle(adapters)
        adapters_arrivals_full: List[List[float]] = []
        total_prompts_size: int = 0
        for adapter_id in adapters:
            adapter_arrivals = adapters_arrivals_dict[adapter_id]
            assert all(adapter_arrivals[i] <= adapter_arrivals[i+1] for i in range(len(adapter_arrivals) - 1))
            adapters_arrivals_full.append(adapter_arrivals)
            total_prompts_size += len(adapter_arrivals)
        del adapters_arrivals_dict

        # distribute ranks between adapters
        adapters_ranks, adapters_paths = assign_ranks_to_adapters(adapters, args.adapter_ranks)
        values, counts = np.unique(adapters_ranks, return_counts=True)
        print(f"Adapter ranks. Values: {values}. Counts: {counts}")

        # create prompts
        total_prompts: List[Tuple[str, int, int]] = [('meh', 1, 1)] * total_prompts_size

        # assign prompts to adapters
        random.shuffle(total_prompts)
        adapters_prompts_full: List[List[Tuple[str, int, int]]] = []
        for index_adapter in range(len(adapters)):
            num_adapter_prompts: int = len(adapters_arrivals_full[index_adapter])
            adapters_prompts_full.append(total_prompts[:num_adapter_prompts])
            total_prompts = total_prompts[num_adapter_prompts:]

        # divide arrivals and prompts into two (first and second hour)
        # extract mean rates and lengths for placement algorithm
        # remove first hour arrivals and prompts (only maintain second one)
        adapters_rates: List[float] = []
        mean_length_count: int = 0
        mean_input_length: float = 0
        mean_output_length: float = 0
        adapters_arrivals: List[List[float]] = []  # second hour only
        adapters_prompts: List[List[Tuple[str, int, int]]] = []  # second hour only
        for index_adapter in range(len(adapters)):
            adapter_arrivals: List[float] = adapters_arrivals_full[index_adapter]
            adapter_prompts: List[Tuple[str, int, int]] = adapters_prompts_full[index_adapter]
            hour_split_index: int = 0
            while hour_split_index < len(adapter_arrivals) and adapter_arrivals[hour_split_index] < 3600:
                hour_split_index += 1
            if hour_split_index == 0:
                adapter_rate: float = 1 / 3600
                adapters_rates.append(adapter_rate)
                adapters_arrivals.append(adapter_arrivals[hour_split_index:])
                adapters_prompts.append(adapter_prompts[hour_split_index:])
                continue
            adapters_arrivals.append(adapter_arrivals[hour_split_index:])
            adapters_prompts.append(adapter_prompts[hour_split_index:])
            adapter_arrivals_prior: List[float] = adapter_arrivals[:hour_split_index]
            adapter_prompts_prior: List[Tuple[str, int, int]] = adapter_prompts[:hour_split_index]
            adapter_rate: float = len(adapter_arrivals_prior) / 3600
            adapters_rates.append(adapter_rate)
            mean_input_length += sum([prompt_len for _, prompt_len, _ in adapter_prompts_prior])
            mean_output_length += sum([output_len for _, _, output_len in adapter_prompts_prior])
            mean_length_count += len(adapter_prompts_prior)

        del adapters_arrivals_full
        del adapters_prompts_full
        mean_input_length /= mean_length_count
        mean_output_length /= mean_length_count
        print('Found mean input/output length', mean_input_length, mean_output_length)
        values, counts = np.unique(adapters_rates, return_counts=True)
        print(f"Adapter rates. Values: {values}. Counts: {counts}")

        # transform arrivals (remove first hour)
        for index_adapter in range(len(adapters)):
            for index_arrival in range(len(adapters_arrivals[index_adapter])):
                adapters_arrivals[index_adapter][index_arrival] -= 3600
                assert adapters_arrivals[index_adapter][index_arrival] > 0

        # define servers (check GPU availability)
        servers: List[str] = [f"server_{index}" for index in range(args.num_servers)]
        print('Defined servers', servers)

        # retrieve placement from algorithm
        init_time: float = time.perf_counter()
        if placement_type == 'default':
            placement_algorithm = get_subclass(args.placement_algorithm)()
        elif placement_type == 'dynamic':
            placement_algorithm = get_subclass_dynamic_routing(args.placement_algorithm)()
        elif placement_type == 'probabilities':
            placement_algorithm = get_subclass_dynamic_routing_with_probabilities(args.placement_algorithm)()
        try:
            placement_output = measure(placement_algorithm.define_placement,
                os.path.basename(model_id),
                servers=servers,
                adapters=adapters,
                adapters_ranks=adapters_ranks,
                adapters_rates=adapters_rates,
                mean_input_length=222.24668637846656,
                mean_output_length=200.3362051386623
            )
            print('---------ORIGINAL PLACEMENT OUTPUT---------')
            print(placement_output)
            placement_output = placement_output["result"]
            if placement_type == 'default':
                servers_adapter_slots, adapters_servers = placement_output
                adapters_servers_new = []
                adapters_servers_probabilities = []
                for index_list in range(len(adapters_servers)):
                    adapters_servers_new.append([adapters_servers[index_list]])
                    adapters_servers_probabilities.append([1])
                adapters_servers = adapters_servers_new
                del adapters_servers_new
            elif placement_type == 'dynamic':
                servers_adapter_slots, adapters_servers = placement_output
                adapters_servers_probabilities = []
                for index_list in range(len(adapters_servers)):
                    n = len(adapters_servers[index_list])
                    adapters_servers_probabilities.append([1.0 / n] * n)
            elif placement_type == 'probabilities':
                servers_adapter_slots, adapters_servers, adapters_servers_probabilities = placement_output
            print('---------PLACEMENT OUTPUT---------')
            print('servers_adapter_slots')
            print(servers_adapter_slots)
            print('adapters_servers')
            print(adapters_servers)
            print('adapters_servers_probabilities')
            print(adapters_servers_probabilities)
            print('---------PLACEMENT OUTPUT---------')
            adapters_servers_aux: List[str] = []
            for aux_list in adapters_servers:
                adapters_servers_aux += aux_list
            values, counts = np.unique(adapters_servers_aux, return_counts=True)
            print(f'Output placement. Adapter slots by server: {servers_adapter_slots}. Server by adapter: -> Values: {values} Counts: {counts}')
        finally:
            print('Elapsed time during placement estimation:', time.perf_counter() - init_time)

        # check if any server is not being used, and act accordingly
        used_servers: Set[str] = set(values)
        if len(servers) > len(used_servers):
            new_servers: List[str] = []
            new_servers_adapter_slots: List[int] = []
            for index, server in enumerate(servers):
                if server in used_servers:
                    new_servers.append(server)
                    new_servers_adapter_slots.append(servers_adapter_slots[index])
            servers = new_servers
            servers_adapter_slots = new_servers_adapter_slots
            print('There were unused servers. The new list of servers is as follows:', servers)
    except Exception as e:
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the ShareGPT dataset, will be deprecated in the "
        "next release.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",
    )
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--server-args",
        type=str,
        default="",
        help="Args to send to the servers when launching",
    )
    parser.add_argument(
        "--server-init-time",
        type=int,
        default=300,
        help="Timeout for server initialization",
    )
    parser.add_argument('--disable-log-stats',
                        action='store_true',
                        help='disable logging statistics'
                        )
    parser.add_argument(
        "--num-servers",
        type=int,
        required=True,
        help="Number of servers to use.",
    )
    parser.add_argument(
        "--num-adapters-total",
        type=int,
        required=True,
        help="Number of adapters to serve to divide trace into.",
    )
    parser.add_argument(
        "--num-adapters",
        type=int,
        required=True,
        help="Number of adapters to serve.",
    )
    parser.add_argument(
        "--total-time",
        type=int,
        required=False,
        default=3600,
        help="Maximum seconds to do benchmarking.",
    )
    parser.add_argument(
        "--trace-type",
        type=str,
        required=True,
        help="Trace type to load.",
    )
    parser.add_argument(
        "--trace-source",
        type=str,
        required=True,
        help="Trace file to load.",
    )
    parser.add_argument(
        "--trace-adapter-weights",
        type=str,
        required=False,
        default="dirichlet:0.5",
        help="Adapter popularity distribution: uniform | zipf:alpha | dirichlet:conc | file:path.json.",
    )
    parser.add_argument(
        "--trace-arrival-multiplier",
        type=int,
        required=False,
        default=1,
        help="Replicate each request K times BEFORE adapter assignment to increase arrival volume.",
    )
    parser.add_argument(
        "--trace-arrival-jitter-eps",
        type=float,
        required=False,
        default=5.0,
        help="Optional +/- seconds jitter applied after multiplying to break timestamp ties (0 disables).",
    )
    parser.add_argument(
        "--trace-cut-per-day",
        type=str,
        required=True,
        help="e.g. 2024-10-18",
    )
    parser.add_argument(
        "--trace-cut-per-hour-start",
        type=str,
        required=True,
        help="e.g. 09:00:00",
    )
    parser.add_argument(
        "--adapter-ranks",
        type=str,
        required=True,
        help="Adapter rank paths to use.",
    )
    parser.add_argument(
        "--placement-algorithm",
        type=str,
        required=True,
        help="Placement algorithm to use.",
    )

    args = parser.parse_args()
    args.adapter_ranks = [item for item in args.adapter_ranks.split(' ')]
    placement_type: str = None
    try:
        placement_algorithm = check_subclass(args.placement_algorithm)
        placement_type = 'default'
    except ValueError:
        try:
            placement_algorithm = check_subclass_dynamic_routing(args.placement_algorithm)
            placement_type = 'dynamic'
        except ValueError:
            try:
                placement_algorithm = check_subclass_dynamic_routing_with_probabilities(args.placement_algorithm)
                placement_type = 'probabilities'
            except ValueError:
                raise ValueError('Proposed placement algorithm was not found.')
    main(args, placement_type=placement_type)
