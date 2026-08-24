import csv
import heapq
import pickle
import os
import re
import json
import glob
from typing import List, Tuple, Dict, Set, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import random
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
    _id: str = ''
    for metric_key in id_metrics:
        _id += f'{metrics[metric_key]}_'
    return _id


def extract_results(path: str) -> List[Dict[str, Any]]:

    def extract_experiment_metric(path: str) -> Dict[str, Any]:
        output: Dict[str, Any] = {}

        # load arguments
        with open(os.path.join(path, 'arguments.json')) as file:
            arguments: dict = json.load(file)

        # load metrics
        with open(os.path.join(path, 'simulation_results.json')) as file:
            metrics: dict = json.load(file)

        # load arrivals
        with open(os.path.join(path, 'arrivals.json')) as file:
            arrivals: dict = json.load(file)

        # extract served adapters
        output['served_adapters'] = arguments['served_adapters']

        # extract adapter slots
        output['adapter_slots'] = arguments['adapter_slots']

        # extract adapter sizes and rates
        adapter_rates: List[float] = []
        adapter_sizes: List[int] = []
        with open(os.path.join(path, 'adapters.csv')) as file:
            reader = csv.DictReader(file)
            for row in reader:
                adapter_rates.append(float(row['adapter_rate']))
                adapter_sizes.append(int(row['adapter_size']))
        output['rates'] = adapter_rates
        output['sizes'] = adapter_sizes

        # extract adapter sizes and rates ids
        output['rates_id'] = ' '.join([str(item) for item in arguments['served_adapters_rates']])
        output['sizes_id'] = ' '.join([str(item) for item in arguments['served_adapters_sizes']])

        # extract ideal throughput
        output['ideal_throughput'] = (arrivals['total_arrivals_input_tokens'] + arrivals['total_arrivals_output_tokens']) / arguments['total_time']

        # populate simulation metrics
        for metric_key, metric_value in metrics.items():
            output[metric_key] = metric_value

        return output

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['served_adapters', 'adapter_slots', 'rates_id', 'sizes_id']
    results = []
    errors: int = 0
    for parent_dir in os.listdir(path):
        parent_dir_path = os.path.join(path, parent_dir)
        if os.path.isdir(parent_dir_path):
            for folder in os.listdir(parent_dir_path):
                folder_path = os.path.join(parent_dir_path, folder)
                try:
                    metrics = extract_experiment_metric(folder_path)
                    metrics['path'] = folder_path
                    _id = create_id(metrics, id_metrics)
                    if _id in collected_ids:
                        raise ValueError('Repeated results')
                    collected_ids.add(_id)
                    results.append(metrics)
                except Exception as e:
                    print(e)
                    errors += 1
    print(f'Extraction errors: {errors}. Should be zero.')
    return results


def main():
    # set random seed
    random.seed(0)
    np.random.seed(0)

    for model in ['llama-3.1-8b-instruct', 'qwen-2.5-7b-instruct']:
        print(model)

        path: str = os.path.join('dt_runs', model, 'results')
        all_results: List[Dict[str, Any]] = extract_results(path)

        total_time: float = 0
        for result in all_results:
            total_time += result['duration']

        print(f'Total time: {total_time}')


if __name__ == '__main__':
    main()
