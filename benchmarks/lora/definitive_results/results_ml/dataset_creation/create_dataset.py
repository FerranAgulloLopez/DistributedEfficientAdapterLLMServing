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


def extract_results_test(path: str) -> List[Dict[str, Any]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

    def extract_experiment_metric(path: str) -> Dict[str, float]:
        output: Dict[str, Any] = {}

        # load metrics
        filenames: List[str] = glob.glob(os.path.join(path, 'openai-*.json'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as file:
            metrics: dict = json.load(file)

        # load benchmark out log
        filenames: List[str] = glob.glob(os.path.join(path, 'log_*.out'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as file:
            benchmark_log: str = file.read()

        # load server out log
        filenames: List[str] = glob.glob(os.path.join(path, 'server_out.log'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as file:
            server_log: str = file.read()

        # compute throughput
        output['total_throughput'] = float(metrics['input_throughput']) + float(metrics['output_throughput'])

        # compute ideal total throughput
        received_input_tokens: int = 0
        received_output_tokens: int = 0
        with open(os.path.join(path, 'arrivals.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header
            for arrival_time, input_tokens, output_tokens, adapter_id in reader:
                received_input_tokens += int(input_tokens)
                received_output_tokens += int(output_tokens)
        output['ideal_throughput'] = (received_input_tokens + received_output_tokens) / float(metrics['duration'])

        # compute itl
        output['itl'] = float(metrics['mean_itl_ms'])

        # compute ttft
        completed_ttft = float(metrics['mean_ttft_ms']) * int(metrics['completed'])
        uncompleted_ttft = (int(metrics['total_prompts_sent']) - int(metrics['completed'])) * float(metrics['duration']) * 1000
        output['ttft'] = (completed_ttft + uncompleted_ttft) / int(metrics['total_prompts_sent'])

        # extract served adapters
        pattern = r'max_cpu_loras=(\d+)'
        found = re.findall(pattern, server_log)[-1]
        if found is None:
            raise ValueError(f'Metric pattern not found on result log')
        output['served_adapters'] = int(found)

        # extract adapter slots
        pattern = r'max_loras=(\d+)'
        found = re.findall(pattern, server_log)[-1]
        if found is None:
            raise ValueError(f'Metric pattern not found on result log')
        output['adapter_slots'] = int(found)

        # extract adapter rates
        pattern = r'Adapter rates. Values: \[(.*?)\]'
        found = re.findall(pattern, benchmark_log)[-1]
        if found is None:
            raise ValueError(f'Metric pattern not found on result log')
        found = found.replace(',', '')
        found = found.replace('  ', ' ')
        while found[-1] == ' ':
            found = found[:-1]
        values = [float(item) for item in found.split(' ')]
        pattern = r'. Counts: \[(.*?)\]'
        found = re.findall(pattern, benchmark_log)[-1]
        if found is None:
            raise ValueError(f'Metric pattern not found on result log')
        counts = [int(item) for item in found.replace(',', '').split(' ')]
        adapter_rates: List[float] = []
        for index, value in enumerate(values):
            adapter_rates += [value] * counts[index]
        output['rates'] = adapter_rates

        # extract adapter sizes
        adapter_sizes: List[int] = []
        with open(os.path.join(path, 'adapters.csv')) as file:
            reader = csv.DictReader(file)
            for row in reader:
                adapter_path: str = row['adapter_path']
                adapter_name: str = os.path.basename(adapter_path)
                if adapter_name == '':
                    adapter_name = os.path.basename(adapter_path[:-1])
                adapter_rank: int = int(adapter_name.split('rank_')[-1])
                adapter_sizes.append(adapter_rank)
        output['sizes'] = adapter_sizes

        # extract adapter rates id
        pattern = r'adapter_rates=\[(.*?)\]'
        found = re.findall(pattern, benchmark_log)[-1]
        if found is None:
            raise ValueError(f'Metric pattern not found on result log')
        output['rates_id'] = found.replace(',', '')

        # extract adapter sizes id
        pattern = r'dummy_lora_modules=\'(.*?)\','
        found = re.findall(pattern, server_log)[-1]
        if found is None:
            raise ValueError(f'Metric pattern not found on result log')
        adapter_folders = found.split(' ')
        sizes = []
        for adapter_folder in adapter_folders:
            adapter_name = os.path.basename(adapter_folder)
            if adapter_name == '':
                adapter_name = os.path.basename(adapter_folder[:-1])
            adapter_rank = adapter_name.split('rank_')[-1]
            sizes.append(int(adapter_rank))
        output['sizes_id'] = ' '.join([str(item) for item in sizes])

        return output

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['served_adapters', 'adapter_slots', 'rates_id', 'sizes_id']
    results = []
    errors: int = 0
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            if 'rank_' in folder:
                continue
            try:
                metrics = extract_experiment_metric(os.path.join(subdir, folder))
                metrics['path'] = os.path.join(subdir, folder)
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


def create_dataset(
        all_results: List[Dict[str, Any]],
        path: str,
        title: str
) -> None:
    # create dataset
    dataset: List[Dict[str, Any]] = []
    for result in all_results:
        features: Dict[str, Any] = {}

        # rate features
        rates: np.ndarray = np.asarray(result['rates'])
        features['sum_rate'] = float(np.sum(rates))
        features['std_rate'] = float(np.std(rates))

        # rank features
        sizes: np.ndarray = np.asarray(result['sizes'])
        features['max_size'] = float(np.max(sizes))
        features['mean_size'] = float(np.mean(sizes))
        features['std_size'] = float(np.std(sizes))

        # slots feature
        features['adapter_slots'] = result['adapter_slots']

        # served adapters feature
        features['served_adapters'] = result['served_adapters']

        # throughput feature
        features['total_throughput'] = result['total_throughput']

        # itl feature
        features['itl'] = result['itl']

        # ttft feature
        features['ttft'] = result['ttft']

        # starvation feature
        starvation = result['total_throughput'] < (result['ideal_throughput'] * 0.9)
        if starvation:
            features['starvation'] = 1
        else:
            features['starvation'] = 0

        # reference features
        features['path'] = os.path.dirname(result['path'])

        dataset.append(features)

    # save dataset
    with open(os.path.join(path, f'dataset_{title.replace(".", "")}.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        header: List[str] = list(dataset[0].keys())
        writer.writerow(header)
        for dataset_row in dataset:
            writer.writerow([dataset_row[key] for key in header])


def main():
    # set random seed
    random.seed(0)
    np.random.seed(0)

    for model in ['llama-3.1-8b-instruct', 'qwen-2.5-7b-instruct']:
        print(model)

        # training split (from dt runs)
        path: str = os.path.join('dt_runs', model, 'results')
        all_results: List[Dict[str, Any]] = extract_results(path)
        create_dataset(
            all_results,
            '.',
            model
        )

        # test split (from real system results)
        test_path: str = os.path.join('../../results_digital_twin/predictable/real_results', model)
        all_results: List[Dict[str, Any]] = extract_results_test(test_path)
        create_dataset(
            all_results,
            '.',
            f'{model}_test'
        )


if __name__ == '__main__':
    main()
