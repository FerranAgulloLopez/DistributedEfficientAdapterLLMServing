import os
import re
import json
import glob
from typing import List, Tuple, Dict, Set, Any
import matplotlib.pyplot as plt
import csv
from matplotlib.lines import Line2D
import numpy as np
from matplotlib.legend_handler import HandlerTuple


def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
    _id: str = ''
    for metric_key in id_metrics:
        _id += f'{metrics[metric_key]}_'
    return _id


def extract_results(path: str, methods: Set[str]) -> List[Dict[str, Any]]:
    def extract_experiment_metric(path: str) -> Dict[str, Any]:
        output: Dict[str, float] = {}

        # load benchmark out log
        filenames: List[str] = glob.glob(os.path.join(path, 'log_*.out'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as file:
            benchmark_out_log: str = file.read()

        # load benchmark err log
        filenames: List[str] = glob.glob(os.path.join(path, 'log_*.err'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as file:
            benchmark_err_log: str = file.read()

        # extract placement time
        pattern = re.search(r'Elapsed time during placement estimation: ([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)\n', benchmark_out_log)
        if pattern:
            output['placement_time'] = float(pattern.group(1))

        # extract number of used servers
        if 'There were unused servers. The new list of servers is as follows:' in benchmark_out_log:
            pattern = re.search(r'The new list of servers is as follows:\s*(\[[^\]]*\])', benchmark_out_log)
            array_str: str = pattern.group(1)
            used_servers: int = len(eval(array_str))
        else:
            try:
                pattern = re.search(r'Defined servers \s*(\[[^\]]*\])', benchmark_out_log)
                array_str: str = pattern.group(1)
                used_servers: int = len(eval(array_str))
            except Exception as e:
                if placement_algorithm != 'dlora-proactive-mechanism':
                    raise e
                else:
                    print('Warning for dLoRA not showing result', path)
                    output['placement_not_possible'] = True
                    output['memory_error'] = False
                    output['total_throughput'] = None
                    output['ideal_total_throughput'] = None
                    output['itl'] = None
                    output['ttft'] = None
                    return output
        output['used_servers'] = used_servers

        # check if placement was possible
        if 'Not enough servers for input workload' in benchmark_err_log or 'Not enough servers for input rate' in benchmark_err_log:
            output['placement_not_possible'] = True
            output['memory_error'] = False
            output['total_throughput'] = None
            output['ideal_total_throughput'] = None
            output['itl'] = None
            output['ttft'] = None
            return output
        else:
            output['placement_not_possible'] = False

        # load server out log
        filenames: List[str] = glob.glob(os.path.join(path, 'server_out_*.log'))
        if len(filenames) != used_servers:
            raise ValueError(f'Not correct number of files {filenames} for path {path}')
        server_out_logs: List[str] = []
        for filename in filenames:
            with open(filename) as file:
                server_out_logs.append(file.read())

        # load server err log
        filenames: List[str] = glob.glob(os.path.join(path, 'server_err_*.log'))
        if len(filenames) != used_servers:
            raise ValueError(f'Not correct number of files {filenames} for path {path}')
        server_err_logs: List[str] = []
        for filename in filenames:
            with open(filename) as file:
                server_err_logs.append(file.read())

        # load metrics
        filenames: List[str] = glob.glob(os.path.join(path, 'openai-*.json'))
        if len(filenames) != 1:
            if (
                    any('torch.OutOfMemoryError:' in server_err_log for server_err_log in server_err_logs) or
                    any('KV cache is needed, which is larger than the available KV cache memory' in server_out_log for server_out_log in server_out_logs) or
                    any('No available memory for the cache blocks' in server_out_log for server_out_log in server_out_logs)
            ):
                output['memory_error'] = True
                output['total_throughput'] = None
                output['ideal_total_throughput'] = None
                output['itl'] = None
                output['ttft'] = None
                return output
            else:
                raise ValueError(f'Unknown error: {path}')
        with open(filenames[0]) as file:
            output['memory_error'] = False
            metrics: dict = json.load(file)

        # extract total time
        total_time: float = float(metrics['duration'])

        # extract total throughput
        output['total_throughput']: float = float(metrics['input_throughput']) + float(metrics['output_throughput'])

        # compute ideal total throughput
        received_input_tokens: int = 0
        received_output_tokens: int = 0
        with open(os.path.join(path, 'arrivals.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header
            for arrival_time, input_tokens, output_tokens, _, _ in reader:
                received_input_tokens += int(input_tokens)
                received_output_tokens += int(output_tokens)
        output['ideal_total_throughput'] = (received_input_tokens + received_output_tokens) / total_time

        # check starvation
        output['starvation'] = (1 - (output['total_throughput'] / output['ideal_total_throughput'])) > 0.1

        # compute itl
        output['itl'] = float(metrics['mean_itl_ms'])

        # compute ttft
        completed_ttft = float(metrics['mean_ttft_ms']) * int(metrics['completed'])
        uncompleted_ttft = (int(metrics['total_prompts_sent']) - int(metrics['completed'])) * float(metrics['duration']) * 1000
        output['ttft'] = (completed_ttft + uncompleted_ttft) / int(metrics['total_prompts_sent'])

        # compute e2e latency
        output['e2e_latency'] = (float(metrics['mean_ttft_ms']) + float(metrics['mean_tpot_ms']) * (int(metrics['total_output_tokens']) / int(metrics['completed']))) / 1000

        return output

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['concurrent_adapters', 'trace_arrival_multiplier', 'placement_algorithm']
    results = []
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            splits = folder.split('_')
            concurrent_adapters: int = int(splits[2])
            trace_arrival_multiplier: int = int(splits[3])
            placement_algorithm: str = splits[4]
            if placement_algorithm not in methods:
                continue
            try:
                metrics = extract_experiment_metric(os.path.join(path, folder))
            except Exception as e:
                print(e)
                continue
            metrics['concurrent_adapters'] = concurrent_adapters
            metrics['trace_arrival_multiplier'] = trace_arrival_multiplier
            metrics['placement_algorithm'] = placement_algorithm
            metrics['path'] = os.path.join(path, folder)
            _id = create_id(metrics, id_metrics)
            if _id in collected_ids:
                raise ValueError('Repeated results')
            collected_ids.add(_id)
            results.append(metrics)
    return results


def extract_results_simple(path: str, methods: Set[str]) -> List[Dict[str, Any]]:
    def extract_experiment_metric(path: str) -> Dict[str, Any]:
        output: Dict[str, float] = {}

        # load benchmark out log
        filenames: List[str] = glob.glob(os.path.join(path, 'log_*.out'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as file:
            benchmark_out_log: str = file.read()

        # load benchmark err log
        filenames: List[str] = glob.glob(os.path.join(path, 'log_*.err'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as file:
            benchmark_err_log: str = file.read()

        # extract placement time
        pattern = re.search(r'Elapsed time during placement estimation: ([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)\n', benchmark_out_log)
        if pattern:
            output['placement_time'] = float(pattern.group(1))

        return output

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['concurrent_adapters', 'trace_arrival_multiplier', 'placement_algorithm']
    results = []
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            splits = folder.split('_')
            concurrent_adapters: int = int(splits[2])
            trace_arrival_multiplier: int = int(splits[3])
            placement_algorithm: str = splits[4]
            if placement_algorithm not in methods:
                continue
            try:
                metrics = extract_experiment_metric(os.path.join(path, folder))
            except Exception as e:
                print(e)
                continue
            metrics['concurrent_adapters'] = concurrent_adapters
            metrics['trace_arrival_multiplier'] = trace_arrival_multiplier
            metrics['placement_algorithm'] = placement_algorithm
            metrics['path'] = os.path.join(path, folder)
            _id = create_id(metrics, id_metrics)
            if _id in collected_ids:
                raise ValueError('Repeated results')
            collected_ids.add(_id)
            results.append(metrics)
    return results


def main():
    for model in ['llama-3.1-8b-instruct', 'qwen-2.5-7b-instruct']:
        print('--------------------------------------------------------', model)
        # extract times
        results: List[Dict[str, Any]] = []
        for sizes in ['sizes_8-16', 'sizes_8-16-32']:
            aux_results: List[Dict[str, Any]] = extract_results(f'results_{sizes}/{sizes}/{model}', {'lora-serve', 'dlora-proactive-mechanism'})
            for result in aux_results:
                result['sizes'] = sizes
            results += aux_results
            aux_results: List[Dict[str, Any]] = extract_results_simple(f'results_{sizes}/{sizes}_check_time/{model}', {'proposal-starvation-2', 'proposal-starvation-2-fast'})
            for result in aux_results:
                result['sizes'] = sizes
            results += aux_results
        del aux_results

        # order them
        time_by_method_dict: Dict[str, Dict[str, float]] = {}
        id_metrics: List[str] = ['concurrent_adapters', 'trace_arrival_multiplier', 'sizes']
        for result in results:
            if 'placement_time' in result:
                placement_algorithm: str = result['placement_algorithm']
                if placement_algorithm not in time_by_method_dict:
                    time_by_method_dict[placement_algorithm] = {}
                _id = create_id(result, id_metrics)
                if _id in time_by_method_dict[placement_algorithm]:
                    raise ValueError('Repeated results')
                time_by_method_dict[placement_algorithm][_id] = result['placement_time']
        del results

        # delete not joint configurations
        for placement_algorithm_1 in time_by_method_dict.keys():
            ids_to_remove: List[str] = []
            for _id in time_by_method_dict[placement_algorithm_1].keys():
                found: bool = True
                for placement_algorithm_2 in time_by_method_dict.keys():
                    if _id not in time_by_method_dict[placement_algorithm_2]:
                        found = False
                if not found:
                    ids_to_remove.append(_id)
            for _id in ids_to_remove:
                del time_by_method_dict[placement_algorithm_1][_id]

        # extract average times
        times: Dict[str, float] = {}
        for placement_algorithm, method_results  in time_by_method_dict.items():
            times[placement_algorithm] = np.mean(np.asarray(list(method_results.values())))

        print(json.dumps(times, indent=4))


if __name__ == '__main__':
    main()
