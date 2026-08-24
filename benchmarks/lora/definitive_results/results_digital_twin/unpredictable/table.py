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

from benchmarks.lora.simulation_pipeline import simulation_pipeline
from benchmarks.lora.predict_digital_twin import predict_digital_twin


def has_nested_keys(d, keys):
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return False
    return True


def extract_results_model(path: str, rate: str) -> List[Dict[str, float]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

    def cut_timewise(start_time: float, end_time: float, times: np.ndarray, values: np.ndarray) -> np.ndarray:
        assert np.shape(times)[0] == np.shape(values)[0]
        start_index = 0
        end_index = np.shape(times)[0]
        while times[start_index] < start_time:
            start_index += 1
        while times[end_index - 1] > end_time:
            end_index -= 1
        return values[start_index:end_index]

    def extract_experiment_metric(path: str, m: int, rank: int) -> Dict[str, float]:
        output: Dict[str, float] = {}

        # load metrics
        filenames: List[str] = glob.glob(os.path.join(path, 'openai-*.json'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as file:
            metrics: dict = json.load(file)

        # load server out log
        filenames: List[str] = glob.glob(os.path.join(path, 'server_out.log'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as file:
            log: str = file.read()

        # load launcher
        filenames: List[str] = glob.glob(os.path.join(path, 'launcher.sh'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as file:
            launcher_config: str = file.read()

        # load benchmark out log
        filenames: List[str] = glob.glob(os.path.join(path, 'log_*.out'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as file:
            benchmark_log: str = file.read()

        # extract start time
        start_time: float = float(metrics['start_time'])

        # extract end time
        end_time: float = float(metrics['end_time'])

        # compute throughput
        output['total_throughput'] = float(metrics['input_throughput']) + float(metrics['output_throughput'])

        # compute itl
        output['itl'] = float(metrics['mean_itl_ms'])

        # compute ttft
        completed_ttft = float(metrics['mean_ttft_ms']) * int(metrics['completed'])
        uncompleted_ttft = (int(metrics['total_prompts_sent']) - int(metrics['completed'])) * float(
            metrics['duration']) * 1000
        output['ttft'] = (completed_ttft + uncompleted_ttft) / int(metrics['total_prompts_sent'])

        # compute duration
        output['duration'] = float(metrics['duration'])

        # extract if benchmark preloading was used
        benchmark_preloading: bool = False
        if '--lora-pre-loading' in launcher_config:
            benchmark_preloading = True
            print('Warn!! LoRA preloading activated, some loads results may not be exactly true')

        # extract max cpu loras
        pattern = r'max_cpu_loras=(\d+)'
        found = re.findall(pattern, log)[-1]
        if found is None:
            raise ValueError(f'Metric pattern not found on result log')
        max_cpu_loras = int(found)

        # extract disk loads
        pattern = r'Total loads from disk: (\d+). Total loading time from disk: ((\d+\.(\d+)?)(e-\d+)?) seconds'
        data = re.findall(pattern, log)
        if data is None:
            raise ValueError(f'Metric pattern not found on result log')
        number_disk_loads: int = len(data)
        assert number_disk_loads == max_cpu_loras

        # extract memory loads
        pattern = r'Total loads from memory: (\d+). Total loading time from memory: ((\d+\.(\d+)?)(e-\d+)?) seconds'
        data = re.findall(pattern, log)
        if data is None:
            raise ValueError(f'Metric pattern not found on result log')
        number_memory_loads: int = len(data)

        # compute number loads from memory after init (with benchmark_preloading the results are not exactly true!!!)
        number_memory_loads_after_init: int = number_memory_loads - number_disk_loads
        output['total_loads_from_memory'] = number_memory_loads_after_init

        # extract memory loading times
        memory_loads_time: List[float] = [float(item[1]) for item in data]

        # extract total memory loading time minus init
        output['loading_time_from_memory'] = memory_loads_time[number_memory_loads - 1] - memory_loads_time[
            number_disk_loads - 1]

        # compute scheduler time
        pattern = r'Timestamp: ((\d+\.\d+)(e-\d+)?). Scheduler time: ((\d+\.\d+)(e-\d+)?) seconds. Unique adapters by batch: ((\d+\.(\d+)?)(e-\d+)?)'
        progress_data = re.findall(pattern, log)
        if progress_data is None:
            raise ValueError(f'Metric pattern not found on result log')
        progress_time: List[float] = []
        progress_scheduler: List[float] = []
        for data_point in progress_data:
            progress_time.append(float(data_point[0]))
            progress_scheduler.append(float(data_point[3]))
        progress_scheduler: np.ndarray = cut_timewise(
            start_time,
            end_time,
            np.asarray(progress_time),
            np.asarray(progress_scheduler)
        )
        output['scheduler_time'] = float(np.sum(progress_scheduler))

        # extract full adapter rates
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
        output['rates_full'] = np.asarray(adapter_rates)

        # extract full adapter sizes
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
        output['sizes_full'] = np.asarray(adapter_sizes)

        return output

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['m', 'cpu_loras', 'rank', 'rate']
    results = []
    errors: int = 0
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            m: int = int(folder.split('_')[1])
            cpu_loras: int = int(folder.split('_')[2])
            rank: int = int(folder.split('_')[3])
            try:
                metrics = extract_experiment_metric(os.path.join(path, folder), m, rank)
            except Exception as e:
                # load server err log
                filenames: List[str] = glob.glob(os.path.join(path, folder, 'server_err.log'))
                if len(filenames) != 1:
                    print(os.path.join(path, folder, 'server_err.log'))
                    raise ValueError(f'More than one output result file or none {filenames} for path {path}')
                with open(filenames[0]) as file:
                    server_err_log: str = file.read()
                if (
                        'torch.OutOfMemoryError:' in server_err_log or
                        'KV cache is needed, which is larger than the available KV cache memory' in server_err_log or
                        'No available memory for the cache blocks' in server_err_log
                ):
                    metrics = {}
                    errors += 1
                else:
                    print(os.path.join(path, folder, 'server_err.log'))
                    continue
            metrics['m'] = m
            metrics['cpu_loras'] = cpu_loras
            metrics['rank'] = rank
            metrics['rate'] = rate
            metrics['path'] = os.path.join(path, folder)
            _id = create_id(metrics, id_metrics)
            if _id in collected_ids:
                raise ValueError('Repeated results')
            collected_ids.add(_id)
            results.append(metrics)
    print(f'Extraction errors: {errors}. Should be zero.')
    return results


def __prepare_lines(results: List[Dict[str, float]], x_axis: str, y_axis: str, selection: str,
                    filter_in: Tuple[str, str] = None, add_all_info: bool = False) -> List[
    Tuple[str, List[int], List[float]]]:
    output_tmp: Dict[str, Tuple[List[int], List[float]]] = {}
    for item in results:
        selection_id = item[selection]
        if filter_in is not None and str(item[filter_in[0]]) != filter_in[1]:
            continue
        if selection_id not in output_tmp:
            output_tmp[selection_id] = ([], [])
            if add_all_info:
                output_tmp[selection_id] = ([], [], [])
        if x_axis not in item:
            output_tmp[selection_id][0].append(None)
        else:
            output_tmp[selection_id][0].append(item[x_axis])
        if y_axis not in item:
            output_tmp[selection_id][1].append(None)
        else:
            output_tmp[selection_id][1].append(item[y_axis])
        if add_all_info:
            output_tmp[selection_id][2].append(item)
    output: List[Tuple[str, List[int], List[float]]] = []
    for key, values in output_tmp.items():
        if not add_all_info:
            x_values, y_values = values
        else:
            x_values, y_values, z_values = values
        x_line = [x_value for index, x_value in enumerate(x_values) if
                  x_value is not None and y_values[index] is not None]
        y_line = [y_value for index, y_value in enumerate(y_values) if
                  y_value is not None and x_values[index] is not None]
        if add_all_info:
            z_line = [z_value for index, z_value in enumerate(z_values) if
                      x_values[index] is not None and y_values[index] is not None]
        y_line = [y_value for _, y_value in sorted(zip(x_line, y_line))]
        if add_all_info:
            z_line = [z_value for _, z_value in sorted(zip(x_line, z_line))]
        x_line.sort()
        if not add_all_info:
            output.append(
                (
                    key,
                    x_line,
                    y_line
                )
            )
        else:
            output.append(
                (
                    key,
                    x_line,
                    y_line,
                    z_line
                )
            )
    # output = [value for _, value in sorted(zip([value[0] for value in output], output))]

    return output


def compute_real_simulated_results(
        model: str,
        model_results: List[Dict[str, Any]],
        metrics: List[str]
) -> Tuple[
    Dict[int, Dict[str, Dict[str, List[float]]]],
    Dict[int, Dict[str, Dict[str, List[float]]]],
    Dict[int, Dict[str, Dict[str, List[float]]]],
    Dict[int, Dict[str, Dict[str, List[float]]]],
    Dict[int, Dict[str, Dict[str, List[float]]]],
    Dict[int, Dict[str, Dict[str, List[float]]]]
]:

    real_results: Dict[int, Dict[str, Dict[str, List[float]]]] = {}  # cpu_loras, metric, rate, value
    simulated_results: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    simulated_simple_results: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    simulated_results_preemption: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    simulated_simple_results_preemption: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    ml_results: Dict[int, Dict[str, Dict[str, List[float]]]] = {}

    # extract cpu loras
    cpu_loras_values: Set[int] = set()
    for results in model_results:
        cpu_loras_values.add(results['cpu_loras'])

    # initialize values
    for cpu_loras in cpu_loras_values:
        real_results[cpu_loras] = {}
        simulated_results[cpu_loras] = {}
        simulated_simple_results[cpu_loras] = {}
        simulated_results_preemption[cpu_loras] = {}
        simulated_simple_results_preemption[cpu_loras] = {}
        ml_results[cpu_loras] = {}
        for metric in metrics + ['m']:
            real_results[cpu_loras][metric] = {}
        for metric in metrics + ['m', 'cpu_usage', 'memory_usage']:
            simulated_results[cpu_loras][metric] = {}
            simulated_simple_results[cpu_loras][metric] = {}
            simulated_results_preemption[cpu_loras][metric] = {}
            simulated_simple_results_preemption[cpu_loras][metric] = {}
            ml_results[cpu_loras][metric] = {}

    # process results
    processed_results_by_cpu_loras = {}
    for index, cpu_loras in enumerate(cpu_loras_values):
        processed_results_by_cpu_loras[cpu_loras] = __prepare_lines(
            model_results,
            'm',
            'total_throughput',
            'rate',
            filter_in=('cpu_loras', str(cpu_loras)),
            add_all_info=True
        )

    # populate output dicts
    for index, cpu_loras in enumerate(cpu_loras_values):
        processed_results = processed_results_by_cpu_loras[cpu_loras]
        for rate, x_line, _, z_line in processed_results:
            real_results[cpu_loras]['m'][rate] = x_line

            # real results
            for metric in metrics:
                real_results[cpu_loras][metric][rate] = []
                for index in range(len(x_line)):
                    real_results[cpu_loras][metric][rate].append(z_line[index][metric])

            # simulated results
            for index in range(len(x_line)):
                experiment_path = z_line[index]['path']
                simulation_full_path = experiment_path.replace('real_results', 'simulation_results_full')
                if os.path.exists(simulation_full_path) and os.path.exists(
                        os.path.join(simulation_full_path, 'simulation_results.json')):
                    # load output
                    with open(os.path.join(simulation_full_path, 'simulation_results.json'), 'r') as file:
                        simulation_output = json.load(file)
                    # load log
                    filenames: List[str] = glob.glob(os.path.join(simulation_full_path, 'log_*.out'))
                    if len(filenames) != 1:
                        raise ValueError(f'More than one output result file or none {simulation_full_path}')
                    with open(filenames[0]) as file:
                        simulation_log: str = file.read()
                    # extract metric value
                    for metric in metrics:
                        if rate not in simulated_results[cpu_loras][metric]:
                            simulated_results[cpu_loras][metric][rate] = []
                        simulated_results[cpu_loras][metric][rate].append(simulation_output[metric])
                    # extract cpu usage
                    pattern = r'Average CPU usage: ((\d+\.\d+)(e-\d+)?)%'
                    found = re.findall(pattern, simulation_log)[-1]
                    if found is None:
                        raise ValueError(f'Metric pattern not found on result log')
                    if rate not in simulated_results[cpu_loras]['cpu_usage']:
                        simulated_results[cpu_loras]['cpu_usage'][rate] = []
                    simulated_results[cpu_loras]['cpu_usage'][rate].append(float(found[0]))
                    # extract memory usage
                    pattern = r'Average memory usage: ((\d+\.\d+)(e-\d+)?) MB'
                    found = re.findall(pattern, simulation_log)[-1]
                    if found is None:
                        raise ValueError(f'Metric pattern not found on result log')
                    if rate not in simulated_results[cpu_loras]['memory_usage']:
                        simulated_results[cpu_loras]['memory_usage'][rate] = []
                    simulated_results[cpu_loras]['memory_usage'][rate].append(float(found[0]))
                simulation_mean_path = experiment_path.replace('real_results', 'simulation_results_mean')
                if os.path.exists(simulation_mean_path) and os.path.exists(
                        os.path.join(simulation_mean_path, 'simulation_results.json')):
                    # load output
                    with open(os.path.join(simulation_mean_path, 'simulation_results.json'), 'r') as file:
                        simulation_output = json.load(file)
                    # load log
                    filenames: List[str] = glob.glob(os.path.join(simulation_mean_path, 'log_*.out'))
                    if len(filenames) != 1:
                        raise ValueError(f'More than one output result file or none {simulation_mean_path}')
                    with open(filenames[0]) as file:
                        simulation_log: str = file.read()
                    # extract metric value
                    for metric in metrics:
                        if rate not in simulated_simple_results[cpu_loras][metric]:
                            simulated_simple_results[cpu_loras][metric][rate] = []
                        simulated_simple_results[cpu_loras][metric][rate].append(simulation_output[metric])
                    # extract cpu usage
                    pattern = r'Average CPU usage: ((\d+\.\d+)(e-\d+)?)%'
                    found = re.findall(pattern, simulation_log)[-1]
                    if found is None:
                        raise ValueError(f'Metric pattern not found on result log')
                    if rate not in simulated_simple_results[cpu_loras]['cpu_usage']:
                        simulated_simple_results[cpu_loras]['cpu_usage'][rate] = []
                    simulated_simple_results[cpu_loras]['cpu_usage'][rate].append(float(found[0]))
                    # extract memory usage
                    pattern = r'Average memory usage: ((\d+\.\d+)(e-\d+)?) MB'
                    found = re.findall(pattern, simulation_log)[-1]
                    if found is None:
                        raise ValueError(f'Metric pattern not found on result log')
                    if rate not in simulated_simple_results[cpu_loras]['memory_usage']:
                        simulated_simple_results[cpu_loras]['memory_usage'][rate] = []
                    simulated_simple_results[cpu_loras]['memory_usage'][rate].append(float(found[0]))

    return real_results, simulated_results, simulated_simple_results, simulated_results_preemption, simulated_simple_results_preemption, ml_results


def create_table(
        labels_list: List[str],
        real_results_list: List[Dict[int, Dict[str, Dict[str, List[float]]]]],
        simulated_results_list: List[Dict[int, Dict[str, Dict[str, List[float]]]]],
        simulated_simple_results_list: List[Dict[int, Dict[str, Dict[str, List[float]]]]],
        simulated_results_preemption_list: List[Dict[int, Dict[str, Dict[str, List[float]]]]],
        simulated_simple_results_preemption_list: List[Dict[int, Dict[str, Dict[str, List[float]]]]],
        path: str,
        title: str,
        metrics
) -> None:
    def smape(y_true: List[float], y_pred: List[float]):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred))
        diff = np.abs(y_true - y_pred)
        return float(100 * np.mean(diff / denominator))

    def correlation(y_true: List[float], y_pred: List[float]):
        correlation_matrix = np.corrcoef(y_true, y_pred)
        return float(correlation_matrix[0, 1])

    cpu_loras_values = sorted(list(real_results_list[0].keys()))

    # merge all results together by metric
    real_results: Dict[str, List[float]] = {}
    simulated_results: Dict[str, List[float]] = {}
    simulated_simple_results: Dict[str, List[float]] = {}
    '''simulated_results_preemption: Dict[str, List[float]] = {}
    simulated_simple_results_preemption: Dict[str, List[float]] = {}'''

    for metric in list(metrics.keys()) + ['cpu_usage', 'memory_usage']:

        if metric not in {'cpu_usage', 'memory_usage'}:
            real_results[metric] = []
        simulated_results[metric] = []
        simulated_simple_results[metric] = []
        '''simulated_results_preemption[metric] = []
        simulated_simple_results_preemption[metric] = []'''

        for cpu_loras in cpu_loras_values:

            for index_x, label_results in enumerate(labels_list):
                for rate in simulated_simple_results_list[index_x][cpu_loras][metric].keys():
                    if metric not in {'cpu_usage', 'memory_usage'}:
                        y_line_real = real_results_list[index_x][cpu_loras][metric][rate]
                    y_line_simulated = simulated_results_list[index_x][cpu_loras][metric][rate]
                    y_line_simulated_simplified = simulated_simple_results_list[index_x][cpu_loras][metric][rate]
                    '''y_line_simulated_preemption = simulated_results_preemption_list[index_x][cpu_loras][metric][rate]
                    y_line_simulated_simplified_preemption = simulated_simple_results_preemption_list[index_x][cpu_loras][metric][rate]'''

                    if metric not in {'cpu_usage', 'memory_usage'}:
                        real_results[metric] += y_line_real
                    simulated_results[metric] += y_line_simulated
                    simulated_simple_results[metric] += y_line_simulated_simplified
                    '''simulated_results_preemption[metric] += y_line_simulated_preemption
                    simulated_simple_results_preemption[metric] += y_line_simulated_simplified_preemption'''

    del real_results_list
    del simulated_results_list
    del simulated_simple_results_list
    del simulated_results_preemption_list
    del simulated_simple_results_preemption_list

    # metrics table
    def print_metric_value(value: float):
        return '{:.2f}'.format(value)

    def print_metric_values(value):
        return f'Mean: {value[0]:.2f}, Margin of error: {value[1]:.2f}'

    for metric in real_results.keys():

        if len(real_results[metric]) > 0:
            for distance_label, distance_metric in [('SMAPE', smape)]:
                default: float = distance_metric(real_results[metric][:len(simulated_results[metric])], simulated_results[metric])
                # print(len(real_results[metric][:len(simulated_simple_results[metric])]), len(simulated_simple_results[metric]))
                simple: float = distance_metric(real_results[metric][:len(simulated_simple_results[metric])], simulated_simple_results[metric])
                '''default_preemption: float = smape(real_results[metric], simulated_results_preemption[metric])
                simple_preemption: float = smape(real_results[metric], simulated_simple_results_preemption[metric])'''

                print(
                    'Distance metric', distance_label,
                    'Metric', metric,
                    '. Default:', print_metric_value(default),
                    '. Simple:', print_metric_value(simple),
                    # '. Default preemption:', print_metric_value(default_preemption),
                    # '. Simple preemption:', print_metric_value(simple_preemption)
                )

    from scipy import stats
    def mean_margin_of_error(data, confidence=0.95):
        data = np.asarray(data)
        n = len(data)
        mean = np.mean(data)
        se = np.std(data, ddof=1) / np.sqrt(n)
        h = stats.t.ppf((1 + confidence) / 2, n - 1) * se  # t-distribution for small samples
        return mean, h  # mean and margin of error

    # duration table
    real_duration = mean_margin_of_error(real_results['duration'])
    simulated_duration = mean_margin_of_error(simulated_results['duration'] + simulated_simple_results['duration'])
    '''simulated_duration_preemption: float = float(np.mean(simulated_results_preemption['duration']))
    simulated_simple_duration_preemption: float = float(np.mean(simulated_simple_results_preemption['duration']))'''
    print(
        '\n\nExecution time. Real:', print_metric_values(real_duration),
        '. Default+Simple:', print_metric_values(simulated_duration),
        # '. Default preemption:', print_metric_value(simulated_duration_preemption),
        # '. Simple preemption:', print_metric_value(simulated_simple_duration_preemption)
    )

    # resources usage
    print(
        '\n\nCPU usage',
        '. Default+Simple:', print_metric_values(
            mean_margin_of_error(simulated_results['cpu_usage'] + simulated_simple_results['cpu_usage'])),
        # '. Default preemption:', print_metric_value(simulated_duration_preemption),
        # '. Simple preemption:', print_metric_value(simulated_simple_duration_preemption)
    )
    print(
        '\n\nMemory usage',
        '. Default+Simple:', print_metric_values(
            mean_margin_of_error(simulated_results['memory_usage'] + simulated_simple_results['memory_usage'])),
        # '. Default preemption:', print_metric_value(simulated_duration_preemption),
        # '. Simple preemption:', print_metric_value(simulated_simple_duration_preemption)
    )


ROOT_PATH = 'results/'
def main():
    global ROOT_PATH
    metrics = {
        'duration': 'duration (s)',
        'total_throughput': 'total throughput (toks/s)',
        'itl': 'inter-token latency (ms)',
        'ttft': 'time to first token (ms)',
        'scheduler_time': 'scheduler time (s)',
        'total_loads_from_memory': 'total loads from memory (#)',
        'loading_time_from_memory': 'loading time from memory (s)'
    }

    for model in ['llama-3.1-8b-instruct', 'qwen-2.5-7b-instruct']:
        print(model)
        labels_list = []
        real_results_list = []
        simulated_results_list = []
        simulated_simple_results_list = []
        simulated_results_preemption_list = []
        simulated_simple_results_preemption_list = []
        ml_results_list = []

        print('First processing batch')
        real_results, simulated_results, simulated_simple_results, simulated_results_preemption, simulated_simple_results_preemption, ml_results = compute_real_simulated_results(
            model,
            extract_results_model(os.path.join(ROOT_PATH, f'real_results/{model}/rank_16_rates_0.1_0.05_0.025/'),
                                  'rates 0.1 0.05 0.025') + extract_results_model(
                os.path.join(ROOT_PATH, f'real_results/{model}/rank_16_rates_1.6_0.8_0.4/'), 'rates 1.6 0.8 0.4'),
            list(metrics.keys())
        )
        labels_list.append('ranks 8, 16')
        real_results_list.append(real_results)
        simulated_results_list.append(simulated_results)
        simulated_simple_results_list.append(simulated_simple_results)
        simulated_results_preemption_list.append(simulated_results_preemption)
        simulated_simple_results_preemption_list.append(simulated_simple_results_preemption)
        ml_results_list.append(ml_results)

        print('Second and last processing batch')
        real_results, simulated_results, simulated_simple_results, simulated_results_preemption, simulated_simple_results_preemption, ml_results = compute_real_simulated_results(
            model,
            extract_results_model(os.path.join(ROOT_PATH, f'real_results/{model}/rank_32_rates_0.1_0.05_0.025/'),
                                  'rates 0.1 0.05 0.025') + extract_results_model(
                os.path.join(ROOT_PATH, f'real_results/{model}/rank_32_rates_1.6_0.8_0.4/'), 'rates 1.6 0.8 0.4'),
            list(metrics.keys())
        )
        labels_list.append('ranks 8, 16, 32')
        real_results_list.append(real_results)
        simulated_results_list.append(simulated_results)
        simulated_simple_results_list.append(simulated_simple_results)
        simulated_results_preemption_list.append(simulated_results_preemption)
        simulated_simple_results_preemption_list.append(simulated_simple_results_preemption)
        ml_results_list.append(ml_results)

        create_table(
            labels_list,
            real_results_list,
            simulated_results_list,
            simulated_simple_results_list,
            simulated_results_preemption_list,
            simulated_simple_results_preemption_list,
            ROOT_PATH,
            f'with_offloading_small_{model}',
            metrics
        )


if __name__ == '__main__':
    main()
