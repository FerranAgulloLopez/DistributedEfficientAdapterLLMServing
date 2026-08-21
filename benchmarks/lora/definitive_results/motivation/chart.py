import copy
import os
import re
import json
import glob
from typing import List, Tuple, Dict, Set
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import random
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import sys


REQUEST_LENGTHS = {
        'SmallRequest': (23, 27),
        'MediumRequest': (250, 231),
        'LargeRequest': (423, 358)
}


def extract_results_model(path: str) -> List[Dict[str, float]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

    def extract_experiment_metric(path: str, m: int, rank: int, rate: float) -> Dict[str, float]:
        output: Dict[str, float] = {}

        filenames: List[str] = glob.glob(os.path.join(path, 'server_out.log'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as file:
            server_out_log: str = file.read()

        filenames: List[str] = glob.glob(os.path.join(path, 'server_err.log'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as file:
            server_err_log: str = file.read()

        filenames: List[str] = glob.glob(os.path.join(path, 'openai-*_intermediate.json'))
        if len(filenames) != 1:
            if (
                    'CUDA out of memory' in server_err_log or
                    'is larger than the maximum number of tokens that can be stored in KV cache' in server_err_log or
                    'KV cache is needed, which is larger than the available KV cache memory' in server_out_log or
                    'No available memory for the cache blocks' in server_out_log
            ):
                output['memory_error'] = True
                output['total_throughput'] = None
                return output
            else:
                raise ValueError(f'Unknown error: {path}')
        with open(filenames[0]) as file:
            output['memory_error'] = False
            metrics: dict = json.load(file)

        # compute lora loading time
        pattern = f"Total LoRA loading time:( +?)([+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)) seconds"
        found = re.findall(pattern, server_out_log)[-1]
        if found is None:
            raise ValueError(f'Metric pattern not found on result log')
        output['lora_loading_time'] = float(found[2])

        # compute lora activating time
        pattern = f"Total LoRA activating time:( +?)([+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)) seconds"
        found = re.findall(pattern, server_out_log)[-1]
        if found is None:
            raise ValueError(f'Metric pattern not found on result log')
        output['lora_activating_time'] = float(found[2])

        # compute throughput
        output['throughput'] = (float(metrics['total_output_tokens'])) / (float(metrics['duration']) - output['lora_loading_time'] - output['lora_activating_time'])
        output['total_throughput'] = (float(metrics['total_input_tokens']) + float(metrics['total_output_tokens'])) / (float(metrics['duration']) - output['lora_loading_time'] - output['lora_activating_time'])

        return output

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['m', 'rank', 'rate']
    if 'varying_output_length' in path:
        id_metrics.append('output_length')
    results = []
    rerun_errors: List[str] = []
    unknown_errors: int = 0
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            m: int = int(folder.split('_')[1])
            rank: int = int(folder.split('_')[2])
            auxiliary: str = folder.split('__')[1]
            if '_' in auxiliary:
                rate: float = float(auxiliary.split('_')[0])
            else:
                rate: float = float(auxiliary)
            if 'varying_output_length' in path:
                output_length: int = int(folder.split('_')[-1])
            try:
                metrics = extract_experiment_metric(os.path.join(path, folder), m, rank, rate)
            except Exception as e:
                error_message: str = f'WARNING! Error while extracting results -> {os.path.join(path, folder)}. '
                try:
                    with open(os.path.join(path, folder, 'server_err.log')) as f:
                        error_log: str = f.read()
                        if 'ValueError: No available memory for the cache blocks' in error_log:
                            error_message += 'Not enough memory'
                        elif 'torch.cuda.OutOfMemoryError: CUDA out of memory' in error_log:
                            error_message += 'Not enough memory'
                        elif 'ValueError: The model\'s max seq len (4096) is larger than the maximum number of tokens that can be stored in KV cache' in error_log:
                            error_message += 'Not enough memory'
                        elif 'RuntimeError: CUDA error: uncorrectable ECC error encountered' in error_log:
                            error_message += 'ECC error'
                            rerun_errors.append(os.path.join(path, folder))
                        elif 'RuntimeError: CUDA error: an illegal memory access was encountered' in error_log:
                            error_message += 'Memory access error'
                            rerun_errors.append(os.path.join(path, folder))
                        elif '[Errno 98] error while attempting to bind on address' in error_log:
                            error_message += 'Port bind error'
                            rerun_errors.append(os.path.join(path, folder))
                        else:
                            error_message += 'Unknown error'
                            unknown_errors += 1
                            print(folder)
                except:
                    error_message += 'Unknown error'
                    unknown_errors += 1
                    print(folder)
                # print(error_message)
                metrics = {}
                continue
            metrics['m'] = m
            metrics['rank'] = rank
            metrics['rate'] = rate
            if 'varying_output_length' in path:
                metrics['output_length'] = output_length
                metrics['input_length'] = 250
            _id = create_id(metrics, id_metrics)
            if _id in collected_ids:
                raise ValueError('Repeated results')
            collected_ids.add(_id)
            results.append(metrics)
    print(f'Unknown extraction errors: {unknown_errors}. Should be zero.')
    print(f'Rerun errors: {len(rerun_errors)}. Should be zero. Full list: {rerun_errors}')
    return results


def extract_results_slots(path: str) -> List[Dict[str, float]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

    def extract_experiment_metric(path: str, m: int, rank: int) -> Dict[str, float]:
        output: Dict[str, float] = {}

        filenames: List[str] = glob.glob(os.path.join(path, 'server_out.log'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as file:
            server_out_log: str = file.read()

        filenames: List[str] = glob.glob(os.path.join(path, 'server_err.log'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as file:
            server_err_log: str = file.read()

        filenames: List[str] = glob.glob(os.path.join(path, 'openai-*.json'))
        if len(filenames) != 1:
            if (
                    'CUDA out of memory' in server_err_log or
                    'is larger than the maximum number of tokens that can be stored in KV cache' in server_err_log or
                    'KV cache is needed, which is larger than the available KV cache memory' in server_out_log or
                    'No available memory for the cache blocks' in server_out_log
            ):
                output['memory_error'] = True
                output['total_throughput'] = None
                return output
            else:
                raise ValueError(f'Unknown error: {path}')
        with open(filenames[0]) as file:
            output['memory_error'] = False
            metrics: dict = json.load(file)

        # compute throughput
        output['total_throughput'] = (float(metrics['input_throughput']) + float(metrics['output_throughput'])) * 1.5

        # compute rate
        output['rate'] = (float(sum(metrics['adapter_rates'])) / len(metrics['adapter_rates'])) * 1.5

        return output

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['m', 'cpu_loras', 'rank']
    results = []
    errors: int = 0
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            if len(folder.split('_')) < 5:  # without offloading
                m: int = int(folder.split('_')[1])
                rank: int = int(folder.split('_')[2])
            else:
                m: int = int(folder.split('_')[1])
                cpu_loras: int = int(folder.split('_')[2])
                rank: int = int(folder.split('_')[3])
            try:
                metrics = extract_experiment_metric(os.path.join(path, folder), m, rank)
            except Exception as e:
                print(e)
                errors += 1
                continue
            if len(folder.split('_')) < 5:  # without offloading
                metrics['cpu_loras'] = m
            else:
                metrics['cpu_loras'] = cpu_loras
            metrics['m'] = m
            metrics['rank'] = rank
            metrics['path'] = os.path.join(path, folder)
            _id = create_id(metrics, id_metrics)
            if _id in collected_ids:
                raise ValueError('Repeated results')
            collected_ids.add(_id)
            results.append(metrics)
    print(f'Errors: {errors}. Should be zero.')
    return results


def __prepare_lines(results: List[Dict[str, float]], x_axis: str, y_axis: str, selection: str, filter_in: Tuple[str, str] = None, add_all_info: bool = False) -> List[
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
        '''x_line = [x_value for index, x_value in enumerate(x_values) if
                  x_value is not None and y_values[index] is not None]
        y_line = [y_value for index, y_value in enumerate(y_values) if
                  y_value is not None and x_values[index] is not None]
        if add_all_info:
            z_line = [z_value for index, z_value in enumerate(z_values) if x_values[index] is not None and y_values[index] is not None]'''
        x_line = [x_value for index, x_value in enumerate(x_values) if
                  x_value is not None]
        y_line = [y_value for index, y_value in enumerate(y_values) if
                  x_values[index] is not None]
        if add_all_info:
            z_line = [z_value for index, z_value in enumerate(z_values) if
                      x_values[index] is not None]
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
    output = [value for _, value in sorted(zip([value[0] for value in output], output))]

    return output


def plot_all_with_maximums_small_3(
        results_size: List[Dict[str, float]],
        results_rate: List[Dict[str, float]],
        results_slots: List[Dict[str, float]],
        dataset_label: str,
        path: str
) -> None:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 17,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'lines.linewidth': 2.0,
        'mathtext.default': 'regular',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.4,
        'figure.figsize': (15, 3)  # Adjusted for balanced horizontal layout
    })

    request_input_size, request_output_size = REQUEST_LENGTHS[dataset_label]

    nrows = 1
    ncols = 3
    fig, axs = plt.subplots(nrows, ncols, sharey=True)
    fig.subplots_adjust(wspace=0.)

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    for index_y, (divisor, title, label_maximum, results) in enumerate([
        ('rank', 'Varying size (rank)', 'size (rank)', results_size),
        ('rate', 'Varying rate (reqs/s)', 'rate (reqs/s)', results_rate),
        ('m', 'Varying $A_{max}$', 'rate (reqs/s)', results_slots)
    ]):
        processed_results = __prepare_lines(
            results,
            'm' if divisor != 'm' else 'cpu_loras',
            'total_throughput',
            divisor,
            add_all_info=True
        )

        # complete results
        for value, x_line, y_line, z_line in processed_results:
            # check memory errors
            if z_line[-1]['memory_error'] is True:
                memory_error = True
                while z_line[-1]['total_throughput'] is None:
                    x_line = x_line[:-1]
                    y_line = y_line[:-1]
                    z_line = z_line[:-1]
            else:
                memory_error = False

            # real results
            line = axs[index_y].plot(
                [],
                [],
                linestyle=None,
                label=value
            )[0]

            # ideal results
            if len(x_line) > 0:
                rate = z_line[0]['rate']
            else:
                rate = 0
            x_line_ideal = np.asarray(x_line)
            y_line_ideal = x_line_ideal * rate * (request_input_size + request_output_size)
            last_index = 0
            while last_index < len(x_line_ideal) and y_line_ideal[last_index] < 2300:
                last_index += 1

            # highlight maximum throughput point
            highlight_index = -1
            while (highlight_index + 1) < len(x_line) and (1 - (y_line[highlight_index + 1] / y_line_ideal[highlight_index + 1])) < 0.1:
                highlight_index += 1

            # not starvation
            axs[index_y].plot(
                x_line[:highlight_index + 1], y_line[:highlight_index + 1],
                linestyle='solid',
                marker='o',
                color=line.get_color(),
            )

            # starvation
            if highlight_index + 1 < len(x_line):
                axs[index_y].plot(
                    x_line[highlight_index:], y_line[highlight_index:],
                    linestyle='solid',
                    marker='o',
                    markerfacecolor='white',
                    color=line.get_color(),
                )

            # maxpack
            axs[index_y].plot(
                x_line[highlight_index],
                y_line[highlight_index],
                marker='^',
                markersize=15,
                zorder=5,
                color=line.get_color(),
            )

            # mark memory errors
            if memory_error:
                x_final_point = x_line[-1] + (x_line[-1] - x_line[-2]) / 2
                y_final_point = y_line[-1] + (y_line[-1] - y_line[-2]) / 2
                axs[index_y].plot(
                    list(x_line[-1:]) + [x_final_point],
                    list(y_line[-1:]) + [y_final_point],
                    marker='x',
                    markevery=[1],
                    linestyle='solid',
                    markersize=10,
                    markeredgewidth=3,
                    color=line.get_color(),
                )

        if index_y == 0:
            axs[index_y].set_ylabel('throughput (toks/s)')

        axs[index_y].set_xlabel('adapters (#)')
        if divisor != 'm':
            axs[index_y].set_xticks([0, 100, 200, 300])
        else:
            axs[index_y].set_xticks([0, 220, 440, 660], [0, 100, 200, 300])
        axs[index_y].set_title(title)

        leg = axs[index_y].legend(loc='upper right')  #, handlelength=0.5, handletextpad=0.5)
        '''for item, label in zip(leg.legend_handles, leg.texts):
            if label._text in [label_legend]:
                width = item.get_window_extent(fig.canvas.get_renderer()).width
                label.set_ha('left')
                label.set_position((-2 * width, 0))'''

    legend_elements = []
    legend_elements.append(Line2D([], [], color='gray', marker='o', linestyle='solid',markersize=10, label='Not starvation'))
    legend_elements.append(Line2D([], [], color='gray', marker='o', linestyle='solid',markersize=10, markerfacecolor='white', label='Starvation'))
    legend_elements.append(Line2D([], [], color='gray', marker='^', linestyle='None', markersize=15, label='$Max_{pack}$'))
    legend_elements.append(Line2D([], [], color='gray', marker='x', linestyle='None', markersize=10, markeredgewidth=3, label='Memory error'))
    '''legend_elements.append(
        Line2D([0], [0], marker='x', color='gray', markerfacecolor='gray', markersize=10, linestyle='None',
               markeredgewidth=5, label='Memory error'))'''
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        ncol=4,
        bbox_to_anchor=(0.5, 1.16),
    )

    plt.savefig(os.path.join(path, f'all_maximum_variations_small_with_slots_3.pdf'), format='pdf', bbox_inches='tight', dpi=400)


def main():
    model = 'llama-2-7b'

    initial_results: List[Dict[str, float]] = extract_results_model(f'{model}/initial_point')
    for index in range(len(initial_results)):
        initial_results[index]['output_length'] = REQUEST_LENGTHS['MediumRequest'][1]

    results_rate: List[Dict[str, float]] = extract_results_model(f'{model}/varying_arrival_rate')
    results_rate += initial_results

    results_size: List[Dict[str, float]] = extract_results_model(f'{model}/varying_adapter_size')
    results_size += initial_results

    results_slots: List[Dict[str, float]] = extract_results_slots(f'{model}/varying_slots')
    results_slots_aux = []
    for results in results_slots:
        if results['m'] in {12, 96, 120, 256, 320}:
            results_slots_aux.append(results)
    results_slots = results_slots_aux

    plot_all_with_maximums_small_3(
        results_size,
        results_rate,
        results_slots,
        'MediumRequest',
        '.',
    )


if __name__ == '__main__':
    main()
