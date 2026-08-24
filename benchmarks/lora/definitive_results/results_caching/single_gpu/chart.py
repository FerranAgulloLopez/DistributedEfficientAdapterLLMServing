import os
import re
import json
import glob
from typing import List, Tuple, Dict, Set, Any
import matplotlib.pyplot as plt
import csv
from matplotlib.lines import Line2D
import numpy as np
import pickle
from matplotlib.legend_handler import HandlerTuple


def thousands_formatter(x, pos):
    if x == 0:
        return "0"
    return f"{int(x/1000)}k"


def extract_results(path: str) -> List[Dict[str, Any]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

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
                    output['starvation'] = False
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
            output['starvation'] = False
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
            for _, input_tokens, output_tokens, _, _ in reader:
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

        # extract adapter slots
        pattern = r'max_loras=(\d+)'
        found = re.findall(pattern, server_out_logs[0])[-1]
        if found is None:
            raise ValueError(f'Metric pattern not found on result log')
        output['adapter_slots'] = int(found)

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
        x_line = [x_value for index, x_value in enumerate(x_values)]
        y_line = [y_value for index, y_value in enumerate(y_values)]
        if add_all_info:
            z_line = [z_value for index, z_value in enumerate(z_values)]
        '''y_line = [y_value for _, y_value in sorted(zip(x_line, y_line))]
        if add_all_info:
            z_line = [z_value for _, z_value in sorted(zip(x_line, z_line))]
        x_line.sort()'''
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


def plot_results_together(
        results_sizes_8_16: List[Dict[str, Any]],
        results_sizes_8_16_32: List[Dict[str, Any]],
        title: str,
        path: str,
        x_metric: str,
        x_label: str,
        y_metrics: List[str],
        y_labels: List[str],
        algorithms_to_use: List[str],
) -> None:
    multiplier = 2.5
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14 * multiplier,
        'axes.titlesize': 14 * multiplier,
        'axes.labelsize': 13 * multiplier,
        'xtick.labelsize': 11 * multiplier,
        'ytick.labelsize': 11 * multiplier,
        'legend.fontsize': 12 * multiplier,
        'lines.linewidth': 2.0 * multiplier,
        'mathtext.default': 'regular',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.4 * multiplier,
        'figure.figsize': (22, 7)  # Adjusted for balanced horizontal layout
    })

    linestyles = {
        'baseline-3': ':',
        'baseline-4': ':',
        'baseline-6': ':',
        'lora-serve': ':',
        'lora-serve-half': '-.',
        'baseline-5': '-.',
        'baseline-1': '--',
        'baseline-2': ':',
        'baseline-4-with-proposal': ':',
        'proposal-starvation': '-',
        'dlora-proactive-mechanism': '-.',
        'proposal-starvation-fast': ':',
    }
    markers = {
        'baseline-3': 'P',
        'baseline-4': 's',
        'baseline-6': 's',
        'lora-serve': 's',
        'lora-serve-half': 'P',
        'baseline-5': 'P',
        'baseline-1': '^',
        'baseline-2': 's',
        'proposal-starvation': 'o',
        'proposal-starvation-fast': '^',
        'dlora-proactive-mechanism': 'P',
        'baseline-4-with-proposal': '^',
    }
    multiplier_2 = 2
    markers_width = {
        'baseline-3': 11,
        'baseline-4': 11,
        'baseline-6': 11,
        'lora-serve': 29 * multiplier_2,
        'lora-serve-half': 11,
        'baseline-5': 11,
        'baseline-1': 18,
        'baseline-2': 11,
        'proposal-starvation': 12 * multiplier_2,
        'proposal-starvation-fast': 9 * multiplier_2,
        'dlora-proactive-mechanism': 23 * multiplier_2,
        'baseline-4-with-proposal': 23,
    }
    good_labels = {
        'baseline-3': 'Random',
        'baseline-4': 'Baseline 4',
        'baseline-6': 'Baseline 6',
        'baseline-5': 'Baseline 5',
        'lora-serve': 'LoRAServe',
        'lora-serve-half': 'LoRAServe*',
        'baseline-1': 'MaxBackbone',
        'baseline-2': 'MaxBackboneLessParallel',
        'proposal-starvation': 'Proposed',
        'proposal-starvation-fast': 'ProposedFast',
        'dlora-proactive-mechanism': 'dLoRAProactive',
        'baseline-4-with-proposal': 'ProposedLat',
    }

    colors = {'proposal-starvation': '#1f77b4', 'lora-serve': '#ff7f0e', 'lora-serve-half': '#2ca02c', 'dlora-proactive-mechanism': '#d62728'}
    nrows = len(y_metrics)
    ncols = 2
    fig, axs = plt.subplots(nrows, ncols, sharex='col', sharey='row')
    fig.subplots_adjust(wspace=0.05)
    starvation_legend_element = {}
    for index_x, (x_axis_label, results) in enumerate([('Sizes (8 & 16)', results_sizes_8_16), ('Sizes (8, 16 & 32)', results_sizes_8_16_32)]):
        for index_y, y_metric in enumerate(y_metrics):
            processed_results = __prepare_lines(
                results,
                x_metric,
                y_metric,
                'placement_algorithm',
                add_all_info=True
            )
            processed_results = {value: (x_line, y_line, z_line) for value, x_line, y_line, z_line in processed_results}
            legend_elements = []
            for value in algorithms_to_use:
                x_line, y_line, z_line = processed_results[value]
                if value == 'proposal-starvation-2':
                    value = 'proposal-starvation'
                if value == 'proposal-starvation-2-fast':
                    value = 'proposal-starvation-fast'
                if value == 'baseline-3-2':
                    value = 'baseline-3'

                # order by concurrent_adapters
                x_line = np.asarray(x_line)
                y_line = np.asarray(y_line)
                z_line = np.asarray(z_line)
                k_line = [values['concurrent_adapters'] for values in z_line]
                k_line = np.asarray(k_line)
                idx = np.argsort(k_line)
                x_line = x_line[idx]
                y_line = y_line[idx]
                z_line = z_line[idx]
                k_line = k_line[idx]

                # reduce left items
                '''if not x_axis_label == 'Low rates (mixed sizes)':
                    x_line = np.delete(x_line, [1])
                    y_line = np.delete(y_line, [1])
                    z_line = np.delete(z_line, [1])
                    k_line = np.delete(k_line, [1])
                else:
                    x_line = np.delete(x_line, [1, 2])
                    y_line = np.delete(y_line, [1, 2])
                    z_line = np.delete(z_line, [1, 2])
                    k_line = np.delete(k_line, [1, 2])'''

                # remove not placement viable results
                while z_line[-1]['placement_not_possible'] is True:
                    x_line = x_line[:-1]
                    y_line = y_line[:-1]
                    z_line = z_line[:-1]
                '''x_max = min(x_max, x_line[-1])'''

                if z_line[-1]['memory_error'] is True:
                    memory_error = True
                    while z_line[-1]['total_throughput'] is None:
                        x_line = x_line[:-1]
                        y_line = y_line[:-1]
                        z_line = z_line[:-1]
                else:
                    memory_error = False

                # real results
                if value not in colors:
                    line = axs[index_y, index_x].plot(
                        x_line,
                        y_line,
                        linestyle=linestyles[value],
                    )[0]
                    colors[value] = line.get_color()
                else:
                    line = axs[index_y, index_x].plot(
                        x_line,
                        y_line,
                        linestyle=linestyles[value],
                        color=colors[value],
                    )[0]

                legend_elements.append(
                    Line2D(
                        [],
                        [],
                        color=line.get_color(),
                        marker=markers[value],
                        linestyle=linestyles[value],
                        label=good_labels[value],
                        markersize=30,
                    )
                )
                if good_labels[value] not in starvation_legend_element and 'proposal' not in value:
                    if 'llama' in title:
                        starvation_legend_element[good_labels[value]] = Line2D(
                            [],
                            [],
                            color=line.get_color(),
                            marker='x',
                            linestyle=None,
                            linewidth=0,
                            label=good_labels[value],
                            markersize=26,
                            markerfacecolor='none',
                        )
                    else:
                        starvation_legend_element[good_labels[value]] = Line2D(
                            [],
                            [],
                            color=line.get_color(),
                            linewidth=0,
                            label=good_labels[value],
                            markerfacecolor='none',
                            marker=markers[value],
                            markersize=26,
                            markeredgewidth=1,
                            linestyle='None',
                        )

                # mark memory errors
                if memory_error:
                    x_final_point = x_line[-1] + (x_line[-1] - x_line[-2]) / 1
                    y_final_point = (y_line[-1] + (y_line[-1] - y_line[-2]) / 1) if value == 'dlora-proactive-mechanism' else (y_line[-1] + (y_line[-1] - y_line[-2]) / 4)
                    axs[index_y, index_x].plot(
                        list(x_line[-1:]) + [x_final_point],
                        list(y_line[-1:]) + [y_final_point],
                        marker='x',
                        markevery=[1],
                        linestyle=linestyles[value],
                        markersize=30,
                        markeredgewidth=2,
                        color=line.get_color(),
                    )

                # mark time error
                if x_axis_label == 'low_rates-mixed_sizes' and value == 'dlora-proactive-mechanism':
                    x_final_point = x_line[-1] + (x_line[-1] - x_line[-2]) / 2
                    y_final_point = y_line[-1] + (y_line[-1] - y_line[-2]) / 2
                    axs[index_y, index_x].plot(
                        list(x_line[-1:]) + [x_final_point],
                        list(y_line[-1:]) + [y_final_point],
                        marker='x',
                        markevery=[1],
                        linestyle=linestyles[value],
                        markersize=13,
                        markeredgewidth=8,
                        color=line.get_color(),
                    )

                # mark starvation
                starvation_x_line = []
                starvation_y_line = []
                not_starvation_x_line = []
                not_starvation_y_line = []
                for index_line in range(len(x_line)):
                    if z_line[index_line]['starvation'] is True:
                        starvation_x_line.append(x_line[index_line])
                        starvation_y_line.append(y_line[index_line])
                    else:
                        not_starvation_x_line.append(x_line[index_line])
                        not_starvation_y_line.append(y_line[index_line])
                axs[index_y, index_x].plot(
                    starvation_x_line,
                    starvation_y_line,
                    marker=markers[value],
                    markerfacecolor='none',
                    color=line.get_color(),
                    markersize=markers_width[value],
                    markeredgewidth=1,
                    linestyle='None',
                )
                axs[index_y, index_x].plot(
                    not_starvation_x_line,
                    not_starvation_y_line,
                    marker=markers[value],
                    color=line.get_color(),
                    markersize=markers_width[value],
                    markeredgewidth=1,
                    linestyle='None',
                )

            if index_x == 0:
                from matplotlib.ticker import FuncFormatter
                axs[index_y, index_x].set_ylabel(y_labels[index_y])
            if index_y == (nrows - 1):
                axs[index_y, index_x].set_xlabel(x_label)
                # axs[index_y, index_x].set_ylim(0, 220)
                # axs[index_y, index_x].set_yticks([0, 100, 200])
                if 'llama' in title:
                    axs[index_y, index_x].set_ylim(-30, 290)
                else:
                    axs[index_y, index_x].set_ylim(-20, 250)
            if index_y == 0:
                from matplotlib.ticker import FuncFormatter

                def k_formatter(x, pos):
                    if x >= 1000:
                        if x % 1000 == 0:
                            return f'{int(x / 1000)}K'
                        else:
                            return f'{x/1000:.1f}K'
                    return str(int(x))

                axs[index_y, index_x].yaxis.set_major_formatter(FuncFormatter(k_formatter))
                # axs[index_y, index_x].set_ylim(500, 9500)
                if 'llama' in title:
                    axs[index_y, index_x].set_ylim(-60, 540)
                else:
                    axs[index_y, index_x].set_ylim(250, 10000)
                axs[index_y, index_x].set_title(x_axis_label)
        '''if 'through' in y_labels[index_x]:
            axs[index_x].yaxis.set_major_formatter(FuncFormatter(thousands_formatter))'''

    from matplotlib.legend_handler import HandlerTuple
    legend_elements = list(reversed(legend_elements))
    labels = [item.get_label() for item in legend_elements]
    legend_elements.append(tuple(starvation_legend_element.values()))
    if 'llama' in title:
        labels.append('Memory error')
    else:
        labels.append('Starvation')
    fig.legend(
        handles=legend_elements,
        labels=labels,
        loc='upper center',
        ncol=5,
        bbox_to_anchor=(0.5, 1.11),
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )

    plt.savefig(os.path.join(path, f'single{title}.pdf'), format='pdf', bbox_inches='tight', dpi=400)


def main():
    for model in ['llama-3.1-8b-instruct', 'qwen-2.5-7b-instruct']:
        if 'llama' in model:
            results_sizes_8_16: List[Dict[str, Any]] = [result for result in extract_results(f'results/sizes_8-16/{model}') if result['concurrent_adapters'] not in {64, 96}]
            results_sizes_8_16_32: List[Dict[str, Any]] = [result for result in extract_results(f'results/sizes_8-16-32/{model}') if result['concurrent_adapters'] not in {64}]
        else:
            results_sizes_8_16: List[Dict[str, Any]] = [result for result in extract_results(f'results/sizes_8-16/{model}')]
            results_sizes_8_16_32: List[Dict[str, Any]] = [result for result in extract_results(f'results/sizes_8-16-32/{model}')]

        plot_results_together(
            results_sizes_8_16=results_sizes_8_16,
            results_sizes_8_16_32=results_sizes_8_16_32,
            title=f'_{model}',
            path='.',
            y_metrics=['total_throughput', 'adapter_slots'],
            y_labels=['through.\n(tks/s)', '$A_{max}$\n(#)'],
            x_metric='concurrent_adapters',
            x_label='adapters to serve (#)',
            algorithms_to_use=['lora-serve', 'dlora-proactive-mechanism', 'proposal-starvation-2'],
        )


if __name__ == '__main__':
    main()
