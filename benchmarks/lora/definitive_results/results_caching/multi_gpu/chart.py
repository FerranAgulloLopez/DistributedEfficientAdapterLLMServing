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
        results_sizes_8_16: Dict[int, List[Dict[str, Any]]],
        results_sizes_8_16_32: Dict[int, List[Dict[str, Any]]],
        title: str,
        path: str,
        x_metric: str,
        x_label: str,
        y_metrics: List[str],
        y_labels: List[str],
        algorithms_to_use: List[str],
) -> None:
    multiplier = 1.5
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14 * multiplier,
        'axes.titlesize': 14 * multiplier,
        'axes.labelsize': 13 * multiplier,
        'xtick.labelsize': 11 * multiplier,
        'ytick.labelsize': 11 * multiplier,
        'legend.fontsize': 14 * multiplier,
        'lines.linewidth': 2.0 * multiplier,
        'mathtext.default': 'regular',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.4 * multiplier,
        'figure.figsize': (22, 3)  # Adjusted for balanced horizontal layout
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
    markers_width = {
        'baseline-3': 11,
        'baseline-4': 11,
        'baseline-6': 11,
        'lora-serve': 23,
        'lora-serve-half': 11,
        'baseline-5': 11,
        'baseline-1': 18,
        'baseline-2': 11,
        'proposal-starvation': 14,
        'proposal-starvation-fast': 9,
        'dlora-proactive-mechanism': 17,
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

    colors = {'proposal-starvation': '#1f77b4', 'lora-serve': '#ff7f0e', 'lora-serve-half': '#2ca02c', 'dlora-proactive-mechanism': '#d62728', 'proposal-starvation-fast': '#6fc0f7'}
    fig = plt.figure()
    gs = fig.add_gridspec(len(y_metrics), 2 * 2 + 1, wspace=0.05, width_ratios=[1, 1, 0.4, 1, 1])
    axs_previous_x = {}
    starvation_legend_element = {}
    memory_error_legend_element = {}
    time_error_legend_element = {}
    for index_x_group, arrival_multiplier in enumerate(results_sizes_8_16.keys()):
        for index_x_intern, (x_axis_label, results) in enumerate([('Sizes (8 & 16)', results_sizes_8_16[arrival_multiplier]),
                                                           ('Sizes (8, 16 & 32)',
                                                            results_sizes_8_16_32[arrival_multiplier])]):
            index_x = index_x_group * len(y_metrics) + index_x_intern
            if index_x_group == 1:
                index_x += 1
            for index_y, y_metric in enumerate(y_metrics):
                add_subplot_params = {}

                if index_x in {0, 3}:
                    pass
                else:
                    add_subplot_params = {
                        "sharey": axs_previous_x[index_y]
                    }

                if index_y == 0:
                    pass
                else:
                    add_subplot_params = {
                        "sharex": axs_previous_y
                    }

                axs = fig.add_subplot(gs[index_y, index_x], **add_subplot_params)

                if index_x in {0, 3}:
                    axs_previous_x[index_y] = axs
                    axs.set_ylabel(y_labels[index_y])
                else:
                    axs.tick_params(labelleft=False)

                if index_y == 0:
                    axs_previous_y = axs
                    axs.tick_params(labelbottom=False)
                    axs.set_title(x_axis_label)
                    axs.set_ylim(0.5, 4.5)
                    axs.set_yticks([1, 2, 3, 4], ["1", "2", "3", "4"])
                else:
                    axs.set_xlabel(x_label)
                    if 'qwen' in title:
                        if index_x_group == 0:
                            axs.set_ylim(-20, 250)
                            axs.set_yticks([0, 120, 240], ["0", "120", "240"])
                        else:
                            axs.set_ylim(-20, 270)
                            axs.set_yticks([0, 130, 260], ["0", "130", "260"])
                    elif 'llama' in title:
                        if index_x_group == 0:
                            axs.set_ylim(13, 26)
                            # axs.set_yticks([0, 120, 240], ["0", "120", "240"])
                        else:
                            axs.set_ylim(5, 80)
                            # axs.set_yticks([0, 130, 260], ["0", "130", "260"])

                processed_results = __prepare_lines(
                    results,
                    x_metric,
                    y_metric,
                    'placement_algorithm',
                    add_all_info=True
                )

                processed_results = {value: (x_line, y_line, z_line) for value, x_line, y_line, z_line in
                                     processed_results}

                def difference_lines(x, y):
                    min_length = min(len(x[0]), len(y[0]))
                    assert x[1][:min_length] == y[1][:min_length]
                    x = x[0][:min_length]
                    y = y[0][:min_length]
                    output = sum(abs(a - b) for a, b in zip(x, y)) / len(x)
                    return output

                def prepare_line(x):
                    x_line, y_line, z_line = x
                    while z_line[-1]['placement_not_possible'] is True:
                        x_line = x_line[:-1]
                        y_line = y_line[:-1]
                        z_line = z_line[:-1]
                    if z_line[-1]['memory_error'] is True:
                        while z_line[-1]['total_throughput'] is None:
                            x_line = x_line[:-1]
                            y_line = y_line[:-1]
                            z_line = z_line[:-1]
                    x_line, y_line = zip(*sorted(zip(x_line, y_line)))
                    x_line = list(x_line)
                    y_line = list(y_line)
                    while x_line[-1] is None:
                        x_line = x_line[:-1]
                        y_line = y_line[:-1]
                    while y_line[-1] is None:
                        x_line = x_line[:-1]
                        y_line = y_line[:-1]

                    return y_line, x_line

                if y_metric == 'used_servers':
                    try:
                        proposed = prepare_line(processed_results['proposal-starvation-2'])
                        lora_serve = prepare_line(processed_results['lora-serve'])
                        dlora = prepare_line(processed_results['dlora-proactive-mechanism'])
                        print('difference between proposed and loraserve')
                        print(difference_lines(proposed, lora_serve))
                        print('difference between proposed and dlora')
                        print(difference_lines(proposed, dlora))
                    except:
                        print("difference computation exploded")


                legend_elements = []
                for value in algorithms_to_use:
                    if value not in processed_results:
                        continue
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
                        line = axs.plot(
                            x_line,
                            y_line,
                            linestyle=linestyles[value],
                        )[0]
                        colors[value] = line.get_color()
                    else:
                        line = axs.plot(
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
                            markersize=15,
                        )
                    )

                    # mark memory errors
                    if memory_error:
                        x_final_point = x_line[-1] + (x_line[-1] - x_line[-2]) / 2
                        y_final_point = (y_line[-1] + (
                                    y_line[-1] - y_line[-2]) / 2) if value == 'dlora-proactive-mechanism' else (
                                    y_line[-1] + (y_line[-1] - y_line[-2]) / 4)
                        axs.plot(
                            list(x_line[-1:]) + [x_final_point],
                            list(y_line[-1:]) + [y_final_point],
                            marker='x',
                            markevery=[1],
                            linestyle=linestyles[value],
                            markersize=13,
                            markeredgewidth=2,
                            color=line.get_color(),
                        )

                        if good_labels[value] not in memory_error_legend_element:
                            memory_error_legend_element[good_labels[value]] = Line2D(
                                [],
                                [],
                                marker='x',
                                markevery=[1],
                                markersize=13,
                                markeredgewidth=2,
                                color=line.get_color(),
                                linestyle=None,
                                linewidth=0,
                            )

                    # mark time error
                    if value == 'dlora-proactive-mechanism' and 'qwen' in title and x_line[-1] != 1280:
                        x_final_point = x_line[-1] + (x_line[-1] - x_line[-2]) / 2
                        y_final_point = y_line[-1] + (y_line[-1] - y_line[-2]) / 2
                        axs.plot(
                            list(x_line[-1:]) + [x_final_point],
                            list(y_line[-1:]) + [y_final_point],
                            marker='*',
                            markevery=[1],
                            linestyle=linestyles[value],
                            markersize=13,
                            markeredgewidth=1,
                            color=line.get_color(),
                        )
                        if good_labels[value] not in time_error_legend_element:
                            time_error_legend_element[good_labels[value]] = Line2D(
                                [],
                                [],
                                marker='*',
                                markevery=[1],
                                linestyle=None,
                                linewidth=0,
                                markersize=13,
                                markeredgewidth=1,
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
                    axs.plot(
                        starvation_x_line,
                        starvation_y_line,
                        marker=markers[value],
                        markerfacecolor='none',
                        color=line.get_color(),
                        markersize=markers_width[value],
                        markeredgewidth=1,
                        linestyle='None',
                    )
                    axs.plot(
                        not_starvation_x_line,
                        not_starvation_y_line,
                        marker=markers[value],
                        color=line.get_color(),
                        markersize=markers_width[value],
                        markeredgewidth=1,
                        linestyle='None',
                    )

                    if len(starvation_x_line) > 0 and good_labels[value] not in starvation_legend_element:
                        starvation_legend_element[good_labels[value]] = Line2D(
                            [],
                            [],
                            color=line.get_color(),
                            marker=markers[value],
                            linestyle=None,
                            linewidth=0,
                            label=good_labels[value],
                            markersize=11,
                            markerfacecolor='none',
                        )
    # from matplotlib.ticker import FuncFormatter
    '''axs[index_x].set_ylabel(y_labels[index_x])
    axs[index_x].set_xlabel(x_label)'''
    '''if 'through' in y_labels[index_x]:
        axs[index_x].yaxis.set_major_formatter(FuncFormatter(thousands_formatter))'''

    fig.text(0.305, 1.05, 'Arrivals x25' if 'qwen' in title else 'Arrivals x1', ha='center', fontweight="bold")
    fig.text(0.73, 1.05, 'Arrivals x50' if 'qwen' in title else 'Arrivals x10', ha='center', fontweight="bold")

    from matplotlib.legend_handler import HandlerTuple
    legend_elements = list(reversed(legend_elements))
    labels = [item.get_label() for item in legend_elements]
    if len(starvation_legend_element) > 0:
        legend_elements.append(tuple(starvation_legend_element.values()))
        labels.append('Starvation')
    if len(memory_error_legend_element) > 0:
        legend_elements.append(tuple(memory_error_legend_element.values()))
        labels.append('Memory error')
    if len(time_error_legend_element) > 0:
        legend_elements.append(tuple(time_error_legend_element.values()))
        labels.append('Time error')
    '''legend_elements.append(
        Line2D([0], [0], marker='x', color='gray', markerfacecolor='gray', markersize=10, linestyle='None',
               markeredgewidth=5))
    labels.append('Memory error')'''
    '''if y_axis_label == 'low_rates-mixed_sizes':
        labels.append('Time error')
        legend_elements.append(
            Line2D([0], [0], marker='x', color='gray', markerfacecolor='gray', markersize=10, linestyle='None',
                   markeredgewidth=5))'''
    '''else:
        labels.append('Memory error')'''
    if 'llama' in title:
        fig.legend(
            handles=legend_elements,
            labels=labels,
            loc='upper center',
            ncol=len(labels),
            bbox_to_anchor=(0.5, 1.4),
            handler_map={tuple: HandlerTuple(ndivide=None)},
        )
    else:
        fig.legend(
            handles=legend_elements,
            labels=labels,
            loc='upper center',
            ncol=len(labels),
            bbox_to_anchor=(0.5, 1.4),
            handletextpad=0.2,
            columnspacing=0.8,
            borderpad=0.3,
            handler_map={tuple: HandlerTuple(ndivide=None)},
        )

    plt.savefig(os.path.join(path, f'multi_{title}.pdf'), format='pdf', bbox_inches='tight', dpi=400)


def main():
    for model in ['llama-3.1-8b-instruct', 'qwen-2.5-7b-instruct']:
        sizes = 'sizes_8-16'
        results_sizes_8_16: List[Dict[str, Any]] = extract_results(f'results_{sizes}/{sizes}/{model}')

        sizes = 'sizes_8-16-32'
        results_sizes_8_16_32: List[Dict[str, Any]] = extract_results(f'results_{sizes}/{sizes}/{model}')

        list_of_results: Dict[int, List[Dict[str, Any]]] = {}
        for result in results_sizes_8_16:
            if result['concurrent_adapters'] in {64, 96, 128, 384, 640, 896, 1152}:
                continue
            trace_arrival_multiplier = result['trace_arrival_multiplier']
            if trace_arrival_multiplier not in list_of_results:
                list_of_results[trace_arrival_multiplier] = []
            list_of_results[trace_arrival_multiplier].append(result)
        results_sizes_8_16 = dict(sorted(list_of_results.items()))

        list_of_results: Dict[int, List[Dict[str, Any]]] = {}
        for result in results_sizes_8_16_32:
            if result['concurrent_adapters'] in {64, 96, 128, 384, 640, 896, 1152}:
                continue
            trace_arrival_multiplier = result['trace_arrival_multiplier']
            if trace_arrival_multiplier not in list_of_results:
                list_of_results[trace_arrival_multiplier] = []
            list_of_results[trace_arrival_multiplier].append(result)
        results_sizes_8_16_32 = dict(sorted(list_of_results.items()))

        plot_results_together(
            results_sizes_8_16=results_sizes_8_16,
            results_sizes_8_16_32=results_sizes_8_16_32,
            title=model,
            path='.',
            y_metrics=['used_servers', 'itl'],
            y_labels=['used\nGPUs (#)\n', 'ITL\n(ms)'],
            x_metric='concurrent_adapters',
            x_label='adapters to serve (#)',
            algorithms_to_use=['lora-serve', 'dlora-proactive-mechanism', 'proposal-starvation-2', 'proposal-starvation-2-fast'],
        )


if __name__ == '__main__':
    main()
