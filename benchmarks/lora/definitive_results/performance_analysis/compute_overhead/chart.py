import os
import re
import json
import glob
from typing import List, Tuple, Dict, Set
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec


def extract_experiment_metric(path: str) -> Dict[str, float]:
    output: Dict[str, float] = {}
    filenames: List[str] = glob.glob(os.path.join(path, 'openai-*.json'))
    for i in range(len(filenames) - 1, -1, -1):
        if 'intermediate' in filenames[i]:
            del filenames[i]
    if len(filenames) != 1:
        raise ValueError(f'More than one output result file or none {filenames} for path {path}')
    with open(filenames[0]) as metrics_file:
        metrics: dict = json.load(metrics_file)

    filenames: List[str] = glob.glob(os.path.join(path, 'server_out.log'))
    if len(filenames) != 1:
        raise ValueError(f'More than one output result file or none {filenames} for path {path}')
    with open(filenames[0]) as metrics_file:
        log: str = metrics_file.read()

    # compute throughput
    output['total_throughput'] = float(metrics['input_throughput']) + float(metrics['output_throughput'])

    # compute max batch size
    num_running: np.ndarray = np.load(os.path.join(path, 'num_running.npy'))
    output['max_batch_size'] = float(np.max(num_running))

    # compute duration
    output['duration'] = int(metrics['duration'])

    # compute itl
    output['mean_itl_ms'] = int(metrics['mean_itl_ms'])

    # compute lora loading time
    pattern = f"Total LoRA loading time:( +?)([+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)) seconds"
    found = re.findall(pattern, log)[-1]
    if found is None:
        raise ValueError(f'Metric pattern not found on result log')
    output['lora_loading_time'] = float(found[2])

    # compute lora activating time
    pattern = f"Total LoRA activating time:( +?)([+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)) seconds"
    found = re.findall(pattern, log)[-1]
    if found is None:
        raise ValueError(f'Metric pattern not found on result log')
    output['lora_activating_time'] = float(found[2])

    # compute mean loras per batch
    pattern = f"Mean LoRAs by batch:( +?)([+-]?([0-9]+([.][0-9]*)?|[.][0-9]+))"
    found = re.findall(pattern, log)[-1]
    if found is None:
        raise ValueError(f'Metric pattern not found on result log')
    output['mean_loras_by_batch'] = float(found[2])

    # compute throughput
    input_tokens: int = int(metrics['total_input_tokens'])
    output_tokens: int = int(metrics['total_output_tokens'])
    output['total_throughput_without_loading'] = (output_tokens + input_tokens) / (output['duration'] - output['lora_loading_time'])
    output['total_throughput_without_loading_and_activating'] = (output_tokens + input_tokens) / (output['duration'] - output['lora_loading_time'] - output['lora_activating_time'])

    return output


def extract_results(path: str) -> List[Dict[str, float]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['m', 'rank', 'cpu_loras']
    results = []
    rerun_errors: List[str] = []
    unknown_errors: int = 0
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            m: int = int(folder.split('_')[1])
            rank: int = int(folder.split('_')[2])
            cpu_loras: int = int(folder.split('__')[1])
            try:
                metrics = extract_experiment_metric(os.path.join(path, folder))
            except ValueError:
                error_message: str = f'WARNING! Error while extracting results -> {os.path.join(path, folder)}. '
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
                # print(error_message)
                metrics = {}
            metrics['m'] = m
            metrics['rank'] = rank
            metrics['cpu_loras'] = cpu_loras
            _id = create_id(metrics, id_metrics)
            if _id in collected_ids:
                raise ValueError('Repeated results')
            collected_ids.add(_id)
            results.append(metrics)
    print(f'Unknown extraction errors: {unknown_errors}. Should be zero.')
    print(f'Rerun errors: {len(rerun_errors)}. Should be zero. Full list: {rerun_errors}')
    return results


def __prepare_lines(results: List[Dict[str, float]], x_axis: str, y_axis: str, selection: str) -> List[
    Tuple[str, List[int], List[float]]]:
    output_tmp: Dict[str, Tuple[List[int], List[float]]] = {}
    for item in results:
        selection_id = item[selection]
        if selection_id not in output_tmp:
            output_tmp[selection_id] = ([], [])
        if x_axis not in item:
            output_tmp[selection_id][0].append(None)
        else:
            output_tmp[selection_id][0].append(item[x_axis])
        if y_axis not in item:
            output_tmp[selection_id][1].append(None)
        else:
            output_tmp[selection_id][1].append(item[y_axis])
    output: List[Tuple[str, List[int], List[float]]] = []
    for key, (x_values, y_values) in output_tmp.items():
        x_line = [x_value for index, x_value in enumerate(x_values) if
                  x_value is not None and y_values[index] is not None]
        y_line = [y_value for index, y_value in enumerate(y_values) if
                  y_value is not None and x_values[index] is not None]
        y_line = [y_value for _, y_value in sorted(zip(x_line, y_line))]
        x_line.sort()
        output.append(
            (
                key,
                x_line,
                y_line
            )
        )
    output = [value for _, value in sorted(zip([value[0] for value in output], output))]

    return output


def print_variable_from_experiments(
        model_results: List[Dict[str, float]],
        variable: str
) -> None:
    variable_values: Dict[float, List[float]] = {}
    for exp in model_results:
        rank = exp['rank']
        if rank not in variable_values:
            variable_values[rank] = []
        variable_values[rank].append(exp[variable])
    for rank, rank_values in variable_values.items():
        variable_values[rank] = sorted(rank_values)
    print(f'{variable} of experiments: {variable_values}')


def plot_computation_overhead_small_new(
        model_results: Dict[str, Dict[str, List[Dict[str, float]]]],
        path: str,
        title: str,
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
        'figure.figsize': (13, 3)  # Adjusted for balanced horizontal layout
    })

    nrows = 1
    ncols = 2
    fig, axs = plt.subplots(nrows, ncols)
    fig.subplots_adjust(wspace=0.1)

    color_by_rank = {}
    new_labels = {
        'llama-2-7b': 'Llama-2-7B',
        'llama-2-13b': 'Llama-2-13B',
    }
    linestyles = {
        'Llama-2-7B': 'solid'
    }
    symbols = {
        'Llama-2-7B': 'o',
        'Llama-2-13B': '^'
    }

    model_results = {new_labels[key]: value['MediumRequest'] for key, value in model_results.items()}

    results_throughput_list = []
    for label_results, results in model_results.items():
        results_throughput = __prepare_lines(
            results,
            'mean_loras_by_batch',
            'total_throughput_without_loading_and_activating',
            'rank'
        )
        results_throughput_list.append((label_results, results_throughput))

    index_x = 0
    for label_results, results_throughput in results_throughput_list:
        for rank, x_line, y_line in results_throughput:
            y_line = np.asarray(y_line)
            print(y_line)
            y_line = (1 - y_line / y_line[0]) * 100
            if rank not in color_by_rank:
                line = axs[index_x].plot(
                    x_line,
                    y_line,
                    marker=symbols[label_results],
                    linestyle=linestyles[label_results],
                )[0]
                color_by_rank[rank] = line.get_color()
            else:
                axs[index_x].plot(
                    x_line,
                    y_line,
                    marker=symbols[label_results],
                    color=color_by_rank[rank],
                    linestyle=linestyles[label_results],
                )
        axs[index_x].set_xlabel('adapters (#)')
        # axs[index_x].set_title(label_results)

        axs[0].set_title('Throughput slowdown (%)')

    xticks = axs[index_x].get_xticks()
    yticks = axs[index_x].get_yticks()
    xticks[1] = 1
    yticks[1] = 1
    xticks = [str(round(value)) for value in xticks]
    yticks = [str(round(value)) for value in yticks]
    axs[index_x].set_xticklabels(xticks)
    axs[index_x].set_yticklabels(yticks)

    index_x = 1
    results_throughput_list = []
    for label_results, results in model_results.items():
        results_throughput = __prepare_lines(
            results,
            'mean_loras_by_batch',
            'mean_itl_ms',
            'rank'
        )
        results_throughput_list.append((label_results, results_throughput))

    for label_results, results_throughput in results_throughput_list:
        for rank, x_line, y_line in results_throughput:
            y_line = np.asarray(y_line)
            y_line = (1 - y_line[0] / y_line) * 100
            axs[index_x].plot(
                x_line,
                y_line,
                marker=symbols[label_results],
                label=f'size {rank}',
                color=color_by_rank[rank],
                linestyle=linestyles[label_results],
            )
    axs[index_x].set_xlabel('adapters (#)')
    # axs[index_x].set_xlim(0, 100)
    # axs[index_x].set_title(label_results)

    axs[index_x].set_title('ITL overhead (%)')
    xticks = axs[index_x].get_xticks()
    yticks = axs[index_x].get_yticks()
    xticks[1] = 1
    yticks[1] = 1
    xticks = [str(round(value)) for value in xticks]
    yticks = [str(round(value)) for value in yticks]
    axs[index_x].set_xticklabels(xticks)
    axs[index_x].set_yticklabels(yticks)

    legend_elements = []
    for rank, linestyle in linestyles.items():
        legend_elements.append(Line2D([], [], linestyle=linestyle, marker=symbols[rank], color='gray', label=rank))
    for label_results, color in color_by_rank.items():
        legend_elements.append(Patch(facecolor=color, edgecolor='black', label=f'size {label_results}'))
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), bbox_to_anchor=(0.5, 1.15))

    plt.savefig(os.path.join(path, f'computation_overhead_throughput_{title}_small_new.pdf'), format='pdf', bbox_inches='tight', dpi=400)


def main():
    model = 'llama-2-7b'
    results_mean: List[Dict[str, float]] = extract_results(f'{model}/mean_dataset')

    print_variable_from_experiments(results_mean, 'max_batch_size')
    print_variable_from_experiments(results_mean, 'duration')
    print_variable_from_experiments(results_mean, 'lora_loading_time')
    print_variable_from_experiments(results_mean, 'lora_activating_time')
    print_variable_from_experiments(results_mean, 'mean_loras_by_batch')

    model_results: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    for model in ['llama-2-7b', 'llama-2-13b']:
        results_mean: List[Dict[str, float]] = extract_results(f'{model}/mean_dataset')
        model_results[model] = {
            'MediumRequest': results_mean,
        }

    plot_computation_overhead_small_new(
        model_results,
        '.',
        ''
    )


if __name__ == '__main__':
    main()
