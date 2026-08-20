import glob
import json
import os
import random
import re
from typing import List, Tuple, Dict, Set

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

REQUEST_LENGTHS = {
        'SmallRequest': (23, 27),
        'MediumRequest': (250, 231),
        'LargeRequest': (423, 358)
}


def comp_overhead_predictor(
        x,
        constant_1=5.65871878e-04,
        constant_2=7.80809137e-02,
):  # slowdown estimation
    cpu_loras = x[0]
    batch_size = x[1]
    assert np.all(cpu_loras >= 1)
    assert np.all(batch_size >= 1)
    output = constant_1 * np.minimum(batch_size, cpu_loras) + constant_2
    return output


def __prepare_lines(results: List[Dict[str, float]], x_axis: str, y_axis: str, selection: str, filter_in: Tuple[str, str] = None, additional_line: str = None) -> List[
    Tuple[str, List[int], List[float]]]:
    output_tmp: Dict[str, Tuple[List[int], List[float]]] = {}
    for item in results:
        selection_id = item[selection]
        if filter_in is not None and str(item[filter_in[0]]) != filter_in[1]:
            continue
        if selection_id not in output_tmp:
            output_tmp[selection_id] = ([], [])
            if additional_line is not None:
                output_tmp[selection_id] = ([], [], [])
        if x_axis not in item:
            output_tmp[selection_id][0].append(None)
        else:
            output_tmp[selection_id][0].append(item[x_axis])
        if y_axis not in item:
            output_tmp[selection_id][1].append(None)
        else:
            output_tmp[selection_id][1].append(item[y_axis])
        if additional_line is not None:
            if additional_line not in item:
                output_tmp[selection_id][2].append(None)
            else:
                output_tmp[selection_id][2].append(item[additional_line])
    output: List[Tuple[str, List[int], List[float]]] = []
    for key, values in output_tmp.items():
        if additional_line is None:
            x_values, y_values = values
        else:
            x_values, y_values, z_values = values
        x_line = [x_value for index, x_value in enumerate(x_values) if
                  x_value is not None and y_values[index] is not None]
        y_line = [y_value for index, y_value in enumerate(y_values) if
                  y_value is not None and x_values[index] is not None]
        if additional_line is not None:
            z_line = [z_value for index, z_value in enumerate(z_values) if
                  z_value is not None and x_values[index] is not None]
        y_line = [y_value for _, y_value in sorted(zip(x_line, y_line))]
        if additional_line is not None:
            z_line = [z_value for _, z_value in sorted(zip(x_line, z_line))]
        x_line.sort()
        if additional_line is None:
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


def extract_results_computation_overhead(path: str) -> List[Dict[str, float]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

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

        # compute latency
        output['mean_itl_ms'] = float(metrics['mean_itl_ms'])

        # compute max batch size
        num_running: np.ndarray = np.load(os.path.join(path, 'num_running.npy'))
        output['max_batch_size'] = float(np.max(num_running))

        # compute mean batch size
        num_running: np.ndarray = np.load(os.path.join(path, 'num_running.npy'))
        output['mean_batch_size'] = float(np.mean(num_running))

        # compute mean loras per batch
        pattern = r'Mean LoRAs by batch:( +?)([+-]?([0-9]+([.][0-9]*)?|[.][0-9]+))'
        found = re.findall(pattern, log)[-1]
        if found is None:
            raise ValueError(f'Metric pattern not found on result log')
        output['mean_loras_by_batch'] = float(found[2])

        return output

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


def process_latency(latency: List[float]):
    latency = np.asarray(latency)

    # clean initial value after 0
    min_value = np.min(latency[1:])
    latency[1] = min_value

    # assure is ascendant
    past_value = min_value
    for index in range(2, np.shape(latency)[0]):
        if latency[index] < past_value:
            latency[index] = past_value
        past_value = latency[index]

    return latency


def plot_relationship(
        computation_overhead: Dict[str, List[Dict[str, float]]],
        path: str,
        title: str,
) -> None:
    nrows = 1
    ncols = 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
    # fig.subplots_adjust(wspace=0)

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    rank_color = {}
    dataset_linestyle = {
        'SmallRequest': 'solid',
        'MediumRequest': 'dashed',
        'LargeRequest': 'dotted'
    }

    all_results_computation_overhead = []
    for label_results, aux_results in computation_overhead.items():
        all_results_computation_overhead.append(
            (
                label_results,
                __prepare_lines(
                    aux_results,
                    'mean_loras_by_batch',
                    'mean_itl_ms',
                    'rank'
                )
            )
        )

    index_x = 0
    for label_results, dataset_results_memory_overhead in all_results_computation_overhead:
        for rank, x_line_comp, y_line_comp in dataset_results_memory_overhead:
            y_line_comp = np.asarray(y_line_comp)
            if rank not in rank_color:
                line = axs[index_x].plot(
                    x_line_comp,
                    y_line_comp,
                    marker='',
                    linestyle=dataset_linestyle[label_results],
                    label=f'real rank {rank}'
                )[0]
                rank_color[rank] = line.get_color()
            else:
                axs[index_x].plot(
                    x_line_comp,
                    y_line_comp,
                    marker='',
                    linestyle=dataset_linestyle[label_results],
                    color=rank_color[rank],
                    label=f'real rank {rank}'
                )

        axs[index_x].set_ylabel('latency (ms)', fontsize=10)
        axs[index_x].set_xlabel('average unique adapters in batch (#)', fontsize=10)

    index_x = 1
    for label_results, dataset_results_memory_overhead in all_results_computation_overhead:
        for rank, x_line_comp, y_line_comp in dataset_results_memory_overhead:
            y_line_comp = process_latency(y_line_comp)

            # y_line_comp = (1 - y_line_comp[0] / y_line_comp) * 100
            if rank not in rank_color:
                line = axs[index_x].plot(
                    x_line_comp,
                    y_line_comp,
                    marker='',
                    linestyle=dataset_linestyle[label_results],
                    label=f'real rank {rank}'
                )[0]
                rank_color[rank] = line.get_color()
            else:
                axs[index_x].plot(
                    x_line_comp,
                    y_line_comp,
                    marker='',
                    linestyle=dataset_linestyle[label_results],
                    color=rank_color[rank],
                    label=f'real rank {rank}'
                )

        axs[index_x].set_ylabel('latency processed (ms)', fontsize=10)
        axs[index_x].set_xlabel('average unique adapters in batch (#)', fontsize=10)

    index_x = 2
    for label_results, dataset_results_memory_overhead in all_results_computation_overhead:
        for rank, x_line_comp, y_line_comp in dataset_results_memory_overhead:
            y_line_comp = process_latency(y_line_comp)
            y_line_comp = (1 - y_line_comp[0] / y_line_comp) * 100
            if rank not in rank_color:
                line = axs[index_x].plot(
                    x_line_comp,
                    y_line_comp,
                    marker='',
                    linestyle=dataset_linestyle[label_results],
                    label=f'real rank {rank}'
                )[0]
                rank_color[rank] = line.get_color()
            else:
                axs[index_x].plot(
                    x_line_comp,
                    y_line_comp,
                    marker='',
                    linestyle=dataset_linestyle[label_results],
                    color=rank_color[rank],
                    label=f'real rank {rank}'
                )

        axs[index_x].set_ylabel('latency processed overhead (%)', fontsize=10)
        axs[index_x].set_xlabel('average unique adapters in batch (#)', fontsize=10)

    all_results_computation_overhead = []
    for label_results, aux_results in computation_overhead.items():
        all_results_computation_overhead.append(
            (
                label_results,
                __prepare_lines(
                    aux_results,
                    'cpu_loras',
                    'mean_loras_by_batch',
                    'rank',
                    additional_line='mean_batch_size'
                )
            )
        )

    index_x = 3
    for label_results, dataset_results_memory_overhead in all_results_computation_overhead:
        for rank, x_line_comp, y_line_comp, z_line in dataset_results_memory_overhead:
            x_line_comp = np.asarray(x_line_comp)
            y_line_comp = np.asarray(y_line_comp)
            z_line = np.asarray(z_line)
            axs[index_x].plot(
                np.minimum(x_line_comp, z_line),
                y_line_comp,
                marker='',
                linestyle=dataset_linestyle[label_results],
                color=rank_color[rank],
                label=f'real rank {rank}'
            )

        axs[index_x].set_ylabel('average unique adapters in batch (#)', fontsize=10)
        axs[index_x].set_xlabel('min(G\', B) (#)', fontsize=10)

    legend_elements = []
    #legend_elements.append(Line2D([], [], color='none', label='rank'))
    for color_key, color_value in rank_color.items():
        legend_elements.append(Line2D([], [], color=color_value, label=f'rank {color_key}'))
    #legend_elements.append(Line2D([], [], color='none', label='dataset'))
    for linestyle_key, linestyle_value in dataset_linestyle.items():
        legend_elements.append(Line2D([], [], color='black', linestyle=linestyle_value, label=linestyle_key))

    leg = fig.legend(handles=legend_elements, fontsize=10, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1))
    for item, label in zip(leg.legend_handles, leg.texts):
        if label._text in ['rank', 'dataset']:
            width = item.get_window_extent(fig.canvas.get_renderer()).width
            label.set_ha('left')
            label.set_position((-2 * width, 0))

    plt.savefig(os.path.join(path, f'compute_relationships_{title}'), bbox_inches='tight')


def predict_comp_overhead(
        computation_overhead: Dict[str, List[Dict[str, float]]],
        path: str,
        title: str,
) -> None:
    all_results_computation_overhead = []
    for label_results, aux_results in computation_overhead.items():
        all_results_computation_overhead.append(
            (
                label_results,
                __prepare_lines(
                    aux_results,
                    'cpu_loras',
                    'mean_itl_ms',
                    'rank',
                    additional_line='mean_batch_size'
                )
            )
        )

    nrows = 1
    ncols = len(all_results_computation_overhead[0][1])
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4), sharex=True, sharey=True)
    # fig.subplots_adjust(wspace=0)

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    xdata = []
    ydata = []
    for index_x, (label_results, dataset_results_memory_overhead) in enumerate(all_results_computation_overhead):
        request_input_size, request_output_size = REQUEST_LENGTHS[label_results]
        for rank, x_line_comp, y_line_comp, z_line_comp in dataset_results_memory_overhead:
            adapter_size = rank
            x_line_comp = np.asarray(x_line_comp)
            z_line_comp = np.asarray(z_line_comp)
            y_line_comp = process_latency(y_line_comp)
            y_line_comp = (1 - y_line_comp[0] / y_line_comp)
            x_line_comp = x_line_comp[1:]  # remove no loras sample
            y_line_comp = y_line_comp[1:]  # remove no loras sample
            z_line_comp = z_line_comp[1:]  # remove no loras sample
            axs[index_x].plot(
                np.minimum(x_line_comp, z_line_comp),
                y_line_comp,
                marker='o',
                linestyle='solid',
                label=f'real rank {rank}'
            )
            if label_results == 'SmallRequest':
                continue
            if rank != 8:
                continue
            for index_y in range(len(x_line_comp)):
                cpu_loras = x_line_comp[index_y]
                batch_size = z_line_comp[index_y]
                xdata.append([cpu_loras, batch_size, request_input_size, request_output_size, adapter_size])
                ydata.append(y_line_comp[index_y])

        axs[index_x].set_ylabel('latency overhead (%)', fontsize=10)
        axs[index_x].set_xlabel('min(cpu_loras, batch size) (#)', fontsize=10)
        axs[index_x].set_title(label_results)

    aux_list = list(zip(xdata, ydata))
    random.shuffle(aux_list)
    xdata, ydata = zip(*aux_list)
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    xdata = np.transpose(xdata)

    popt_exponential, _ = curve_fit(comp_overhead_predictor, xdata, ydata)#, bounds=([0., 0., 0.], [0.0025, 10e50, 1.0e50]), maxfev=15000)
    print(popt_exponential)

    for index_x, (label_results, dataset_results_memory_overhead) in enumerate(all_results_computation_overhead):
        request_input_size, request_output_size = REQUEST_LENGTHS[label_results]
        for rank, x_line_comp, y_line_comp, z_line_comp in dataset_results_memory_overhead:
            x_line_comp = np.asarray(x_line_comp)
            z_line_comp = np.asarray(z_line_comp)
            x_line_comp = x_line_comp[1:]  # remove no loras sample
            z_line_comp = z_line_comp[1:]  # remove no loras sample
            axs[index_x].plot(
                x_line_comp,
                comp_overhead_predictor(
                    np.asarray([x_line_comp, z_line_comp, [request_input_size] * len(x_line_comp), [request_output_size] * len(x_line_comp), [rank] * len(x_line_comp)]),
                    popt_exponential[0],
                    popt_exponential[1],
                    popt_exponential[2],
                    popt_exponential[3]
                ),
                marker='o',
                linestyle='dotted',
                label=f'model rank {rank}'
            )
            axs[index_x].legend(loc='upper right', fontsize=10)

    plt.savefig(os.path.join(path, f'comp_overhead_predictor_{title}'), bbox_inches='tight')


def main():
    model = 'llama-2-7b'

    path = os.path.join('../../../benchmarks/lora/definitive_results/performance_analysis/compute_overhead', model)
    results_p25: List[Dict[str, float]] = extract_results_computation_overhead(f'{path}/p25_dataset')
    results_mean: List[Dict[str, float]] = extract_results_computation_overhead(f'{path}/mean_dataset')
    results_p75: List[Dict[str, float]] = extract_results_computation_overhead(f'{path}/p75_dataset')
    computation_overhead: Dict[str, List[Dict[str, float]]] = {  # dataset, rank, cpu_loras
        'SmallRequest': results_p25,
        'MediumRequest': results_mean,
        'LargeRequest': results_p75
    }

    plot_relationship(
        computation_overhead,
        '',
        ''
    )

    predict_comp_overhead(
        computation_overhead,
        '',
        ''
    )


if __name__ == '__main__':
    main()
