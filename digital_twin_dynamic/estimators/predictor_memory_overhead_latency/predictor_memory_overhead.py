import copy
import glob
import json
import os
import random
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
MAX_TOKEN_CAPACITY = 82528


def memory_overhead_predictor(
        x,
        constant_1=0.49785841,
        constant_2=17.20746021
):
    batch_size = x[0]
    assert np.all(batch_size >= 1)
    return constant_1 * batch_size + constant_2


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


def extract_results_memory_overhead(path: str) -> List[Dict[str, float]]:
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

        # load summary file
        with open(filenames[0]) as metrics_file:
            metrics: dict = json.load(metrics_file)

        # compute latency
        output['mean_itl_ms'] = float(metrics['mean_itl_ms'])

        # compute max batch size
        num_running: np.ndarray = np.load(os.path.join(path, 'num_running.npy'))
        output['max_batch_size'] = float(np.max(num_running))

        # compute mean batch size
        num_running: np.ndarray = np.load(os.path.join(path, 'num_running.npy'))
        output['mean_batch_size'] = float(np.mean(num_running))

        return output

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['m', 'rank']
    results = []
    rerun_errors: List[str] = []
    unknown_errors: int = 0
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            if folder == '_':
                continue
            m: int = int(folder.split('_')[1])
            rank: int = int(folder.split('_')[2])
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
            _id = create_id(metrics, id_metrics)
            if _id in collected_ids:
                raise ValueError('Repeated results')
            collected_ids.add(_id)
            results.append(metrics)
    print(f'Unknown extraction errors: {unknown_errors}. Should be zero.')
    print(f'Rerun errors: {len(rerun_errors)}. Should be zero. Full list: {rerun_errors}')
    return results


def plot_relationship(
        memory_overhead: Dict[str, List[Dict[str, float]]],
        path: str,
        title: str,
) -> None:
    all_results_memory_overhead = []
    for label_results, aux_results in memory_overhead.items():
        all_results_memory_overhead.append(
            (
                label_results,
                __prepare_lines(
                    aux_results,
                    'm',
                    'mean_itl_ms',
                    'rank',
                    additional_line='mean_batch_size'
                )
            )
        )

    nrows = 1
    ncols = 1
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

    for label_results, dataset_results_memory_overhead in all_results_memory_overhead:
        for rank, x_line_memory, y_line_memory, z_line in dataset_results_memory_overhead:
            if rank not in rank_color:
                line = axs.plot(
                    z_line,
                    y_line_memory,
                    marker='',
                    linestyle=dataset_linestyle[label_results],
                    label=f'real rank {rank}'
                )[0]
                rank_color[rank] = line.get_color()
            else:
                axs.plot(
                    z_line,
                    y_line_memory,
                    marker='',
                    linestyle=dataset_linestyle[label_results],
                    color=rank_color[rank],
                    label=f'real rank {rank}'
                )

        axs.set_ylabel('inter-token latency (ms)', fontsize=10)
        axs.set_xlabel('average batch size (reqs)', fontsize=10)
        # axs.set_ylim(0, 100)

        legend_elements = []
        legend_elements.append(Line2D([], [], color='none', label='rank'))
        for color_key, color_value in rank_color.items():
            legend_elements.append(Line2D([], [], color=color_value, label=color_key))
        legend_elements.append(Line2D([], [], color='none', label='dataset'))
        for linestyle_key, linestyle_value in dataset_linestyle.items():
            legend_elements.append(Line2D([], [], color='black', linestyle=linestyle_value, label=linestyle_key))

        leg = axs.legend(handles=legend_elements, loc='lower right', fontsize=10)
        for item, label in zip(leg.legend_handles, leg.texts):
            if label._text in ['rank', 'dataset']:
                width = item.get_window_extent(fig.canvas.get_renderer()).width
                label.set_ha('left')
                label.set_position((-2 * width, 0))

    plt.savefig(os.path.join(path, f'relationship_throughput_batch_size_{title}'), bbox_inches='tight')


def predict_mem_overhead(
        memory_overhead: Dict[str, List[Dict[str, float]]],
        path: str,
        title: str,
) -> None:
    all_results_memory_overhead = []
    for label_results, aux_results in memory_overhead.items():
        lines = __prepare_lines(
            aux_results,
            'm',
            'mean_itl_ms',
            'rank',
            additional_line='mean_batch_size'
        )
        lines_augmented = []
        for rank, x_line, y_line, z_line in lines:
            x_line_augmented = copy.deepcopy(x_line)
            y_line_augmented = copy.deepcopy(y_line)
            '''x_line_augmented.append(x_line_augmented[-1] + 16)
            y_line_augmented.append(0)
            z_line.append(1)'''
            '''split_times = 5
            split_size = y_line_augmented[-1] / split_times
            for index in range(split_times):
                x_value = x_line_augmented[-1] + 16
                y_value = y_line_augmented[-1] - split_size
                x_line_augmented.append(x_value)
                y_line_augmented.append(y_value)'''

            # remove outlier
            if rank == 8 and label_results == 'MediumRequest':
                x_line_augmented = x_line_augmented[:-1]
                y_line_augmented = y_line_augmented[:-1]
                z_line = z_line[:-1]
            lines_augmented.append((rank, x_line_augmented, y_line_augmented, z_line))
        all_results_memory_overhead.append(
            (
                label_results,
                lines_augmented
            )
        )

    nrows = 1
    ncols = len(all_results_memory_overhead[0][1])
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
    # fig.subplots_adjust(wspace=0)

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    xdata = []
    ydata = []
    for index_x, (label_results, dataset_results_memory_overhead) in enumerate(all_results_memory_overhead):
        for rank, x_line_memory, y_line_memory, z_line in dataset_results_memory_overhead:
            '''if rank != 16:
                continue'''
            '''if rank == 8:
                index_x = 0
            elif rank == 16:
                index_x = 1
            else:
                index_x = 2'''
            '''if rank != 8:
                continue'''
            x_line_memory = np.asarray(x_line_memory)
            y_line_memory = np.asarray(y_line_memory)
            axs[index_x].plot(
                z_line,
                y_line_memory,
                marker='o',
                linestyle='solid',
                label=f'real rank {rank}'
            )

            '''request_input_size, request_output_size = REQUEST_LENGTHS[label_results]
            x_line_memory = np.asarray(x_line_memory)
            y_line_memory = np.asarray(y_line_memory)
            z_line_input = [batch_size * request_input_size for batch_size in z_line]
            z_line_output = [batch_size * request_output_size for batch_size in z_line]
            axs[index_x].plot(
                z_line_input,
                y_line_memory,
                marker='o',
                linestyle='solid',
                label=f'real rank {rank} input'
            )

            axs[index_x].plot(
                z_line_output,
                y_line_memory,
                marker='o',
                linestyle='solid',
                label=f'real rank {rank} output'
            )'''

            if label_results == 'SmallRequest':
                continue
            for index_y in range(len(x_line_memory)):
                adapter_size = rank
                m = x_line_memory[index_y]
                request_input_size, request_output_size = REQUEST_LENGTHS[label_results]
                # xdata.append([adapter_size, m, request_input_size, request_output_size, MAX_TOKEN_CAPACITY])
                # xdata.append([z_line[index_y], request_input_size, request_output_size, adapter_size])
                xdata.append([z_line[index_y], adapter_size])
                ydata.append(y_line_memory[index_y])

        axs[index_x].set_ylabel('latency (ms)', fontsize=10)
        axs[index_x].set_xlabel('average batch size (#)', fontsize=10)
        axs[index_x].set_title(label_results)

    aux_list = list(zip(xdata, ydata))
    random.shuffle(aux_list)
    xdata, ydata = zip(*aux_list)
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    xdata = np.transpose(xdata)

    # popt_exponential, _ = curve_fit(memory_overhead_predictor, xdata, ydata, bounds=([0, 0, 0, 0], [10e500, 10e500, 1, 100000]), maxfev=15000)
    # popt_exponential, _ = curve_fit(memory_overhead_predictor, xdata, ydata)#, bounds=([0, -1, 0.00000001, 0], [10e500, 10e500, 10e500, 10e500]), maxfev=15000)
    popt_exponential, _ = curve_fit(memory_overhead_predictor, xdata, ydata)# , bounds=([0, 0], [10e500, 10e500, 10e500]), maxfev=15000)
    print(popt_exponential)

    for index_x, (label_results, dataset_results_memory_overhead) in enumerate(all_results_memory_overhead):
        for rank, x_line_memory, y_line_memory, z_line in dataset_results_memory_overhead:
            axs[index_x].plot(
                z_line,
                memory_overhead_predictor(
                    np.asarray([
                        z_line,
                        [rank] * len(z_line),
                    ]),
                    popt_exponential[0],
                    popt_exponential[1]
                ),
                marker='o',
                linestyle='dotted',
                label=f'model rank {rank}'
            )
            axs[index_x].legend(loc='upper right', fontsize=10)

    plt.savefig(os.path.join(path, f'mem_overhead_predictor_{title}'), bbox_inches='tight')


def main():
    model = 'llama-2-7b'

    path = os.path.join('../../../benchmarks/lora/definitive_results/performance_analysis/memory_overhead/', model)
    results_p25: List[Dict[str, float]] = extract_results_memory_overhead(f'{path}/p25_dataset')
    results_mean: List[Dict[str, float]] = extract_results_memory_overhead(f'{path}/mean_dataset')
    results_p75: List[Dict[str, float]] = extract_results_memory_overhead(f'{path}/p75_dataset')
    memory_overhead: Dict[str, List[Dict[str, float]]] = {
        'SmallRequest': results_p25,
        'MediumRequest': results_mean,
        'LargeRequest': results_p75
    }

    plot_relationship(
        memory_overhead,
        '',
        ''
    )

    predict_mem_overhead(
        memory_overhead,
        '',
        ''
    )


if __name__ == '__main__':
    main()
