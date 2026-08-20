import glob
import os
import random
import re
from typing import List, Tuple, Dict, Set

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def adapter_token_size_predictor(  # TODO if multiple, vllm takes the bigger
        x,
        constant_1=11.23434802
):
    adapter_size = x[0]
    assert np.all(adapter_size > 0)
    output = constant_1 * adapter_size
    return output


def retrieve_maximum_gpu_token_capacity(  # get the available tokens in KV cache when not using any adapter
        paths: List[str]
) -> int:
    maximum_tokens: int = None
    for path in paths:
        # load server log
        server_out: List[str] = glob.glob(os.path.join(path, 'server_out.log'))
        if len(server_out) != 1:
            raise ValueError(f'More than one output result file or none {server_out} for path {path}')
        with open(server_out[0]) as file:
            server_out: str = file.read()

        # check adapters are disabled
        pattern = r'enable_lora=False'
        found = re.findall(pattern, server_out)[-1]
        if found is None:
            raise ValueError(f'Adapters are not disabled')

        # compute tokens per block
        pattern = r'block_size=(\d+)'
        found = re.findall(pattern, server_out)[-1]
        if found is None:
            raise ValueError(f'Metric pattern not found on result log')
        tokens_per_block = int(found)

        # compute maximum blocks
        pattern = r'GPU blocks: (\d+)'
        found = re.findall(pattern, server_out)[-1]
        if found is None:
            raise ValueError(f'Metric pattern not found on result log')
        maximum_blocks = int(found)

        # compute maximum tokens
        if maximum_tokens is None:
            maximum_tokens = maximum_blocks * tokens_per_block
        elif maximum_tokens != maximum_blocks * tokens_per_block:
            raise ValueError('Maximum tokens do not match between paths')
    return maximum_tokens


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

        # load server log
        server_out: List[str] = glob.glob(os.path.join(path, 'server_out.log'))
        if len(server_out) != 1:
            raise ValueError(f'More than one output result file or none {server_out} for path {path}')
        with open(server_out[0]) as file:
            server_out: str = file.read()

        # compute tokens per block
        pattern = 'block_size=(\d+)'
        found = re.findall(pattern, server_out)[-1]
        if found is None:
            raise ValueError(f'Metric pattern not found on result log')
        tokens_per_block = int(found)

        # compute maximum blocks
        pattern = 'GPU blocks: (\d+)'
        found = re.findall(pattern, server_out)[-1]
        if found is None:
            raise ValueError(f'Metric pattern not found on result log')
        maximum_blocks = int(found)

        # compute maximum tokens
        output['maximum_tokens'] = maximum_blocks * tokens_per_block

        # compute max batch size
        num_running: np.ndarray = np.load(os.path.join(path, 'num_running.npy'))
        output['max_batch_size'] = float(np.max(num_running))

        # compute max batch size
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
            else:
                m: int = int(folder.split('_')[1])
                rank: int = int(folder.split('_')[2])
            try:
                metrics = extract_experiment_metric(os.path.join(path, folder))
            except Exception:
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


def predict_tokens_per_adapter(
        memory_overhead: Dict[str, List[Dict[str, float]]],
        path: str,
        title: str,
        maximum_tokens_without_adapters: int,
        remove_outliers: bool = False
) -> float:
    all_results_memory_overhead = []
    for label_results, aux_results in memory_overhead.items():
        all_results_memory_overhead.append(
            (
                label_results,
                __prepare_lines(
                    aux_results,
                    'm',
                    'maximum_tokens',
                    'rank'
                )
            )
        )

    nrows = 1
    ncols = 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))

    xdata = []
    ydata = []
    by_rank_x = {}
    by_rank_y = {}
    removed_outliers = 0
    for label_results, dataset_results_memory_overhead in all_results_memory_overhead:
        for rank, x_line_memory, y_line_memory in dataset_results_memory_overhead:
            for index_y in range(len(x_line_memory)):
                m = x_line_memory[index_y]
                free_tokens = y_line_memory[index_y]
                tokens_per_adapter = (maximum_tokens_without_adapters - free_tokens) / m

                if remove_outliers:
                    if rank == 8 and tokens_per_adapter > 150:
                        removed_outliers += 1
                        continue
                    elif rank == 16 and tokens_per_adapter > 190:
                        removed_outliers += 1
                        continue
                    elif rank == 32 and tokens_per_adapter > 350:
                        removed_outliers += 1
                        continue

                if rank not in by_rank_x:
                    by_rank_x[rank] = []
                    by_rank_y[rank] = []

                by_rank_x[rank].append(tokens_per_adapter)
                by_rank_y[rank].append(f'adapter {rank}')

                xdata.append([rank])
                ydata.append(tokens_per_adapter)

    print('Removed outliers:', removed_outliers)

    aux_list = list(zip(xdata, ydata))
    random.shuffle(aux_list)
    xdata, ydata = zip(*aux_list)
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    xdata = np.transpose(xdata)

    popt_exponential, _ = curve_fit(adapter_token_size_predictor, xdata, ydata)#, bounds=([0., 0., 0.], [10e5, 0.000001, 10e5]), maxfev=15000)
    print(popt_exponential)

    for rank in by_rank_x.keys():
        line = axs.plot(
            by_rank_x[rank],
            by_rank_y[rank],
            marker='o',
            label='real',
            linewidth=0
        )[0]
        prediction = adapter_token_size_predictor([rank], popt_exponential[0])
        plt.axvline(x=prediction, color=line.get_color(), label='predicted')

    #axs[index_x].set_ylabel('max batch size', fontsize=10)
    axs.set_xlabel('tokens per adapter (#)', fontsize=10)
    #axs[index_x].set_title(label_results)

    axs.legend(loc='upper right', fontsize=10)

    plt.savefig(os.path.join(path, f'adapter_token_size_predictor_{title}'), bbox_inches='tight')

    return popt_exponential[0]


def predict_max_batch_size(
        memory_overhead: Dict[str, List[Dict[str, float]]],
        path: str,
        title: str,
        maximum_tokens_without_adapters: int,
        predictor_constant: float,
        sharey: bool = False,
) -> None:
    REQUEST_LENGTHS = {
        'SmallRequest': (23, 27),
        'MediumRequest': (250, 231),
        'LargeRequest': (423, 358)
    }
    all_results_memory_overhead = []
    for label_results, aux_results in memory_overhead.items():
        all_results_memory_overhead.append(
            (
                label_results,
                __prepare_lines(
                    aux_results,
                    'm',
                    'mean_batch_size',
                    'rank'
                )
            )
        )

    nrows = 1
    ncols = len(all_results_memory_overhead[0][1])
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4), sharex=True, sharey=sharey)
    # fig.subplots_adjust(wspace=0)

    for index_x, (label_results, dataset_results_memory_overhead) in enumerate(all_results_memory_overhead):
        for rank, x_line_memory, y_line_memory in dataset_results_memory_overhead:
            x_line_memory = np.asarray(x_line_memory)
            y_line_memory = np.asarray(y_line_memory)
            axs[index_x].plot(
                x_line_memory,
                y_line_memory,
                marker='o',
                linestyle='solid',
                label=f'real rank {rank}'
            )

            predicted_batch_size = []
            for index_y in range(len(x_line_memory)):
                m = x_line_memory[index_y]
                request_input_size, request_output_size = REQUEST_LENGTHS[label_results]

                tokens_per_adapter = adapter_token_size_predictor([rank], predictor_constant)
                predicted_batch_size.append(
                    (maximum_tokens_without_adapters - tokens_per_adapter * m) / (request_input_size + request_output_size)
                )

            axs[index_x].plot(
                x_line_memory,
                predicted_batch_size,
                marker='o',
                linestyle='dotted',
                label=f'predicted rank {rank}'
            )

        axs[index_x].set_ylabel('mean batch size', fontsize=10)
        axs[index_x].set_xlabel('served adapters (#)', fontsize=10)
        axs[index_x].set_title(label_results)
        axs[index_x].legend(loc='upper right', fontsize=10)

    plt.savefig(os.path.join(path, f'mean_batch_size_predictor_IN_MAX_SCENARIOS_{title}'), bbox_inches='tight')


def main():
    model = 'llama-2-7b'

    # retrieve maximum tokens
    maximum_tokens: int = retrieve_maximum_gpu_token_capacity([
        '../../../benchmarks/lora/definitive_results/performance_analysis/memory_overhead/llama-2-7b/mean_dataset/_',
        '../../../benchmarks/lora/definitive_results/performance_analysis/memory_overhead/llama-2-7b/p25_dataset/_',
        '../../../benchmarks/lora/definitive_results/performance_analysis/memory_overhead/llama-2-7b/p75_dataset/_'
    ])

    # predict tokens per adapter size
    path = os.path.join('../../../benchmarks/lora/definitive_results/performance_analysis/memory_overhead/', model)
    results_p25: List[Dict[str, float]] = extract_results_memory_overhead(f'{path}/p25_dataset')
    results_mean: List[Dict[str, float]] = extract_results_memory_overhead(f'{path}/mean_dataset')
    results_p75: List[Dict[str, float]] = extract_results_memory_overhead(f'{path}/p75_dataset')
    memory_overhead: Dict[str, List[Dict[str, float]]] = {
        'SmallRequest': results_p25,
        'MediumRequest': results_mean,
        'LargeRequest': results_p75
    }

    for without_outliers in [False, True]:
        predictor_constant = predict_tokens_per_adapter(
            memory_overhead,
            '',
            '' if not without_outliers else 'without_outliers',
            maximum_tokens,
            remove_outliers=without_outliers
        )

        predict_max_batch_size(
            memory_overhead,
            '',
            '' if not without_outliers else 'without_outliers',
            maximum_tokens,
            predictor_constant
        )


if __name__ == '__main__':
    main()
