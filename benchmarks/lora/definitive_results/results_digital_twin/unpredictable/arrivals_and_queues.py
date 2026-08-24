import csv
import os
from typing import List, Tuple, Dict, Set, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


EXPS_TO_PLOT = {
    'high_rates': {
        'real': 'results/traces_example/real_results/llama-3.1-8b-instruct/rates_1.6_0.8_0.4/_32_32_8_',
        'simulated': 'results/traces_example/simulation_results_full/llama-3.1-8b-instruct/rates_1.6_0.8_0.4/_32_32_8_',
    },
}


def extract_arrivals_per_adapter(path: str) -> Dict[str, List[float]]:
    arrivals_per_adapter: Dict[str, List[float]] = {}
    with open(os.path.join(path, 'arrivals.csv'), newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for arrival_time, _, _, adapter in reader:
            if adapter not in arrivals_per_adapter:
                arrivals_per_adapter[adapter] = []
            arrivals_per_adapter[adapter].append(float(arrival_time))
    return arrivals_per_adapter


def extract_time(input_path: str) ->np.ndarray:
    path = os.path.join(input_path, 'time_.npy')
    if not os.path.exists(path):
        path = os.path.join(input_path, 'debug_time_array.npy')
    return np.load(path)


def extract_running_queue(input_path: str) ->np.ndarray:
    path = os.path.join(input_path, 'num_running_.npy')
    if not os.path.exists(path):
        path = os.path.join(input_path, 'debug_num_running_array.npy')
    return np.load(path)


def extract_waiting_queue(input_path: str) ->np.ndarray:
    path = os.path.join(input_path, 'num_waiting_.npy')
    if not os.path.exists(path):
        path = os.path.join(input_path, 'debug_num_waiting_array.npy')
    return np.load(path)


def plot_results(
        arrivals_per_adapter: Dict[str, List[float]],
        time: np.ndarray,
        running_queue: np.ndarray,
        waiting_queue: np.ndarray,
        simulated_time: np.ndarray,
        simulated_running_queue: np.ndarray,
        simulated_waiting_queue: np.ndarray,
        title: str,
        path: str,
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
    fig, axs = plt.subplots(nrows, ncols, sharex=True)
    fig.subplots_adjust(wspace=0.2)

    # plot arrivals
    index_x = 0
    random.seed(0)
    adapters: List[str] = random.sample(list(arrivals_per_adapter.keys()), k=3)  # subsample adapters
    min_arrival: float = max([arrivals[0] for arrivals in arrivals_per_adapter.values()])
    min_arrival = min(min_arrival, time[0])
    arrivals_per_adapter = {key: (np.asarray(values) - min_arrival) / 60 for key, values in arrivals_per_adapter.items()}
    for adapter in adapters:
        axs[index_x].hist(arrivals_per_adapter[adapter], bins=60, alpha=0.5, label=adapter.replace('dummy', 'adapter'))

    axs[index_x].legend()
    axs[index_x].set_xlabel('time (min)')
    axs[index_x].set_ylabel('arrivals per adapter')
    axs[index_x].set_title('Arrivals')

    # update time
    if np.shape(time)[0] == (np.shape(running_queue)[0] - 1):
        time = np.concatenate([[time[0]], time, ])
    time = (time - min_arrival) / 60

    # update simulated time
    simulated_time = (simulated_time - min_arrival) / 60

    # plot running queue
    index_x = 1
    line_running = axs[index_x].plot(time, running_queue, label='Real running')[0]
    line_running_simulated = axs[index_x].plot(simulated_time, simulated_running_queue, label='DT running')[0]

    # plot waiting queue
    index_x = 1
    line_waiting_simulated = axs[index_x].plot(simulated_time, simulated_waiting_queue, label='DT waiting')[0]
    line_waiting = axs[index_x].plot(time, waiting_queue, label='Real waiting')[0]

    # legend right
    index_x = 1
    '''legend_elements = []
    legend_elements.append(line_running)
    legend_elements.append(line_waiting)
    legend = axs[index_x].legend(
        title='real',
        loc='upper left',
        handles=legend_elements,
    )
    axs[index_x].add_artist(legend)
    legend_elements = []
    legend_elements.append(line_running_simulated)
    legend_elements.append(line_waiting_simulated)
    axs[index_x].legend(
        title='DT',
        loc='upper right',
        handles=legend_elements,
    )'''
    legend_elements = []
    legend_elements.append(line_running)
    legend_elements.append(line_waiting)
    legend_elements.append(line_running_simulated)
    legend_elements.append(line_waiting_simulated)
    axs[index_x].legend(
        loc='upper center',
        handles=legend_elements,
        bbox_to_anchor = (0.5, 1.48),
        ncol=len(legend_elements) / 2,
    )

    axs[index_x].set_xlabel('time (min)')
    axs[index_x].set_ylabel('adapters (#)')
    axs[index_x].set_title('Adapters in running and waiting queues')

    '''legend_elements_axis = [Patch(facecolor=line.get_color(), edgecolor='black', label=line.get_label()) for line in lines]
    # axs[index_x].legend(handles=legend_elements_axis, loc='upper right')

    axis_title = metrics_labels[metric].replace(' (toks/s)', '').replace(' (ms)', '').capitalize()
    if 'ttft' in metric:
        axis_title = 'Mean TTFT (ms)'
    elif 'itl' in metric:
        axis_title = 'Mean ITL (ms)'
    elif 'total_throughput' in metric:
        axis_title = 'Throughput (toks/s)'

    # axs[index_x].set_ylabel(metrics_labels[metric])
    axs[index_x].set_title(axis_title)'''

    '''legend_elements = []
    legend_elements.append(Line2D([], [], color='black', linestyle='solid', label='Real results'))
    legend_elements.append(Line2D([], [], color='black', linestyle='dashed', label='Digital Twin'))
    legend_elements += legend_elements_axis
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), bbox_to_anchor=(0.5, 1.15))'''

    plt.savefig(os.path.join(path, f'arrivals_and_queues_{title}.pdf'), format='pdf', bbox_inches='tight', dpi=400)


def main():
    global EXPS_TO_PLOT

    # extract arrivals
    arrivals_per_adapter_dict: Dict[str, Dict[str, List[float]]] = {}
    for label, label_paths in EXPS_TO_PLOT.items():
        arrivals_per_adapter_dict[label] = extract_arrivals_per_adapter(label_paths['real'])

    # extract times
    times: Dict[str, np.ndarray] = {}
    simulated_times: Dict[str, np.ndarray] = {}
    for label, label_paths in EXPS_TO_PLOT.items():
        times[label] = extract_time(label_paths['real'])
        simulated_times[label] = extract_time(label_paths['simulated'])

    # extract running queue
    running_queues: Dict[str, np.ndarray] = {}
    simulated_running_queues: Dict[str, np.ndarray] = {}
    for label, label_paths in EXPS_TO_PLOT.items():
        running_queues[label] = extract_running_queue(label_paths['real'])
        simulated_running_queues[label] = extract_running_queue(label_paths['simulated'])

    # extract waiting queue
    waiting_queues: Dict[str, np.ndarray] = {}
    simulated_waiting_queues: Dict[str, np.ndarray] = {}
    for label, label_paths in EXPS_TO_PLOT.items():
        waiting_queues[label] = extract_waiting_queue(label_paths['real'])
        simulated_waiting_queues[label] = extract_waiting_queue(label_paths['simulated'])

    # plot arrivals and queues
    for label, arrivals_per_adapter in arrivals_per_adapter_dict.items():
        plot_results(
            arrivals_per_adapter=arrivals_per_adapter,
            time=times[label],
            running_queue=running_queues[label],
            waiting_queue=waiting_queues[label],
            simulated_time=simulated_times[label],
            simulated_running_queue=simulated_running_queues[label],
            simulated_waiting_queue=simulated_waiting_queues[label],
            path='.',
            title=label,
        )


if __name__ == '__main__':
    main()
