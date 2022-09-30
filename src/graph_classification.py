import os
import pathlib
from typing import List

import networkx as nx
from grakel.kernels import WeisfeilerLehman, VertexHistogram, ShortestPath
from grakel.utils import graph_from_networkx
from sklearn.model_selection import train_test_split

from src.train_eval import train, evaluate
from src.utils import set_global_verbose, Logger, load_graphs

LOGGER_FILE = 'results_general_GK.json'

KERNELS = {
    'WL': WeisfeilerLehman(n_iter=5,
                           normalize=True,
                           base_graph_kernel=VertexHistogram),
    'SP': ShortestPath()
}


def make_hashable_attr(nx_graphs: List[nx.Graph],
                       node_attr: str = 'x') -> None:
    """
    Transform the node attribute `x` to str to be hashable.

    Args:
        nx_graphs:
        node_attr:

    Returns:

    """
    for nx_graph in nx_graphs:
        for idx_node, data_node in nx_graph.nodes(data=True):
            str_data = str(data_node[node_attr])
            nx_graph.nodes[idx_node][node_attr] = str_data


def graph_classifier(root_dataset: str,
                     graph_kernel: str,
                     size_splits: List[float],
                     seed: int,
                     Cs: List[float],
                     n_cores: int,
                     save_predictions: bool,
                     folder_results: str,
                     verbose: bool,
                     args):
    """

    Args:
        root_dataset:
        graph_kernel:
        size_splits:
        seed:
        Cs:
        n_cores:
        save_predictions:
        folder_results:
        verbose:
        args:

    Returns:

    """
    set_global_verbose(verbose)

    pathlib.Path(folder_results).mkdir(parents=True, exist_ok=True)

    # Init logger
    logger_filename = os.path.join(folder_results,
                                   LOGGER_FILE)
    logger = Logger(logger_filename)

    # Save all the input parameters
    logger.data['parameters'] = vars(args)
    logger.save_data()

    nx_graphs, classes = load_graphs(root_dataset,
                                     load_classes=True)
    make_hashable_attr(nx_graphs, node_attr='x')
    grakel_graphs = [graph for graph in graph_from_networkx(nx_graphs,
                                                            node_labels_tag='x',
                                                            as_Graph=True)]

    size_train, size_val, size_test = size_splits
    G_train, G_test, y_train, y_test = train_test_split(grakel_graphs,
                                                        classes,
                                                        test_size=size_test,
                                                        random_state=seed)

    kernel = KERNELS[graph_kernel]
    # kernel.n_jobs = n_cores if n_cores > 0 else None
    acc_tracker = train(logger,
                        kernel,
                        Cs,
                        G_train, y_train,
                        n_cores)

    evaluate(logger,
             acc_tracker,
             kernel,
             G_train, G_test,
             y_train, y_test,
             folder_results,
             save_predictions)
