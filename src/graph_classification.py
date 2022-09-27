import os
import pathlib
from typing import List

import numpy as np
from grakel.kernels import WeisfeilerLehman, VertexHistogram, ShortestPath
from grakel.utils import graph_from_networkx
from sklearn.model_selection import train_test_split

from src.train_eval import train, evaluate
from src.utils import set_global_verbose, Logger, load_graphs

LOGGER_FILE = 'results_general_GK'


def graph_classifier(root_dataset: str,
                     graph_kernel: str,
                     size_splits: List[float],
                     seed: int,
                     C: float,
                     folder_results: str,
                     save_predictions: bool,
                     verbose: bool,
                     args):
    """

    Args:
        root_dataset:
        graph_kernel:
        size_splits:
        seed:
        C:
        folder_results:
        save_predictions:
        verbose:
        args:

    Returns:

    """
    set_global_verbose(verbose)

    pathlib.Path(folder_results).mkdir(parents=True, exist_ok=True)

    # Init logger
    logger_filename = os.path.join(folder_results,
                                   'results_general.json')
    logger = Logger(logger_filename)

    # Save all the input parameters
    logger.data['parameters'] = vars(args)
    logger.save_data()

    nx_graphs, classes = load_graphs(root_dataset,
                                     load_classes=True)

    node_attr = 'x'
    for nx_graph in nx_graphs:
        for idx_node, data_node in nx_graph.nodes(data=True):
            str_data = str(data_node[node_attr])
            nx_graph.nodes[idx_node][node_attr] = str_data

    grakel_graphs = [graph for graph in graph_from_networkx(nx_graphs,
                                                            node_labels_tag='x',
                                                            as_Graph=True)]

    size_train, size_val, size_test = size_splits
    G_train, G_test, y_train, y_test = train_test_split(grakel_graphs,
                                                        classes,
                                                        test_size=size_test,
                                                        random_state=seed)
    # G_train, G_val, G_test, y_train, y_val, y_test = train_val_test_split(grakel_graphs,
    #                                                                       classes,
    #                                                                       val_size=size_val,
    #                                                                       test_size=size_test,
    #                                                                       random_state=seed)
    # Ks = []
    # for i in range(2, 7):
    #     gk = WeisfeilerLehman(n_iter=i,
    #                           base_graph_kernel=VertexHistogram)
    #     K = gk.fit_transform(grakel_graphs)
    #     Ks.append(K)
    #
    # out = cross_validate_Kfold_SVM([Ks], classes, n_iter=1)
    # print(out)
    Cs = (10. ** np.arange(-1, 6, 0.5) / len(G_train)).tolist()

    KERNELS = {
        'WL': WeisfeilerLehman(n_iter=5,
                               normalize=True,
                               base_graph_kernel=VertexHistogram),
        'SP': ShortestPath()
    }

    kernel = KERNELS[graph_kernel]
    acc_tracker = train(logger,
                        kernel,
                        Cs,
                        G_train, y_train)

    evaluate(logger,
             acc_tracker,
             kernel,
             G_train, G_test,
             y_train, y_test,
             folder_results,
             save_predictions)
