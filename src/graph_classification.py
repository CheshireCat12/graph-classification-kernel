import os
import pathlib
from typing import List, Tuple
from src.utils import set_global_verbose, train_val_test_split, Logger, load_graphs
from grakel.kernels import WeisfeilerLehman, VertexHistogram

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

    graphs, classes = load_graphs(root_dataset,
                                  load_classes=True)
    size_train, size_val, size_test = size_splits
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(graphs,
                                                                          classes,
                                                                          val_size=size_val,
                                                                          test_size=size_test,
                                                                          random_state=seed)

    wl_kernel = WeisfeilerLehman(n_iter=5,
                                 normalize=True,
                                 base_graph_kernel=VertexHistogram)

    pass