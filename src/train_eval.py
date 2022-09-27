from typing import List
from src.utils import Logger
import networkx as nx
from collections import namedtuple

from grakel import GraphKernel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


AccuracyTracker = namedtuple('AccuracyTracker',
                             ['acc', 'best_c'])


def train(logger: Logger,
          kernel: GraphKernel,
          Cs: List[float],
          G_train: List[nx.Graph],
          G_val: List[nx.Graph],
          y_train: List[int],
          y_val: List[int],
          n_cores: int) -> AccuracyTracker:
    """

    Args:
        logger:
        kernel:
        Cs:
        G_train:
        G_val:
        y_train:
        y_val:
        n_cores:

    Returns:

    """

    K_train = kernel.fit_transform(G_train)
    K_val = kernel.transform(G_val)

    clf = SVC(kernel='precomputed')
    clf.fit(K_train, y_train)

    y_pred = clf.predict(K_val)

    print(f'Accuracy: {accuracy_score(y_val, y_pred)}')

    # knn = KNNClassifier(coordinator.ged, parallel=is_parallel)
    # knn.train(X_train, y_train)
    #
    # acc_tracker = AccuracyTracker(float('-inf'), None, None)
    #
    # # Save all the hyperparameters tested
    # logger.data['hyperparameters_tuning'] = {'alphas': alphas,
    #                                          'ks': ks}
    # logger.data['val_accuracies'] = []  # List of tuple (acc, alpha, k)
    # logger.data['val_prediction_times'] = []
    #
    # for alpha, k in tqdm(product(alphas, ks), total=len(alphas) * len(ks)):
    #     # Update alpha parameter
    #     coordinator.edit_cost.update_alpha(alpha)
    #
    #     # Perform prediction
    #     start_time = time()
    #     predictions = knn.predict(X_val, k=k, num_cores=n_cores)
    #     prediction_time = time() - start_time
    #
    #     # compute accuracy
    #     current_acc = 100 * ((np.array(predictions) == y_val).sum() / len(y_val))
    #
    #     # Keep track of the best acc with the corresponding hyperparameters
    #     if current_acc > acc_tracker.acc:
    #         acc_tracker = acc_tracker._replace(acc=current_acc, best_alpha=alpha, best_k=k)
    #
    #     logger.data['val_accuracies'].append((current_acc, alpha, k))
    #     logger.data['val_prediction_times'].append(prediction_time)
    #     logger.save_data()
    #
    # logging.info(f'Best val classification accuracy {acc_tracker.acc: .2f}'
    #              f'(alpha: {acc_tracker.best_alpha}, k: {acc_tracker.best_k})')
    #
    # logger.data['best_acc'] = acc_tracker.acc
    # logger.data['best_params'] = {'alpha': acc_tracker.best_alpha,
    #                               'k': acc_tracker.best_k}
    # logger.save_data()

    return acc_tracker


def write_predictions(filename: str,
                      predictions: List[int],
                      GT_labels: List[int]) -> None:
    """
    Write the predictions and the corresponding GT labels in `filename`.

    Args:
        filename: File where to save the predictions.
        predictions: Iterable of predictions
        GT_labels: Iterable of the GT labels

    Returns:

    """

    with open(filename, 'w') as csv_file:
        fieldnames = ['predictions', 'GT_labels']

        writer = csv.DictWriter(csv_file,
                                fieldnames=fieldnames)

        writer.writeheader()
        for pred, GT_lbl in zip(predictions, GT_labels):
            writer.writerow({'predictions': pred, 'GT_labels': GT_lbl})


def write_distances(filename: str, distances: np.ndarray) -> None:
    """
    Save the GEDs in `.npy` file

    Args:
        filename: File where to save the GEDs.
        distances: `np.array` containing the GEDs

    Returns:

    """
    with open(filename, 'wb') as file:
        np.save(file, distances)


def evaluate(coordinator: Coordinator,
             logger: Logger,
             acc_tracker: AccuracyTracker,
             X_train: List[Graph],
             X_test: List[Graph],
             y_train: List[int],
             y_test: List[int],
             n_cores: int,
             folder_results: str,
             save_predictions: bool,
             save_distances: bool) -> None:
    """

    Args:
        coordinator:
        logger:
        acc_tracker:
        X_train:
        X_test:
        y_train:
        y_test:
        n_cores:
        folder_results:
        save_distances:
        save_predictions:

    Returns:

    """
    is_parallel = n_cores > 0

    knn = KNNClassifier(coordinator.ged, parallel=is_parallel)
    knn.train(X_train, y_train)

    alpha, k = acc_tracker.best_alpha, acc_tracker.best_k

    # Set the best alpha parameter
    coordinator.edit_cost.update_alpha(alpha)

    # Perform prediction
    start_time = time()
    predictions = knn.predict(X_test, k=k, num_cores=n_cores)
    prediction_time = time() - start_time

    # compute accuracy
    current_acc = 100 * ((np.array(predictions) == y_test).sum() / len(y_test))

    logger.data['test_accuracy'] = (current_acc, alpha, k)
    logger.data['test_prediction_time'] = prediction_time
    logger.save_data()

    logging.info(f'Classification accuracy (test) {current_acc: .2f}'
                 f'(alpha: {acc_tracker.best_alpha}, k: {acc_tracker.best_k})')

    if save_predictions:
        file_predictions = os.path.join(folder_results,
                                        'predictions.csv')
        write_predictions(file_predictions, predictions, y_test)

    if save_distances:
        file_distances = os.path.join(folder_results,
                                      'distances.npy')
        write_distances(file_distances, knn.current_distances)
