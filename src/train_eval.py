import csv
import logging
import os
from collections import namedtuple
from time import time
from typing import List

import numpy as np
from grakel import GraphKernel
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from src.utils import Logger

AccuracyTracker = namedtuple('AccuracyTracker',
                             ['acc', 'best_c'])


def train(logger: Logger,
          kernel: GraphKernel,
          Cs: List[float],
          G_train: List,
          y_train: List[int]) -> AccuracyTracker:
    """

    Args:
        logger:
        kernel:
        Cs:
        G_train:
        y_train:

    Returns:

    """

    svc = SVC(kernel='precomputed')

    K_train = kernel.fit_transform(G_train)

    clf = GridSearchCV(svc, {'C': Cs})
    clf.fit(K_train, y_train)

    acc_tracker = AccuracyTracker(clf.best_score_,
                                  clf.best_params_['C'])

    # Save all the hyperparameters tested
    logger.data['hyperparameters_tuning'] = {'Cs': Cs}
    # logger.data['val_accuracies'] = []  # List of tuple (acc, alpha, k)
    # logger.data['val_prediction_times'] = []

    logging.info(f'Best val classification accuracy {100 * acc_tracker.acc: .2f}% '
                 f'(C: {acc_tracker.best_c})')

    logger.data['best_acc'] = acc_tracker.acc
    logger.data['best_params'] = clf.best_params_
    logger.save_data()

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


def evaluate(logger: Logger,
             acc_tracker: AccuracyTracker,
             kernel: GraphKernel,
             G_train: List,
             G_test: List,
             y_train: List[int],
             y_test: List[int],
             folder_results: str,
             save_predictions: bool) -> None:
    """

    Args:
        logger:
        acc_tracker:
        kernel:
        G_train:
        G_test:
        y_train:
        y_test:
        folder_results:
        save_predictions:

    Returns:

    """

    # Set SVM with the best parameter c
    clf = SVC(kernel='precomputed',
              C=acc_tracker.best_c)

    start_time = time()

    # Embed graph with the kernel
    K_train = kernel.fit_transform(G_train)
    K_test = kernel.transform(G_test)

    # Perform prediction
    clf.fit(K_train, y_train)
    predictions = clf.predict(K_test)
    prediction_time = time() - start_time

    # compute accuracy
    current_acc = 100 * ((np.array(predictions) == y_test).sum() / len(y_test))

    logger.data['test_accuracy'] = (current_acc, acc_tracker.best_c)
    logger.data['test_prediction_time'] = prediction_time
    logger.save_data()

    logging.info(f'Classification accuracy (test) {current_acc: .2f} '
                 f'(alpha: {acc_tracker.best_c})')

    if save_predictions:
        file_predictions = os.path.join(folder_results,
                                        'predictions.csv')
        write_predictions(file_predictions, predictions, y_test)
