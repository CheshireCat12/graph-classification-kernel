import os
from glob import glob

import networkx as nx
import numpy as np
from grakel.utils import graph_from_networkx
import json
from grakel.kernels import RandomWalk

import argparse
import csv
import logging
import os
import pathlib
from collections import namedtuple
from itertools import product
from time import time
from typing import List, Tuple

import numpy as np

from src.graph_classification import graph_classifier


def main(args):
    graph_classifier(args.root_dataset,
                     args.graph_kernel,
                     args.size_splits,
                     args.seed,
                     args.C,
                     args.folder_results,
                     args.save_predictions,
                     args.verbose,
                     args)
    # path = './data/NCI1'
    # files = glob(os.path.join(path, '*.graphml'))
    # attr_node = 'x'
    #
    # nx_graphs = [nx.read_graphml(file) for file in files]
    # for nx_graph in nx_graphs:
    #     for idx_node, data_node in nx_graph.nodes(data=True):
    #         np_data = np.fromstring(data_node[attr_node][1:-1], sep=' ')
    #         nx_graph.nodes[idx_node][attr_node] = np_data
    #
    # grakel_graphs = graph_from_networkx(nx_graphs)
    # print(len(nx_graphs))
    # print(nx_graphs[0].nodes(data=True))
    # print(grakel_graphs)
    # pass
    #
    # kernel = RandomWalk().fit_transform(grakel_graphs)
    # print(kernel)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Graph reduction by coarsening')
    subparser = args_parser.add_subparsers()

    args_parser.add_argument('--root_dataset',
                             type=str,
                             required=True,
                             default='./data',
                             help='Root of the dataset')

    # Hyperparameters for the Graph kernel
    args_parser.add_argument('--graph_kernel',
                             default='WL',
                             choices=['WL'],
                             type=str,
                             help='Graph kernel to embed the graphs')

    # Hyperparameters for the SVM
    args_parser.add_argument('--C',
                             nargs='*',
                             default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                             type=float,
                             help='List of alphas to test')

    # Parameters used during the optimization process
    args_parser.add_argument('--size_splits',
                             nargs=3,
                             type=float,
                             default=[0.6, 0.2, 0.2],
                             help='Arguments that set the size of the splits'
                                  '(e.g., --size_split size_train size_val size_test')
    args_parser.add_argument('--seed',
                             default=1,
                             type=int,
                             help='Choose the random seed')

    args_parser.add_argument('--save_predictions',
                             action='store_true',
                             help='save the predicted classes if activated')

    args_parser.add_argument('--folder_results',
                             type=str,
                             required=True,
                             help='Folder where to save the reduced graphs')

    args_parser.add_argument('-v',
                             '--verbose',
                             action='store_true',
                             help='Activate verbose print')

    parse_args = args_parser.parse_args()

    main(parse_args)

#
# def main2():
#     import numpy as np
#
#     from sklearn.svm import SVC
#     from sklearn.model_selection import GridSearchCV
#     from sklearn.model_selection import cross_val_predict
#     from sklearn.pipeline import make_pipeline
#     from sklearn.metrics import accuracy_score
#
#     from grakel.datasets import fetch_dataset
#     from grakel.kernels import ShortestPath
#
#     # Loads the Mutag dataset from:
#     NCI1 = fetch_dataset("NCI1", verbose=False)
#     G, y = NCI1.data, NCI1.target
#
#     # Values of C parameter of SVM
#     C_grid = (10. ** np.arange(-4, 6, 1) / len(G)).tolist()
#
#     # Creates pipeline
#     estimator = make_pipeline(
#         ShortestPath(normalize=True),
#         GridSearchCV(SVC(kernel='precomputed'),
#                      dict(C=C_grid),
#                      scoring='accuracy', cv=2))
#
#     # Performs cross-validation and computes accuracy
#     n_folds = 2
#     acc = accuracy_score(y, cross_val_predict(estimator, G, y, cv=n_folds))
#     print("Accuracy:", str(round(acc * 100, 2)) + "%")
