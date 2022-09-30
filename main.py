import argparse

import numpy as np

from src.graph_classification import graph_classifier


def main(args):
    graph_classifier(args.root_dataset,
                     args.graph_kernel,
                     args.size_splits,
                     args.seed,
                     args.Cs,
                     args.n_cores,
                     args.save_gt_labels,
                     args.save_predictions,
                     args.folder_results,
                     args.verbose,
                     args)


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
                             choices=['WL', 'SP'],
                             type=str,
                             help='Graph kernel to embed the graphs')

    # Hyperparameters for the SVM
    args_parser.add_argument('--Cs',
                             nargs='*',
                             default=(10. ** np.arange(-2, 2, .5)).tolist(),
                             type=float,
                             help='List of Cs to test')

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

    args_parser.add_argument('--n_cores',
                             default=1,
                             type=int,
                             help='Set the number of cores to use.')

    args_parser.add_argument('--save_gt_labels',
                             action='store_true',
                             help='save the ground truth classes if activated')
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
