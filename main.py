import os
from glob import glob

import networkx as nx
import numpy as np
from grakel.utils import graph_from_networkx
import json
from grakel.kernels import RandomWalk

def main():
    path = './data/NCI1'
    files = glob(os.path.join(path, '*.graphml'))
    attr_node = 'x'

    nx_graphs = [nx.read_graphml(file) for file in files]
    for nx_graph in nx_graphs:
        for idx_node, data_node in nx_graph.nodes(data=True):
            np_data = np.fromstring(data_node[attr_node][1:-1], sep=' ')
            nx_graph.nodes[idx_node][attr_node] = np_data

    grakel_graphs = graph_from_networkx(nx_graphs)
    print(len(nx_graphs))
    print(nx_graphs[0].nodes(data=True))
    print(grakel_graphs)
    pass

    kernel = RandomWalk().fit_transform(grakel_graphs)
    print(kernel)

def main2():
    import numpy as np

    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score

    from grakel.datasets import fetch_dataset
    from grakel.kernels import ShortestPath

    # Loads the Mutag dataset from:
    NCI1 = fetch_dataset("NCI1", verbose=False)
    G, y = NCI1.data, NCI1.target

    # Values of C parameter of SVM
    C_grid = (10. ** np.arange(-4, 6, 1) / len(G)).tolist()

    # Creates pipeline
    estimator = make_pipeline(
        ShortestPath(normalize=True),
        GridSearchCV(SVC(kernel='precomputed'), dict(C=C_grid),
                     scoring='accuracy', cv=2))

    # Performs cross-validation and computes accuracy
    n_folds = 2
    acc = accuracy_score(y, cross_val_predict(estimator, G, y, cv=n_folds))
    print("Accuracy:", str(round(acc * 100, 2)) + "%")

if __name__ == '__main__':
    main()
    main2()