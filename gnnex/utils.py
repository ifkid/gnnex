# -*- coding: utf-8 -*-
# @Time    : 2019-10-15 14:32
# @Author  : Jason
# @FileName: utils.py


import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
import random
import matplotlib.pyplot as plt


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora dataset only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}_small.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return features, adj, labels


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_splits(y):
    num = [i for i in range(y.shape[0])]
    random.shuffle(num)
    idx_train = num[:1500]
    idx_val = num[1500:2200]
    idx_test = num[2200:2700]
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    # print("idx_train: ", idx_train)
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


def categorical_crossentropy(preds, labels):
    temp = np.extract(labels, preds)
    epsilon = 1e-5
    return np.mean(-np.log(temp + epsilon))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):
    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k + 1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def pltFig(n, x1, x2, x1_label, x2_label, xlabel, ylabel, title, path, dpi):
    fig = plt.figure()
    plt.plot(range(1, n + 1), x1, label=x1_label)
    plt.plot(range(1, n + 1), x2, label=x2_label)
    plt.legend(loc="best")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig.savefig(path, dpi=dpi)
    plt.show()


def pltFigSingle(n, x,x_label, xlabel, ylabel, title, path, dpi):
    fig = plt.figure()
    plt.plot(range(1, n + 1), x,label=x_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig.savefig(path, dpi=dpi)
    plt.show()


def getNeighbor(node, adj):
    neighbors = []
    for i in range(len(adj[node])):
        if adj[node][i] == 1 and i != node:
            neighbors.append(i)
    return neighbors


def calWeight(w, threshold):
    if w > threshold:
        if w >= 1:
            weight = 1
        else:
            weight = w
    else:
        weight = abs(0.5 - (1 / (1 + np.exp(-w))))
    return weight


def int2str(n):
    s = str(n).split(".")
    return s[0] + s[-1]


def randomList(edgeList, percent):
    l = []
    for _ in range(min(len(edgeList), percent)):
        l.append(edgeList[min(len(edgeList) - 1, int(random.random() * len(edgeList)))])
    return l


def subMatrix(ADJ_MATRIX_PATH, WEIGHT_PATH, node, threshold):
    adj = np.loadtxt(ADJ_MATRIX_PATH)
    weights = np.loadtxt(WEIGHT_PATH)
    dic = {}
    neighbors = getNeighbor(node, adj)
    dic[node] = neighbors
    for n in neighbors:
        dic[n] = getNeighbor(n, adj)
    all_nodes = []
    for v in dic.values():
        all_nodes += v
    all_nodes = list(set(all_nodes))
    n = len(all_nodes)
    matrix = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(n):
            if weights[all_nodes[i]][all_nodes[j]] > threshold:
                if weights[all_nodes[i]][all_nodes[j]] >= 1.0:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = weights[all_nodes[i]][all_nodes[j]]
            else:
                matrix[i][j] = abs(0.5 - (1 / (1 + np.exp(-weights[all_nodes[i]][all_nodes[j]]))))
    return matrix, adj