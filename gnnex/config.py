# -*- coding: utf-8 -*-
# @Time    : 2019-11-18 10:33
# @Author  : Jason
# @FileName: config.py


class Config(object):

    def __init__(self, epochs, patience, base_model_path, hidden_dim, dataset, weights_path, adj_path, graph_path,
                 heatmap_path, support, batch_size):
        self.epochs = epochs
        self.patience = patience
        self.base_model_path = base_model_path
        self.hidden_dim = hidden_dim
        self.dataset = dataset
        self.weights_path = weights_path
        self.adj_path = adj_path
        self.graph_path = graph_path
        self.heatmap_path = heatmap_path
        self.support = support
        self.batch_size = batch_size
