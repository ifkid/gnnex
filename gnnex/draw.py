# -*- coding: utf-8 -*-
# @Time    : 2019-10-22 17:58
# @Author  : Jason
# @FileName: draw.py

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from gnnex.utils import calWeight, int2str, randomList, subMatrix


def showGraph(config, matrix, node, threshold):
    G = nx.Graph()
    nodeN = len(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            G.add_edge(i, j, weight=matrix[i][j])

    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if
              0.9 <= calWeight(d['weight'], threshold) <= 1]  # Useful edges
    emid = [(u, v) for (u, v, d) in G.edges(data=True) if
            0.4 <= calWeight(d['weight'], threshold) <= 0.6]  # Unless edges
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if
              calWeight(d['weight'], threshold) <= threshold]  # Unless edges

    elarge = randomList(elarge, 50)
    emid = randomList(emid, 30)
    esmall = randomList(esmall, 20)

    large_color = "#800026"
    mid_color = "#FD9E43"
    small_color = "#1E90FF"
    node_color = "#3CB371"
    #
    if nodeN > 50:
        if nodeN > 100:
            a = 0.5
        else:
            a = 0.7
    else:
        a = 0.8
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=10)
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=1, edge_color=large_color)
    nx.draw_networkx_edges(G, pos, edgelist=emid, width=1, edge_color=mid_color)
    nx.draw_networkx_edges(G, pos, edgelist=esmall, width=1, alpha=a, edge_color=small_color)
    plt.axis("off")
    plt.savefig(config.graph_path + "{}_graph_node{}_{}".format(config.dataset, node, int2str(threshold)),
                dpi=1000)  # save as png
    plt.close()


def showGraphHeat(config, matrix, node, threshold):
    heat_f = plt.figure(dpi=800, frameon=False)
    sns.heatmap(matrix, cmap="YlOrRd")
    heat_f.savefig(config.heatmap_path + "{}_heat_node{}_{}".format(config.dataset, node, int2str(threshold)))
    plt.close()


def draw(config, nodes, threshold):
    print("========= Start to draw ========== ")
    for node in nodes:
        matrix, adj = subMatrix(config.adj_path, config.weights_path, node, threshold)
        showGraph(config, matrix, node, threshold)
        showGraphHeat(config, matrix, node, threshold)
