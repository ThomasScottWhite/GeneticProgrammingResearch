import random
from data_structures import Variable, Node

import matplotlib.pyplot as plt
import networkx as nx
import random


def add_edges(graph, node, pos, x=0, y=0, layer=1):
    if node is not None:
        if callable(node.value):
            string_value = node.value.__name__
        elif isinstance(node.value, Variable):
            string_value = node.value.label
        else:
            string_value = str(node.value)

        graph.add_node(node, pos=(x, y), label=string_value)
        
        dx = 1 / (2 ** layer)
        if node.left:
            graph.add_edge(node, node.left)
            add_edges(graph, node.left, pos, x - dx, y - 1, layer + 1)
        if node.right:
            graph.add_edge(node, node.right)
            add_edges(graph, node.right, pos, x + dx, y - 1, layer + 1)

def plot_tree(root):
    graph = nx.DiGraph()
    pos = {}
    add_edges(graph, root, pos)
    
    pos = nx.get_node_attributes(graph, 'pos')
    labels = nx.get_node_attributes(graph, 'label')
    
    plt.figure(figsize=(10, 8))
    nx.draw(graph, pos, with_labels=False, arrows=False, node_size=1000)
    nx.draw_networkx_labels(graph, pos, labels)
    plt.show()