from graph import Graph
import numpy as np

graph = Graph.build([[0, 10, 30, 50, 10],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 10],
                     [0, 40, 20, 0, 0],
                     [10, 0, 10, 30, 0]])

weights = np.zeros(len(graph.verticles))
verticles_weights = weights.copy()
verticles_weights[1:] = np.inf

while verticles_weights[np.isnan(verticles_weights) == False].shape[0] != 0:
    current_node = np.nanargmin(verticles_weights)
    new_weights = graph.matrix[current_node] + verticles_weights[current_node]
    logical = np.logical_and(verticles_weights > new_weights, new_weights != verticles_weights[current_node])
    verticles_weights[logical] = new_weights[logical]
    weights[current_node], verticles_weights[current_node] = verticles_weights[current_node], np.nan

print(weights)
