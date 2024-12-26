from graph import Graph
from colony import Colony

import numpy as np
from numba import njit

import json
from time import time


with open("graph1000.json") as f: matrix = json.load(f)
for i,_ in enumerate(matrix):
    for j,_ in enumerate(matrix[i]):
        if matrix[i][j] == 0:
            matrix[i][j] = np.inf


graph = Graph.build(matrix)

# graph = Graph.build([[np.inf, 3, np.inf, np.inf, 1, 3],
#                      [3, np.inf, 8, np.inf, np.inf, 3],
#                      [np.inf, 8, np.inf, 1, np.inf, 1],
#                      [np.inf, np.inf, 1, np.inf, 1, 5],
#                      [1, np.inf, np.inf, 1, np.inf, 4],
#                      [3, 3, 1, 5, 4, np.inf]])

pheromons = np.ones(graph.matrix.shape)

b = 1
a = 1.5
d = 0.5

ways_history_min = []
ways_history_var = []
min_way_len = np.inf
min_way = []

@njit
def take_step(matrix: np.ndarray, pheromons: np.ndarray, a: int, b: int,
              positions: np.ndarray, ways: np.ndarray, lenghts: np.ndarray):
    # Рассчёт вероятностей перемещения
    probabilities = ((1/matrix[positions] ** b) * pheromons[positions] ** a)
    indeces = np.vstack((np.broadcast_to(np.arange(ways.shape[0]), (ways.shape[1], ways.shape[0])).flatten(), ways.T.flatten()))
    probabilities[indeces] = 0
    probabilities /= probabilities.sum(axis=1).reshape(-1, 1)
        
    # Выбираем куда пойдёт дальше
    next_posses = (np.cumsum(probabilities, axis=1) - np.random.random((probabilities.shape[0], 1)) > 0).argmax(axis=1)

    # Присваиваем следуйщие варианты
    return lenghts + matrix[positions, next_posses], next_posses, np.hstack((ways, next_posses.reshape(-1, 1)))


start = time()
for step in range(1):
    ants = Colony.create(graph.matrix.shape[0])
    for _ in range(graph.matrix.shape[0]-1):
        ants.ways_lenghts, ants.positions, ants.ways = take_step(graph.matrix, pheromons, a, b, 
                                                                 ants.positions, ants.ways, ants.ways_lenghts)

    # Убираем тупиковые варианты
    bad_ways = ants.ways.sum(axis=1) != np.arange(ants.ways.shape[0]).sum()
    ants.ways_lenghts[bad_ways] = np.nan

    print(ants.ways)
    print(ants.ways_lenghts)

    # Обновляем пути
    way_i = np.nanargmin(ants.ways_lenghts)
    if min_way_len > ants.ways_lenghts[way_i]:
        min_way_len, min_way = ants.ways_lenghts[way_i], ants.ways[way_i]
    
    ways_history_min.append(ants.ways_lenghts.min())
    ways_history_var.append(ants.ways_lenghts.var())

    # Обновляем фиромоны
    pheromons *= d
    pheromons[min_way[:-1], min_way[1:]] += 1/graph.matrix[min_way[:-1], min_way[1:]]

print("For ", time() - start)