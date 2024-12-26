from graph import Graph
from colony import Colony

import numpy as np
import matplotlib.pyplot as plt

import json


# with open("graph1000.json") as f: matrix = json.load(f)
# for i,_ in enumerate(matrix):
#     for j,_ in enumerate(matrix[i]):
#         if matrix[i][j] == 0:
#             matrix[i][j] = np.inf


# graph = Graph.build(matrix)

graph = Graph.build([[np.inf, 3, np.inf, np.inf, 1, 3],
                     [3, np.inf, 8, np.inf, np.inf, 3],
                     [np.inf, 8, np.inf, 1, np.inf, 1],
                     [np.inf, np.inf, 1, np.inf, 1, 5],
                     [1, np.inf, np.inf, 1, np.inf, 4],
                     [3, 3, 1, 5, 4, np.inf]])

pheromons = np.ones(graph.matrix.shape)

b = 1
a = 1.5
d = 0.5

ways_history_min = []
ways_history_var = []
min_way_len = np.inf
min_way = []


for step in range(1):
    ants = Colony.create(graph.matrix.shape[0])
    for _ in range(graph.matrix.shape[0]-1):
        # Рассчёт вероятностей перемещения
        probabilities = ((1/graph.matrix[ants.positions] ** b) * pheromons[ants.positions] ** a)
        # print((np.broadcast_to(np.arange(ants.ways.shape[0]), (ants.ways.shape[1], ants.ways.shape[0])).flatten(), ants.ways.T.flatten()))
        # indeces = np.hstack((np.broadcast_to(np.arange(ants.ways.shape[0]), (ants.ways.shape[1], ants.ways.shape[0])).flatten().reshape(-1, 1), ants.ways.T.flatten().reshape(-1, 1)))
        print(ants.ways)
        probabilities[:, ants.ways] = 0
        print(probabilities)
        probabilities /= probabilities.sum(axis=1).reshape(-1, 1)
        
        # Выбираем куда пойдёт дальше
        next_posses = (np.cumsum(probabilities, axis=1) - np.random.random((probabilities.shape[0], 1)) > 0).argmax(axis=1)

        # Присваиваем следуйщие варианты
        ants.ways_lenghts += graph.matrix[ants.positions, next_posses].astype(int)
        ants.positions = next_posses
        ants.ways = np.hstack((ants.ways, next_posses.reshape(-1, 1)))

    # Убираем тупиковые варианты
    bad_ways = ants.ways.sum(axis=1) != np.arange(ants.ways.shape[0]).sum()
    ants.ways_lenghts[bad_ways] = np.inf

    print("step: ", step+1)
    print(ants.positions)
    print(ants.ways)
    print(ants.ways_lenghts)

    # Обновляем пути
    way_i = np.argmin(ants.ways_lenghts)
    if min_way_len > ants.ways_lenghts[way_i]:
        min_way_len, min_way = ants.ways_lenghts[way_i], ants.ways[way_i]
    
    ways_history_min.append(ants.ways_lenghts.min())
    ways_history_var.append(ants.ways_lenghts.var())

    # Обновляем фиромоны
    pheromons *= d
    pheromons[min_way[:-1], min_way[1:]] += 1/graph.matrix[min_way[:-1], min_way[1:]]


plt.plot(ways_history_min, lw=3, label="Min way")
w=10
density = np.array(ways_history_var) + np.array(ways_history_min)
# plt.plot(np.arange(w, len(ways_history_min)+1), np.convolve(density, np.ones(w), 'valid') / w, lw=3, label="MA Variability")
plt.fill_between(np.arange(len(ways_history_min)), ways_history_min, density, color='r', alpha=.3, label='Variability')
plt.legend()
# plt.show()
        