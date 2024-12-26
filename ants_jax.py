import jax.extend
from graph import Graph
from colony_jax import Colony

import jax.numpy as jnp
import jax.random as jrnd
from jax.lib import xla_bridge
import jax

# jax.config.update('jax_platform_name', 'cpu')
print(jax.extend.backend.get_backend().platform)
device = jax.devices('gpu')[0]

from time import time
from functools import partial
import json
import gc


# with open("graph1000.json") as f: matrix = json.load(f)
# for i,_ in enumerate(matrix):
#     for j,_ in enumerate(matrix[i]):
#         if matrix[i][j] == 0:
#             matrix[i][j] = jnp.inf


# graph = Graph.build(matrix)

graph = Graph.build([[jnp.inf, 3, jnp.inf, jnp.inf, 1, jnp.inf],
                     [3, jnp.inf, 8, jnp.inf, jnp.inf, 3],
                     [jnp.inf, 3, jnp.inf, 1, jnp.inf, 1],
                     [jnp.inf, jnp.inf, 1, jnp.inf, 1, jnp.inf],
                     [3, jnp.inf, jnp.inf, 3, jnp.inf, jnp.inf],
                     [3, 3, 3, 5, 4, jnp.inf]])

pheromons = jnp.ones(graph.matrix.shape)
graph_matrix = jnp.array(graph.matrix)
n_vetricles = graph.matrix.shape[0]


min_way_len = jnp.inf
min_way = tuple()
key = jrnd.key(42)


b = 1
a = 1.5
d = 0.5


# @partial(jax.jit, static_argnames=['a', 'b'], backend='gpu')
# @jax.jit
def take_epoch(matrix: jax.Array, pheromons: jax.Array, random: jax.Array,
                a: int, b: int, key: jax.Array, colony: Colony) -> Colony:
    print(ants)
    # Рассчёт вероятностей перемещения
    probabilities = ((1/matrix[colony.positions] ** b) * pheromons[colony.positions] ** a)
    indeces1 = jnp.broadcast_to(jnp.arange(colony.ways.shape[0]), shape=colony.ways.shape[::-1]).flatten() 
    indeces2 = colony.ways.T.flatten()
    print(indeces1, indeces2)
    probabilities = probabilities.at[indeces1, indeces2].set(0)
    probabilities /= probabilities.sum(axis=1).reshape(-1, 1)

    # Выбираем куда пойдёт дальше
    next_posses = (jnp.cumsum(probabilities, axis=1) - random > 0).argmax(axis=1)
    next_ways = jnp.empty((colony.ways.shape[0], colony.ways.shape[1]+1))
    next_ways = next_ways.at[:, :-1].set(colony.ways)
    next_ways = next_ways.at[:, -1].set(next_posses)

    # Присваиваем следуйщие варианты
    return Colony(next_posses, next_ways, 
                  colony.ways_lenghts + matrix[colony.positions, next_posses] * jnp.round(probabilities.sum(axis=1)))


@partial(jax.jit, static_argnames=['d'])
def update_pheromons(pheromons: jnp.ndarray, d: int, min_way: jnp.ndarray, matrix: jnp.ndarray):
    pheromons *= d
    return pheromons.at[min_way[:-1], min_way[1:]].set(pheromons[min_way[:-1], min_way[1:]] + 1/matrix[min_way[:-1], min_way[1:]])


for step in range(1):
    ants = Colony(jnp.arange(n_vetricles), jnp.arange(n_vetricles).reshape(-1, 1), jnp.zeros(n_vetricles))

    for _ in range(n_vetricles-1):
        start = time()
        random = jrnd.uniform(key, shape=(n_vetricles, 1))
        ants = take_epoch(graph_matrix, pheromons, random, a, b, key, ants)
        print(time() - start)
        print(ants)
        gc.collect()

    print(ants)

    # Обновляем пути
    way_i = jnp.nanargmin(ants.ways_lenghts)
    if ants.ways_lenghts[way_i] < min_way_len:
        min_way_len, min_way = ants.ways_lenghts[way_i], tuple(ants.ways[way_i])

    # Обновляем фиромоны
    pheromons = update_pheromons(pheromons, d, min_way, graph_matrix)

    print(f"Step {step} for {time() - start}")
    print(min_way_len, min_way)
