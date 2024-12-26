# import numpy as np
# import json

# verticles_number = 100
# graph = np.random.randint(1, 30, size=(verticles_number, verticles_number))
# np.fill_diagonal(graph, 0)
# print(graph)
# print(graph.shape)
# print(np.unique(graph == 0, return_counts=True))

# with open("graph100.json", 'w') as f:
#     json.dump(graph.tolist(), f, indent=4)

# import jax.random as jrnd
# import jax.numpy as jnp

# key = jrnd.key(42)

# print(jnp.unique_counts(jrnd.uniform(key, shape=(1000, 1)) > 1))

# import numpy as np

# matrix =np.array([[0, 4, 5, 2, 3, 0],
#  [1, 5, 2, 3, 4, 0],
#  [2, 1, 0, 4, 3, 5],
#  [3, 2, 5, 4, 0, 1],
#  [4, 3, 2, 5, 0, 1],
#  [5, 2, 3, 4, 0, 1]])

# print(matrix[matrix.sum(axis=1) == np.arange(matrix.shape[0]).sum()])

import jax.numpy as jnp
import jax.random as jrnd
from time import time
import jax


# key = jrnd.key(42)

# @jax.jit
# def f(x, y):
#     return jnp.append(x, y, axis=1)

# for _ in range(10):
#     x = jrnd.uniform(key=key, shape=(10000000, 1))
#     y = jrnd.uniform(key=key, shape=(10000000, 1))
#     start = time()
#     z = f(x, y)
#     print(time() - start)

# print(jnp.round(jnp.nan))

some_matrix = jnp.array([[0, 0, 0, 0],
                         [0,0 ,0, 0]])

print(some_matrix.shape)