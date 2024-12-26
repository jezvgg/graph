# import jax.numpy as jnp
import numpy as jnp


class Colony:
    positions: jnp.ndarray
    ways: jnp.ndarray
    ways_lenghts: jnp.ndarray


    @staticmethod
    def create(n: int):
        return Colony(jnp.arange(n))
    

    def __init__(self, positions: jnp.ndarray):
        self.positions = positions
        self.ways = jnp.array([[position] for position in positions])
        self.ways_lenghts = jnp.array([0.] * len(positions))
        
