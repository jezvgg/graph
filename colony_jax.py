# import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jax import Array
from dataclasses import dataclass
from functools import partial


@partial(register_dataclass,
         data_fields=['positions', 'ways', 'ways_lenghts'],
         meta_fields=[])
@dataclass
class Colony:
    positions: Array
    ways: Array
    ways_lenghts: Array