from functools import partial

import jax.numpy as jnp
import jax


# задаём значения, так чтоб суммы не совпадали
PLAYER = 1
AI = 5

board = jnp.array([[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [AI, AI, 0, 0, 0, 0, 0],
                   [AI, AI, AI, 0, 0, 0, 0]])


score_player = jnp.array([0, 0, 2, 4, jnp.inf, 0] + [0]*9 + [-4] + [0]*5)
score_ai = jnp.array([0, 0, 0, -4, 0, 0] + [0]*4 + [2] + [0]*4 + [5] + [0]*4 + [jnp.inf])


@partial(jax.jit, static_argnames=['piece'])
def score_window(window: jnp.ndarray, piece: int):
    piece_score = score_ai
    if piece == PLAYER: piece_score = score_player

    return piece_score[jnp.sum(window)]


@partial(jax.jit, static_argnames=['piece'])
def get_score(board: jnp.ndarray, piece: int):
    score = 0
    
    # Горизонтальные окна
    for y_j in range(board.shape[1]-3):
        score += jnp.sum(jax.vmap(score_window, [1, None], 0)(board[:, y_j:y_j+4], piece))

    # Вертикальные окна
    for x_i in range(board.shape[0]-3):
         score += jnp.sum(jax.vmap(score_window, [0, None], 0)(board[x_i:x_i+4, :], piece))

    # Диагональные окна
    for x_i in range(board.shape[0]-3):
        for y_j in range(board.shape[1]-3):
            score += score_window(board[jnp.arange(x_i, x_i+4), jnp.arange(y_j,y_j+4)], piece)

    return score




print(get_score(board, AI))