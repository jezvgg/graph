import time
import random

import numpy as np


EMPTY = 0
HUMAN = 1
AI = 2
DEPTH = 5


def create_board():
    return np.zeros((6, 7), np.int8)


def is_valid_column(board, column):
    '''
    Проверка на корректную колонку (и не заполнена)
    '''
    return column in range(1, 8) and board[0][column - 1] == EMPTY


def valid_locations(board):
    '''
    Вернёт корректные ходы
    '''
    valid_locations = []
    for i in range(1,8):
       if is_valid_column(board, i):
           valid_locations.append(i)
    return valid_locations


def place_piece(board, player, column):
    '''
    Сделать ход
    '''
    index = column - 1
    for row in reversed(range(6)):
        if board[row][index] == EMPTY:
            board[row][index] = player
            return


def clone_and_place_piece(board, player, column):
    new_board = board.copy()
    place_piece(new_board, player, column)
    return new_board


def detect_win(board: np.ndarray, player: int):
    '''
    Проверка на победу
    '''

    # Горизонтальная победа
    for col in range(board.shape[1] - 3):
        for row in range(board.shape[0]):
            if board[row][col] == player and board[row][col+1] == player and \
                    board[row][col+2] == player and board[row][col+3] == player:
                return True
            
    # Вертикальная победа
    for col in range(board.shape[1]):
        for row in range(board.shape[0] - 3):
            if board[row][col] == player and board[row+1][col] == player and \
                    board[row+2][col] == player and board[row+3][col] == player:
                return True
            
    for col in range(board.shape[1] - 3):
        for row in range(board.shape[0] - 3):
            if board[row][col] == player and board[row+1][col+1] == player and \
                    board[row+2][col+2] == player and board[row+3][col+3] == player:
                return True
            
    for col in range(board.shape[1] - 3):
        for row in range(3, board.shape[0]):
            if board[row][col] == player and board[row-1][col+1] == player and \
                    board[row-2][col+2] == player and board[row-3][col+3] == player:
                return True
    return False
    

def is_terminal_board(board):
    '''
    Конечная доска
    '''
    return detect_win(board, HUMAN) or detect_win(board, AI) or \
        len(valid_locations(board)) == 0
        

def score(board: np.ndarray, player: int):
    '''
    Получить оценку позиции
    '''
    score = 0

    # Больше веса в центр
    for col in range(2, 5):
        for row in range(6):
            if board[row][col] == player:
                if col == 3:
                    score += 3
                else:
                    score+= 2

    # Горизонтальные линии
    for col in range(board.shape[1] - 3):
        for row in range(board.shape[0]):
            piece = board[row, col:col+3]
            score += get_score(piece, player)

    # Векртикальные линии
    for col in range(board.shape[1]):
        for row in range(board.shape[0] - 3):
            piece = board[row:row+3, col]
            score += get_score(piece, player)

    # Диагональные линии вверх
    for col in range(board.shape[1] - 3):
        for row in range(board.shape[0] - 3):
            piece = board[np.arange(row, row+3), np.arange(col, col+3)]
            score += get_score(piece, player)

    # Диагональные линии вниз
    for col in range(board.shape[1] - 3):
        for row in range(3, board.shape[0]):
            piece = board[np.arange(row-3, row), np.arange(col, col+3)]
            score += get_score(piece, player)
    return score


def get_score(piece: np.ndarray, player: int):
    '''
    Полчить оценку линии
    '''
    score = 0
    piece = piece.tolist()
    
    if piece.count(player) == 4: score += 99999
    elif piece.count(player) == 3 and piece.count(EMPTY) == 1: score += 100
    elif piece.count(player) == 2 and piece.count(EMPTY) == 2: score += 10
    if score > 9999: print(score)
    if score > 99: print(score)
    return score


def minimax(board, depth, alpha, beta, maxi_player):
    '''
    Аьфа-бета
    '''
    valid_cols = valid_locations(board)

    if is_terminal_board(board):
        if detect_win(board, HUMAN): return (None,-100000000)
        elif detect_win(board, AI): return (None,1000000000)
        else: return (None,0)

    if depth == 0: return (None,score(board, AI))

    # Максимизируем игрока
    if maxi_player:
        value = -np.inf
        # Выбираем случайных ход на случий равных ходов
        next_col = random.choice(valid_cols)

        for col in valid_cols:
            next_board = clone_and_place_piece(board, AI, col)
            new_score = minimax(next_board, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                next_col = col

            if value > alpha:
                alpha = new_score
            
            # Отсекаем
            if beta <= alpha:
                break
        return next_col, value
    
    # Минимизируем игрока
    else:
        value = np.inf
        next_col = random.choice(valid_cols)
        for col in valid_cols:
            next_board = clone_and_place_piece(board, HUMAN, col)
            new_score = minimax(next_board, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                next_col = col

            if value < beta:
                beta  = value

            # Отсекаем
            if beta <= alpha:
                break
        return next_col, value


def draw_game(board, turn, game_over=False, AI_move=0):
    highlight_index = AI_move - 1

    #  Взял с StackOverflow красивую доску
    for row in board:
        line = "\033[4;30;47m|\033[0m"
        for col, piece in enumerate(row):
            if piece == HUMAN:
                if col == highlight_index:
                    highlight_index = -1
                line += "\033[4;34;47m●\033[0m"
            elif piece == AI:
                if col == highlight_index:
                    line = line[:-19] + "\033[4;31;43m|●|\033[0m"
                    highlight_index = -1
                    continue
                else:
                    line += "\033[4;31;47m●\033[0m"
            else:
                line += "\033[4;30;47m \033[0m"
            line += "\033[4;30;47m|\033[0m"
        print("                       " + line + "  ")
    print("                        1 2 3 4 5 6 7  ")

    if turn == HUMAN and not game_over:
        print("Ход игрока: ")
    elif turn == AI and not game_over:
        print("Ход ИИ: ")
    elif turn == HUMAN and game_over:
        print("Игрок выйграл.")
    elif turn == AI and game_over:
        print("ИИ выйграл. ")

    if not game_over and turn != HUMAN:
        print("Ждём ход компьютера...")



# =================
#
# Начало самой игры
#
# =================


board = create_board()
turn = random.choice([HUMAN, AI])

is_game_won = False
AI_move = -1
running_time = 0
draw_game(board, turn)

minimax_times = []

while not is_game_won:
    
    if turn == HUMAN:
        pressed_key = int(input("Наберите колонку в которую ставить: "))

        if not is_valid_column(board, pressed_key):
            print("Неправильный ввод")
            continue

        # Продолжаем игру
        place_piece(board, HUMAN, pressed_key)
        is_game_won = detect_win(board, turn)
        if not is_game_won:
            turn = AI
            draw_game(board, turn)
            continue

        # Если мы выйграли
        draw_game(board, turn, game_over=True)
        break


    elif turn == AI:
        initial_time = time.time()

        AI_move, minimax_value = minimax(board, DEPTH, -np.inf, np.inf, True)
        place_piece(board, AI, AI_move)
        is_game_won = detect_win(board, AI)

        # Замеряем время
        running_time = time.time() - initial_time
        minimax_times.append(running_time)

        # Продолжаем игру если не победили
        if not is_game_won:
            turn = HUMAN
            draw_game(board, turn, AI_move=AI_move)
            continue

        # Победили
        draw_game(board, turn, game_over=True, AI_move=AI_move)
        break


if is_game_won:
    running_time = sum(minimax_times) / len(minimax_times)
    print(f"Среднее время рассчёта ИИ: {running_time} s")