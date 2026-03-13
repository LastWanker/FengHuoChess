"""End-game scoring helpers."""

import numpy as np

from src.fenghuo_chess.constants.gameplay import BOARD_SIZE


def _is_three_in_direction(board: np.ndarray, row: int, col: int, dr: int, dc: int) -> bool:
    player = int(board[row, col])
    count = 1

    r, c = row + dr, col + dc
    while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == player:
        count += 1
        r += dr
        c += dc

    r, c = row - dr, col - dc
    while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == player:
        count += 1
        r -= dr
        c -= dc

    return count >= 3


def count_threes(board: np.ndarray, player: int) -> int:
    count = 0
    visited: set[tuple[int, int]] = set()
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i, j] == player and (i, j) not in visited:
                for dr, dc in directions:
                    if _is_three_in_direction(board, i, j, dr, dc):
                        seq = [(i + k * dr, j + k * dc) for k in range(3)]
                        if not any(pos in visited for pos in seq):
                            for pos in seq:
                                visited.add(pos)
                            count += 1
                            break
    return count


def determine_winner(board: np.ndarray) -> int:
    p1 = count_threes(board, 1)
    p2 = count_threes(board, 2)
    if p1 > p2:
        return 1
    if p2 > p1:
        return 2
    return 0


