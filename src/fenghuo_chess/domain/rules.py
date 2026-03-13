"""Rule helpers that operate on GameState."""

from typing import Literal

import numpy as np

from src.fenghuo_chess.constants.gameplay import BOARD_SIZE, STAGE_TITLES, WIN_CONDITIONS
from src.fenghuo_chess.domain.models import GameState


def get_stage_positions(stage: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    center = BOARD_SIZE // 2

    if stage == 1:
        for i in range(center - 1, center + 2):
            for j in range(center - 1, center + 2):
                positions.append((i, j))
    elif stage == 2:
        for i in range(center - 2, center + 3):
            for j in range(center - 2, center + 3):
                if not (center - 1 <= i <= center + 1 and center - 1 <= j <= center + 1):
                    positions.append((i, j))
    elif stage == 3:
        for i in range(center - 4, center + 5):
            for j in range(center - 4, center + 5):
                if not (center - 2 <= i <= center + 2 and center - 2 <= j <= center + 2):
                    positions.append((i, j))
    elif stage == 4:
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                positions.append((i, j))

    return positions


def is_valid_move(state: GameState, row: int, col: int) -> bool:
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        return False
    if state.board[row, col] != 0:
        return False
    if (row, col) not in state.stage_positions:
        return False
    return True


def count_stones(board: np.ndarray) -> int:
    return int(np.count_nonzero(board))


def _win_length(stage: int, game_mode: str) -> int:
    win_length = WIN_CONDITIONS[stage]
    if isinstance(win_length, dict):
        return int(win_length[game_mode])
    return int(win_length)


def check_win(board: np.ndarray, row: int, col: int, stage: int, game_mode: str) -> bool:
    player = int(board[row, col])
    if player == 0:
        return False

    target = _win_length(stage, game_mode)
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for dr, dc in directions:
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

        if count >= target:
            return True
    return False


def is_out_of_battlefield(stage: int, row: int, col: int) -> bool:
    center = BOARD_SIZE // 2
    if stage == 2:
        return not (center - 2 <= row <= center + 2 and center - 2 <= col <= center + 2)
    if stage == 3:
        return not (center - 4 <= row <= center + 4 and center - 4 <= col <= center + 4)
    return False


def advance_stage_if_needed(state: GameState) -> Literal["none", "stage", "final_scoring"]:
    for row, col in state.stage_positions:
        if state.board[row, col] == 0:
            return "none"

    state.stage += 1
    state.exposure_positions = []
    state.exposure_markers = []
    state.checked_positions = set()

    if state.stage in STAGE_TITLES:
        state.message = STAGE_TITLES[state.stage]
        if state.stage == 2:
            state.mode_locked = False
        elif state.stage >= 3:
            state.mode_locked = True
        state.stage_positions = get_stage_positions(state.stage)
        return "stage"

    state.game_over = True
    return "final_scoring"


