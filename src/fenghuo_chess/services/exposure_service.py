"""Exposure (出锋) detection and penalty service."""

from collections.abc import Callable

from src.fenghuo_chess.constants.colors import RED
from src.fenghuo_chess.constants.gameplay import BOARD_SIZE
from src.fenghuo_chess.domain.models import GameState
from src.fenghuo_chess.domain.rules import check_win, is_out_of_battlefield


BroadcastFn = Callable[[str, tuple[int, int, int]], None]


class ExposureService:
    @staticmethod
    def _required_exposure_length(state: GameState) -> int:
        # Stage 3 in slow mode relaxes exposure trigger from 3-in-line to 4-in-line.
        if state.stage == 3 and state.game_mode == "slow":
            return 4
        return 3

    @staticmethod
    def _in_board(row: int, col: int) -> bool:
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def _line_exposure_from_anchor(
        self, state: GameState, row: int, col: int, dr: int, dc: int, required: int
    ) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], tuple[int, int]] | None:
        board = state.board
        player = int(board[row, col])
        if player == 0:
            return None

        prev_r, prev_c = row - dr, col - dc
        if self._in_board(prev_r, prev_c) and int(board[prev_r, prev_c]) == player:
            return None

        run_len = 0
        r, c = row, col
        while self._in_board(r, c) and int(board[r, c]) == player:
            run_len += 1
            r += dr
            c += dc
        if run_len < required:
            return None

        items: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for offset in range(run_len - required + 1):
            start_r = row + offset * dr
            start_c = col + offset * dc
            end_r = start_r + (required - 1) * dr
            end_c = start_c + (required - 1) * dc

            before = (start_r - dr, start_c - dc)
            if (
                self._in_board(before[0], before[1])
                and board[before[0], before[1]] == 0
                and is_out_of_battlefield(state.stage, before[0], before[1])
            ):
                items.append((before, (-dr, -dc)))

            after = (end_r + dr, end_c + dc)
            if (
                self._in_board(after[0], after[1])
                and board[after[0], after[1]] == 0
                and is_out_of_battlefield(state.stage, after[0], after[1])
            ):
                items.append((after, (dr, dc)))

        if not items:
            return None

        deduped: list[tuple[tuple[int, int], tuple[int, int]]] = []
        seen: set[tuple[int, int, int, int]] = set()
        for pos, direction in items:
            key = (pos[0], pos[1], direction[0], direction[1])
            if key in seen:
                continue
            seen.add(key)
            deduped.append((pos, direction))

        marker_offset = (required - 1) // 2
        marker = (row + marker_offset * dr, col + marker_offset * dc)
        return deduped, marker

    def _apply_penalty(
        self,
        state: GameState,
        exposure_items: list[tuple[tuple[int, int], tuple[int, int]]],
        center: tuple[int, int],
        broadcast: BroadcastFn,
    ) -> bool:
        player = int(state.board[center[0], center[1]])
        opponent = 3 - player
        penalty_made = False
        msg_parts: list[str] = []

        if len(exposure_items) == 2:
            broadcast("锋芒毕露", RED)

        for pos, direction in exposure_items:
            r, c = pos
            if state.board[r, c] != 0:
                continue

            state.board[r, c] = opponent
            state.exposure_positions.append((r, c))
            msg_parts.append(f"({r},{c})")
            penalty_made = True

            if check_win(state.board, r, c, state.stage, state.game_mode):
                state.game_over = True
                state.winner = opponent
                return True

            if state.stage == 3:
                r2, c2 = r + direction[0], c + direction[1]
                if 0 <= r2 < BOARD_SIZE and 0 <= c2 < BOARD_SIZE and state.board[r2, c2] == 0:
                    state.board[r2, c2] = opponent
                    state.exposure_positions.append((r2, c2))
                    msg_parts.append(f"({r2},{c2})")
                    if check_win(state.board, r2, c2, state.stage, state.game_mode):
                        state.game_over = True
                        state.winner = opponent
                        return True

        if penalty_made:
            positions = "和".join(msg_parts)
            if state.stage == 2:
                state.exposure_message = f"玩家{player}出锋！对手截锋于{positions}"
            else:
                state.exposure_message = f"玩家{player}出锋！对手双截锋于{positions}"

            if state.last_exposure_player != 0 and state.consecutive_exposure:
                if state.last_exposure_player != player:
                    broadcast("黄雀在后", RED)
                else:
                    broadcast("买一送一", RED)

            state.last_exposure_player = player
            state.consecutive_exposure = True

        return penalty_made

    def check_all_exposures(self, state: GameState, broadcast: BroadcastFn) -> bool:
        if state.stage == 1 or state.game_over:
            return False

        penalty_occurred = False
        required = self._required_exposure_length(state)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if state.board[row, col] == 0:
                    continue

                for dr, dc in directions:
                    found = self._line_exposure_from_anchor(state, row, col, dr, dc, required)
                    if found is None:
                        continue

                    exposure_items, marker = found
                    if marker not in state.exposure_markers:
                        state.exposure_markers.append(marker)
                    if self._apply_penalty(state, exposure_items, marker, broadcast):
                        return True
                    penalty_occurred = True

        if not penalty_occurred:
            state.consecutive_exposure = False

        return penalty_occurred


