"""Heuristic teacher AI based on simulate-and-score."""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from src.fenghuo_chess.application.logic_api import Action, simulate_action
from src.fenghuo_chess.constants.gameplay import BOARD_SIZE
from src.fenghuo_chess.domain.models import GameState
from src.fenghuo_chess.domain.rules import check_win
from src.fenghuo_chess.domain.scoring import count_threes
from src.fenghuo_chess.services.exposure_service import ExposureService


class HeuristicAISource:
    """A fast, rules-based teacher policy that is stronger than random play."""

    def __init__(
        self,
        *,
        rng: random.Random | None = None,
        tie_noise: float = 1e-3,
        threat_cap: int = 8,
        explore_second_prob: float = 0.25,
        explore_gap_threshold: float = 320.0,
        prune_top_k: int = 40,
        prune_trigger_legal_count: int = 24,
    ) -> None:
        self._rng = rng or random.Random()
        self._tie_noise = tie_noise
        self._threat_cap = max(1, threat_cap)
        self._explore_second_prob = max(0.0, min(1.0, float(explore_second_prob)))
        self._explore_gap_threshold = max(0.0, float(explore_gap_threshold))
        self._prune_top_k = max(0, int(prune_top_k))
        self._prune_trigger_legal_count = max(2, int(prune_trigger_legal_count))
        self._exposure_service = ExposureService()

    def select_action(self, state: GameState, legal_actions: Sequence[Action]) -> Action | None:
        if not legal_actions:
            return None

        if self._should_use_stage1_solver(state, legal_actions):
            solved = self._select_stage1_action(state, legal_actions)
            if solved is not None:
                return solved

        me = int(state.current_player)
        opp = 3 - me
        candidates = self._prune_candidates(state, legal_actions, me, opp)
        weights = self._mode_weights(state)
        pre_opp_direct_wins = self._count_direct_wins(state, opp, cap=self._threat_cap)
        scored: list[tuple[float, Action]] = []

        for action in candidates:
            result = simulate_action(state, action, exposure_service=self._exposure_service)
            if not result.ok:
                continue

            score = self._score_after_move(
                pre=state,
                post=result.state,
                me=me,
                weights=weights,
                pre_opp_direct_wins=pre_opp_direct_wins,
            )
            score += self._rng.uniform(-self._tie_noise, self._tie_noise)
            scored.append((score, action))

        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            if self._should_pick_second(scored):
                return scored[1][1]
            return scored[0][1]

        # Fallback for extreme edge cases (should not happen with valid legal_actions).
        return self._rng.choice(list(legal_actions))

    def _should_pick_second(self, scored: list[tuple[float, Action]]) -> bool:
        if self._explore_second_prob <= 0.0:
            return False
        if len(scored) < 2:
            return False
        gap = float(scored[0][0] - scored[1][0])
        if gap > self._explore_gap_threshold:
            return False
        return self._rng.random() < self._explore_second_prob

    def _prune_candidates(
        self,
        state: GameState,
        legal_actions: Sequence[Action],
        me: int,
        opp: int,
    ) -> list[Action]:
        legal = list(legal_actions)
        if self._prune_top_k <= 0:
            return legal
        if len(legal) <= self._prune_trigger_legal_count:
            return legal

        board = state.board
        immediate_wins: list[Action] = []
        forced_blocks: list[Action] = []
        for action in legal:
            row, col = action.row, action.col

            board[row, col] = me
            me_wins = check_win(board, row, col, state.stage, state.game_mode)
            board[row, col] = 0
            if me_wins:
                immediate_wins.append(action)
                continue

            board[row, col] = opp
            opp_wins = check_win(board, row, col, state.stage, state.game_mode)
            board[row, col] = 0
            if opp_wins:
                forced_blocks.append(action)

        if immediate_wins:
            return immediate_wins
        if forced_blocks:
            return forced_blocks

        ranked = [(self._cheap_score(state, action, me, opp), action) for action in legal]
        ranked.sort(key=lambda x: x[0], reverse=True)
        keep = min(len(ranked), self._prune_top_k)
        return [action for _, action in ranked[:keep]]

    def _cheap_score(self, state: GameState, action: Action, me: int, opp: int) -> float:
        board = state.board
        row, col = action.row, action.col
        center = BOARD_SIZE // 2

        center_bias = max(0, 8 - (abs(row - center) + abs(col - center)))
        my_neighbors = 0
        opp_neighbors = 0
        for r in range(max(0, row - 1), min(BOARD_SIZE, row + 2)):
            for c in range(max(0, col - 1), min(BOARD_SIZE, col + 2)):
                if r == row and c == col:
                    continue
                piece = int(board[r, c])
                if piece == me:
                    my_neighbors += 1
                elif piece == opp:
                    opp_neighbors += 1

        my_line = self._line_potential(board, row, col, me)
        opp_line = self._line_potential(board, row, col, opp)
        return my_line * 8.0 + opp_line * 5.0 + my_neighbors * 2.0 + opp_neighbors * 1.6 + center_bias

    @staticmethod
    def _line_potential(board: np.ndarray, row: int, col: int, player: int) -> int:
        best = 1
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            run = 1

            r, c = row + dr, col + dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and int(board[r, c]) == player:
                run += 1
                r += dr
                c += dc

            r, c = row - dr, col - dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and int(board[r, c]) == player:
                run += 1
                r -= dr
                c -= dc

            if run > best:
                best = run
        return best

    def _should_use_stage1_solver(self, state: GameState, legal_actions: Sequence[Action]) -> bool:
        return state.stage == 1 and len(state.stage_positions) <= 9 and len(legal_actions) <= 9

    def _select_stage1_action(self, state: GameState, legal_actions: Sequence[Action]) -> Action | None:
        me = int(state.current_player)
        cache: dict[tuple, int] = {}
        best_value = -2
        best_actions: list[Action] = []

        for action in legal_actions:
            result = simulate_action(state, action, exposure_service=self._exposure_service)
            if not result.ok:
                continue
            value = self._stage1_value(result.state, me, cache)
            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)

        if not best_actions:
            return None
        return self._rng.choice(best_actions)

    def _stage1_value(self, state: GameState, me: int, cache: dict[tuple, int]) -> int:
        key = (
            tuple(int(x) for x in state.board.ravel()),
            int(state.current_player),
            int(state.stage),
            int(state.game_over),
            int(state.winner),
        )
        if key in cache:
            return cache[key]

        opp = 3 - me
        if state.game_over:
            if state.winner == me:
                cache[key] = 1
            elif state.winner == opp:
                cache[key] = -1
            else:
                cache[key] = 0
            return cache[key]

        if state.stage != 1:
            cache[key] = 0
            return 0

        legal = [Action(r, c) for r, c in state.stage_positions if state.board[r, c] == 0]
        if not legal:
            cache[key] = 0
            return 0

        maximizing = int(state.current_player) == me
        best = -2 if maximizing else 2

        for action in legal:
            result = simulate_action(state, action, exposure_service=self._exposure_service)
            if not result.ok:
                continue
            value = self._stage1_value(result.state, me, cache)
            if maximizing:
                if value > best:
                    best = value
                if best == 1:
                    break
            else:
                if value < best:
                    best = value
                if best == -1:
                    break

        if best == -2 or best == 2:
            best = 0
        cache[key] = best
        return best

    def _score_after_move(
        self,
        *,
        pre: GameState,
        post: GameState,
        me: int,
        weights: dict[str, float] | None = None,
        pre_opp_direct_wins: int | None = None,
    ) -> float:
        opp = 3 - me

        if post.game_over:
            if post.winner == me:
                return 1_000_000.0
            if post.winner == opp:
                return -1_000_000.0
            return -500.0

        score = 0.0
        board = post.board
        if weights is None:
            weights = self._mode_weights(pre)

        if pre_opp_direct_wins is None:
            pre_opp_direct_wins = self._count_direct_wins(pre, opp, cap=self._threat_cap)
        my_stones = int(np.count_nonzero(board == me))
        opp_stones = int(np.count_nonzero(board == opp))
        score += (my_stones - opp_stones) * weights["stone_diff"]

        score += self._center_pressure(post, me) * weights["center_me"]
        score -= self._center_pressure(post, opp) * weights["center_opp"]

        my_direct_wins = self._count_direct_wins(post, me, cap=self._threat_cap)
        opp_direct_wins = self._count_direct_wins(post, opp, cap=self._threat_cap)
        score += my_direct_wins * weights["direct_me"]
        score -= opp_direct_wins * weights["direct_opp"]

        blocked_opp_wins = max(0, pre_opp_direct_wins - opp_direct_wins)
        score += blocked_opp_wins * weights["blocked_opp"]

        if pre.stage in {2, 3} and post.exposure_occurred and post.last_exposure_player == me:
            stage_penalty = weights["exposure_stage2"] if pre.stage == 2 else weights["exposure_stage3"]
            score -= stage_penalty
            score -= len(post.exposure_positions) * weights["exposure_piece"]

        if post.stage == 3:
            if post.game_mode == "fast":
                my_triplets = self._count_lines_at_least(post, me, 3, cap=24)
                opp_triplets = self._count_lines_at_least(post, opp, 3, cap=24)
                score += (my_triplets - opp_triplets) * 42.0
            else:
                my_fours = self._count_lines_at_least(post, me, 4, cap=16)
                opp_fours = self._count_lines_at_least(post, opp, 4, cap=16)
                score += (my_fours - opp_fours) * 120.0

        if post.stage >= 4:
            score += (count_threes(board, me) - count_threes(board, opp)) * 80.0

        if post.stage > pre.stage:
            score += weights["stage_advance"]

        return score

    @staticmethod
    def _mode_weights(state: GameState) -> dict[str, float]:
        weights: dict[str, float] = {
            "stone_diff": 4.0,
            "center_me": 2.4,
            "center_opp": 1.8,
            "direct_me": 900.0,
            "direct_opp": 14_000.0,
            "blocked_opp": 2_400.0,
            "exposure_stage2": 220.0,
            "exposure_stage3": 320.0,
            "exposure_piece": 28.0,
            "stage_advance": 10.0,
        }
        if state.stage == 3 and state.game_mode == "fast":
            weights.update(
                direct_me=1_150.0,
                direct_opp=16_000.0,
                blocked_opp=2_800.0,
                exposure_stage3=280.0,
                exposure_piece=24.0,
                stage_advance=14.0,
            )
        elif state.stage == 3 and state.game_mode == "slow":
            weights.update(
                direct_me=780.0,
                direct_opp=18_000.0,
                blocked_opp=3_200.0,
                exposure_stage3=500.0,
                exposure_piece=42.0,
                stage_advance=7.0,
            )
        return weights

    @staticmethod
    def _center_pressure(state: GameState, player: int) -> float:
        center = BOARD_SIZE // 2
        total = 0.0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if int(state.board[row, col]) != player:
                    continue
                dist = abs(row - center) + abs(col - center)
                total += max(0, 8 - dist)
        return total

    @staticmethod
    def _count_direct_wins(state: GameState, player: int, cap: int) -> int:
        wins = 0
        board = state.board
        for row, col in state.stage_positions:
            if board[row, col] != 0:
                continue
            board[row, col] = player
            if check_win(board, row, col, state.stage, state.game_mode):
                wins += 1
            board[row, col] = 0
            if wins >= cap:
                return wins
        return wins

    @staticmethod
    def _count_lines_at_least(state: GameState, player: int, length: int, cap: int) -> int:
        if length <= 1:
            return 0
        board = state.board
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        lines = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if int(board[row, col]) != player:
                    continue
                for dr, dc in directions:
                    prev_r, prev_c = row - dr, col - dc
                    if 0 <= prev_r < BOARD_SIZE and 0 <= prev_c < BOARD_SIZE and int(board[prev_r, prev_c]) == player:
                        continue
                    run = 0
                    r, c = row, col
                    while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and int(board[r, c]) == player:
                        run += 1
                        r += dr
                        c += dc
                    if run >= length:
                        lines += 1
                        if lines >= cap:
                            return lines
        return lines


class MasterAISource(HeuristicAISource):
    """Placeholder source for future stronger AI tiers."""

    def __init__(self, *, rng: random.Random | None = None) -> None:
        # Placeholder: currently same behavior as baseline heuristic.
        super().__init__(rng=rng)
