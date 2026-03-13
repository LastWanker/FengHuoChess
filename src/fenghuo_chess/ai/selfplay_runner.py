"""Headless self-play runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.fenghuo_chess.application.logic_api import apply_action, legal_actions
from src.fenghuo_chess.application.match_runner import MatchRunner, build_match_runner
from src.fenghuo_chess.domain.models import create_initial_state
from src.fenghuo_chess.services.exposure_service import ExposureService
from src.fenghuo_chess.services.trace_recorder import TraceRecorder


SelfPlayProgressFn = Callable[[int, int, int, int, int, int], None]


@dataclass
class SelfPlaySummary:
    games: int
    p1_wins: int
    p2_wins: int
    draws: int
    steps: int
    output_path: str | None = None


def run_selfplay(
    games: int = 1,
    *,
    output_path: str | None = None,
    append_output: bool = True,
    runner: MatchRunner | None = None,
    progress_callback: SelfPlayProgressFn | None = None,
    initial_game_mode: str | None = None,
) -> SelfPlaySummary:
    out_path = Path(output_path) if output_path else None
    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_steps = 0

    for game_idx in range(games):
        state = create_initial_state()
        state.show_intro = False
        if initial_game_mode in {"fast", "slow"}:
            state.game_mode = initial_game_mode

        game_runner = runner or build_match_runner("eve", with_ui=False)
        exposure_service = ExposureService()
        recorder = TraceRecorder()

        while not state.game_over:
            legal = legal_actions(state)
            if not legal:
                break

            action = game_runner.select_action(state, legal)
            if action is None:
                break

            player = state.current_player
            recorder.record_step(
                state=state,
                legal_actions=legal,
                action=action,
                player=player,
                events=[],
            )
            step = apply_action(state, action, exposure_service=exposure_service)
            recorder.rows[-1]["events"] = [event.kind for event in step.events]

            if not step.ok:
                break
            total_steps += 1

        recorder.finalize_outcome(state.winner)
        if out_path:
            recorder.write_jsonl(out_path, append=append_output or game_idx > 0)

        if state.winner == 1:
            p1_wins += 1
        elif state.winner == 2:
            p2_wins += 1
        else:
            draws += 1

        if progress_callback is not None:
            progress_callback(game_idx + 1, games, p1_wins, p2_wins, draws, total_steps)

    return SelfPlaySummary(
        games=games,
        p1_wins=p1_wins,
        p2_wins=p2_wins,
        draws=draws,
        steps=total_steps,
        output_path=str(out_path) if out_path else None,
    )
