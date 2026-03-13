"""GameState serialization helpers for persistence and training exports."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.fenghuo_chess.domain.models import GameState
from src.fenghuo_chess.domain.rules import get_stage_positions


def state_to_dict(state: GameState, core_only: bool = True) -> dict[str, Any]:
    data: dict[str, Any] = {
        "board": state.board.tolist(),
        "stage": int(state.stage),
        "current_player": int(state.current_player),
        "game_mode": state.game_mode,
        "mode_locked": bool(state.mode_locked),
        "game_over": bool(state.game_over),
        "winner": int(state.winner),
        "stage_positions": _pairs_to_lists(state.stage_positions),
        "exposure_core": {
            "exposure_occurred": bool(state.exposure_occurred),
            "exposure_positions": _pairs_to_lists(state.exposure_positions),
            "exposure_markers": _pairs_to_lists(state.exposure_markers),
            "checked_positions": _pairs_to_lists(state.checked_positions),
            "last_exposure_player": int(state.last_exposure_player),
            "consecutive_exposure": bool(state.consecutive_exposure),
        },
    }

    if core_only:
        return data

    data["message"] = state.message
    data["exposure_message"] = state.exposure_message
    data["show_intro"] = bool(state.show_intro)
    data["intro_p1_source"] = state.intro_p1_source
    data["intro_p2_source"] = state.intro_p2_source
    data["end_overlay_closed"] = bool(state.end_overlay_closed)
    data["restart_button_state"] = state.restart_button_state
    data["restart_button_press_time"] = float(state.restart_button_press_time)
    return data


def state_from_dict(data: dict[str, Any]) -> GameState:
    board = np.array(data["board"], dtype=int)
    state = GameState(board=board)

    state.stage = int(data.get("stage", state.stage))
    state.current_player = int(data.get("current_player", state.current_player))
    state.game_mode = str(data.get("game_mode", state.game_mode))
    state.mode_locked = bool(data.get("mode_locked", state.mode_locked))
    state.game_over = bool(data.get("game_over", state.game_over))
    state.winner = int(data.get("winner", state.winner))
    state.stage_positions = _lists_to_pairs(data.get("stage_positions")) or get_stage_positions(state.stage)

    exposure_core = data.get("exposure_core", {})
    state.exposure_occurred = bool(exposure_core.get("exposure_occurred", state.exposure_occurred))
    state.exposure_positions = _lists_to_pairs(exposure_core.get("exposure_positions"))
    state.exposure_markers = _lists_to_pairs(exposure_core.get("exposure_markers"))
    state.checked_positions = set(_lists_to_pairs(exposure_core.get("checked_positions")))
    state.last_exposure_player = int(exposure_core.get("last_exposure_player", state.last_exposure_player))
    state.consecutive_exposure = bool(exposure_core.get("consecutive_exposure", state.consecutive_exposure))

    state.message = str(data.get("message", state.message))
    state.exposure_message = str(data.get("exposure_message", state.exposure_message))
    state.show_intro = bool(data.get("show_intro", state.show_intro))
    state.intro_p1_source = str(data.get("intro_p1_source", state.intro_p1_source))
    state.intro_p2_source = str(data.get("intro_p2_source", state.intro_p2_source))
    # Backward compatibility for old intro fields.
    legacy_mode = data.get("intro_match_mode")
    legacy_human = data.get("intro_human_player")
    if legacy_mode and ("intro_p1_source" not in data and "intro_p2_source" not in data):
        mode = str(legacy_mode)
        human = int(legacy_human) if legacy_human is not None else 1
        if mode == "pvp":
            state.intro_p1_source, state.intro_p2_source = "human", "human"
        elif mode == "pve":
            if human == 2:
                state.intro_p1_source, state.intro_p2_source = "baseline", "human"
            else:
                state.intro_p1_source, state.intro_p2_source = "human", "baseline"
        elif mode == "eve":
            state.intro_p1_source, state.intro_p2_source = "baseline", "baseline"
    state.end_overlay_closed = bool(data.get("end_overlay_closed", state.end_overlay_closed))
    state.restart_button_state = str(data.get("restart_button_state", state.restart_button_state))
    state.restart_button_press_time = float(data.get("restart_button_press_time", state.restart_button_press_time))
    return state


def _pairs_to_lists(items: list[tuple[int, int]] | set[tuple[int, int]]) -> list[list[int]]:
    return [[int(r), int(c)] for r, c in items]


def _lists_to_pairs(items: Any) -> list[tuple[int, int]]:
    if not items:
        return []
    return [(int(v[0]), int(v[1])) for v in items]
