"""Main pygame loop."""

import sys
import time
from typing import Literal, cast

import pygame

from src.fenghuo_chess.application.game_controller import GameController
from src.fenghuo_chess.application.match_runner import MatchMode, SourceKind
from src.fenghuo_chess.constants.gameplay import UI_AI_MIN_TURN_DELAY
from src.fenghuo_chess.ui.broadcast import BroadcastOverlay
from src.fenghuo_chess.ui.input_mapper import board_coords_from_pos, compute_layout
from src.fenghuo_chess.ui.renderer import GameRenderer


def run(match_mode: MatchMode = "pvp", human_player: int = 1) -> None:
    pygame.init()
    min_w, min_h = 920, 620
    screen = pygame.display.set_mode((1280, 820), pygame.RESIZABLE)
    pygame.display.set_caption("锋火 三阶段战略棋")

    controller = GameController(match_mode=match_mode, with_ui=True, human_player=human_player)
    renderer = GameRenderer()
    broadcasts = BroadcastOverlay()
    clock = pygame.time.Clock()
    next_ai_step_at: float | None = None
    pending_ai_reaction = False

    while True:
        layout = compute_layout(screen.get_width(), screen.get_height())
        mouse_pos = pygame.mouse.get_pos()

        broadcasts.enqueue_many(controller.consume_broadcasts())
        broadcasts.update(*layout.screen_size)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((max(min_w, event.w), max(min_h, event.h)), pygame.RESIZABLE)
                layout = compute_layout(screen.get_width(), screen.get_height())

            if event.type == pygame.MOUSEBUTTONDOWN:
                click_result = _handle_click(controller, layout, mouse_pos)
                if click_result in {"intro_started", "human_moved"}:
                    pending_ai_reaction = True
                    next_ai_step_at = None
                elif click_result == "undo":
                    pending_ai_reaction = False
                    next_ai_step_at = None
                elif click_result == "restart":
                    pending_ai_reaction = False
                    next_ai_step_at = None
                elif click_result in {"intro_config", "board_click", "end_close"}:
                    next_ai_step_at = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_u:
                    if controller.undo():
                        pending_ai_reaction = False
                        next_ai_step_at = None

        state = controller.state
        if not state.show_intro and not state.game_over:
            is_human_turn = controller.match_runner.is_human_turn(state.current_player)
            if is_human_turn:
                next_ai_step_at = None
            else:
                allow_ai_auto = (not controller.match_runner.has_human_players()) or pending_ai_reaction
                if not allow_ai_auto:
                    next_ai_step_at = None
                else:
                    now = time.monotonic()
                    if next_ai_step_at is None:
                        next_ai_step_at = now + UI_AI_MIN_TURN_DELAY
                    if now >= next_ai_step_at:
                        stepped = controller.step_turn()
                        next_ai_step_at = None
                        if not controller.match_runner.has_human_players():
                            pending_ai_reaction = True
                        elif stepped:
                            pending_ai_reaction = not controller.match_runner.is_human_turn(
                                controller.state.current_player
                            )
                        else:
                            pending_ai_reaction = False
        controller.update_restart_confirmation_timeout()

        renderer.draw(screen, controller.state, layout, mouse_pos, broadcasts)
        pygame.display.flip()
        clock.tick(60)


def _handle_click(
    controller: GameController, layout, mouse_pos: tuple[int, int]
) -> Literal["none", "intro_config", "intro_started", "undo", "restart", "human_moved", "board_click", "end_close"]:
    state = controller.state

    if state.show_intro:
        if layout.intro_p1_human_rect.collidepoint(mouse_pos):
            state.intro_p1_source = "human"
            return "intro_config"
        if layout.intro_p1_weak_rect.collidepoint(mouse_pos):
            state.intro_p1_source = "weak"
            return "intro_config"
        if layout.intro_p1_baseline_rect.collidepoint(mouse_pos):
            state.intro_p1_source = "baseline"
            return "intro_config"
        if layout.intro_p1_master_rect.collidepoint(mouse_pos):
            state.intro_p1_source = "master"
            return "intro_config"
        if layout.intro_p2_human_rect.collidepoint(mouse_pos):
            state.intro_p2_source = "human"
            return "intro_config"
        if layout.intro_p2_weak_rect.collidepoint(mouse_pos):
            state.intro_p2_source = "weak"
            return "intro_config"
        if layout.intro_p2_baseline_rect.collidepoint(mouse_pos):
            state.intro_p2_source = "baseline"
            return "intro_config"
        if layout.intro_p2_master_rect.collidepoint(mouse_pos):
            state.intro_p2_source = "master"
            return "intro_config"
        if layout.intro_close_rect.collidepoint(mouse_pos):
            controller.configure_players(
                p1_source_kind=cast(SourceKind, state.intro_p1_source),
                p2_source_kind=cast(SourceKind, state.intro_p2_source),
                with_ui=True,
            )
            state.show_intro = False
            return "intro_started"
        return "intro_config"

    # Undo/restart should always remain available, even when the game is over.
    if layout.undo_rect.collidepoint(mouse_pos):
        if controller.undo():
            return "undo"
        return "none"

    if layout.restart_rect.collidepoint(mouse_pos):
        controller.click_restart()
        return "restart"

    if state.game_over:
        if layout.end_close_rect.collidepoint(mouse_pos):
            state.end_overlay_closed = True
            return "end_close"
        return "none"

    if not state.mode_locked:
        if layout.fast_mode_rect.collidepoint(mouse_pos):
            controller.change_mode("fast")
            return "none"
        if layout.slow_mode_rect.collidepoint(mouse_pos):
            controller.change_mode("slow")
            return "none"

    board_pos = board_coords_from_pos(mouse_pos, layout)
    if board_pos is not None:
        if controller.make_move(*board_pos):
            return "human_moved"
        return "board_click"
    return "none"
