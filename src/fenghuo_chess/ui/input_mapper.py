"""Responsive layout and hit-test helpers (always horizontal)."""

from dataclasses import dataclass

import pygame

from src.fenghuo_chess.constants.gameplay import BOARD_SIZE


@dataclass(frozen=True)
class UILayout:
    screen_size: tuple[int, int]
    pad: int
    top_rect: pygame.Rect
    board_plate_rect: pygame.Rect
    board_rect: pygame.Rect
    sidebar_rect: pygame.Rect
    player_rect: pygame.Rect
    mode_rect: pygame.Rect
    fast_mode_rect: pygame.Rect
    slow_mode_rect: pygame.Rect
    exposure_rect: pygame.Rect
    undo_rect: pygame.Rect
    restart_rect: pygame.Rect
    intro_rect: pygame.Rect
    intro_rules_rect: pygame.Rect
    intro_config_rect: pygame.Rect
    intro_p1_human_rect: pygame.Rect
    intro_p1_weak_rect: pygame.Rect
    intro_p1_baseline_rect: pygame.Rect
    intro_p1_master_rect: pygame.Rect
    intro_p2_human_rect: pygame.Rect
    intro_p2_weak_rect: pygame.Rect
    intro_p2_baseline_rect: pygame.Rect
    intro_p2_master_rect: pygame.Rect
    intro_close_rect: pygame.Rect
    end_rect: pygame.Rect
    end_close_rect: pygame.Rect
    grid_origin: tuple[int, int]
    cell: int
    grid_span: int


def compute_layout(screen_w: int, screen_h: int) -> UILayout:
    pad = max(14, int(min(screen_w, screen_h) * 0.018))
    top_h = max(62, int(screen_h * 0.10))
    top_rect = pygame.Rect(pad, pad, screen_w - pad * 2, top_h)

    # Always horizontal: board on the left, sidebar on the right.
    side_w = min(400, max(260, int(screen_w * 0.28)))
    sidebar_rect = pygame.Rect(screen_w - side_w - pad, top_rect.bottom + pad, side_w, screen_h - top_h - pad * 3)

    board_area_x = pad
    board_area_y = top_rect.bottom + pad * 2
    board_area_w = sidebar_rect.left - pad * 2
    board_area_h = screen_h - top_h - pad * 5

    # Reserve space for rounded board plate so it never presses top/bottom edges.
    plate_factor = 1.4
    cell = max(14, min(72, int(min(board_area_w, board_area_h) / ((BOARD_SIZE - 1) + plate_factor))))
    grid_span = cell * (BOARD_SIZE - 1)
    grid_x = board_area_x + (board_area_w - grid_span) // 2
    grid_y = board_area_y + (board_area_h - grid_span) // 2

    board_rect = pygame.Rect(grid_x, grid_y, grid_span, grid_span)
    plate_pad = max(12, int(cell * 0.70))
    board_plate_rect = pygame.Rect(
        board_rect.x - plate_pad,
        board_rect.y - plate_pad,
        board_rect.w + plate_pad * 2,
        board_rect.h + plate_pad * 2,
    )

    inner = sidebar_rect.inflate(-pad * 2, -pad * 2)
    player_h = max(64, int(inner.h * 0.14))
    mode_h = max(180, int(inner.h * 0.34))
    mode_title_h = max(36, int(mode_h * 0.20))
    option_h = max(56, int((mode_h - mode_title_h - 26) / 2))
    btn_h = max(48, int(inner.h * 0.11))
    exposure_h = max(70, int(inner.h * 0.16))

    player_rect = pygame.Rect(inner.x, inner.y, inner.w, player_h)
    mode_rect = pygame.Rect(inner.x, player_rect.bottom + 12, inner.w, mode_h)

    fast_mode_rect = pygame.Rect(mode_rect.x + 12, mode_rect.y + mode_title_h + 8, mode_rect.w - 24, option_h)
    slow_mode_rect = pygame.Rect(mode_rect.x + 12, fast_mode_rect.bottom + 10, mode_rect.w - 24, option_h)

    restart_rect = pygame.Rect(inner.x, inner.bottom - btn_h, inner.w, btn_h)
    undo_rect = pygame.Rect(inner.x, restart_rect.y - 10 - btn_h, inner.w, btn_h)
    exposure_rect = pygame.Rect(inner.x, undo_rect.y - 10 - exposure_h, inner.w, exposure_h)

    intro_w = min(860, max(520, int(screen_w * 0.68)))
    intro_h = min(640, max(390, int(screen_h * 0.74)))
    intro_rect = pygame.Rect((screen_w - intro_w) // 2, (screen_h - intro_h) // 2, intro_w, intro_h)
    intro_inner = intro_rect.inflate(-24, -24)
    split_gap = 16
    left_w = int(intro_inner.w * 0.56)
    right_w = intro_inner.w - left_w - split_gap
    intro_rules_rect = pygame.Rect(intro_inner.x, intro_inner.y, left_w, intro_inner.h - 62)
    intro_config_rect = pygame.Rect(intro_rules_rect.right + split_gap, intro_inner.y, right_w, intro_inner.h - 62)

    option_gap = 8
    option_h = max(30, int(intro_config_rect.h * 0.095))
    option_w = int((intro_config_rect.w - option_gap) / 2)
    section_gap = max(18, int(option_h * 0.7))
    p1_title_h = 26
    p2_title_h = 26
    p1_y = intro_config_rect.y + 10 + p1_title_h
    p2_y = p1_y + option_h * 2 + option_gap + section_gap + p2_title_h

    intro_p1_human_rect = pygame.Rect(intro_config_rect.x, p1_y, option_w, option_h)
    intro_p1_weak_rect = pygame.Rect(intro_p1_human_rect.right + option_gap, p1_y, option_w, option_h)
    intro_p1_baseline_rect = pygame.Rect(intro_config_rect.x, intro_p1_human_rect.bottom + option_gap, option_w, option_h)
    intro_p1_master_rect = pygame.Rect(
        intro_p1_baseline_rect.right + option_gap,
        intro_p1_human_rect.bottom + option_gap,
        option_w,
        option_h,
    )

    intro_p2_human_rect = pygame.Rect(intro_config_rect.x, p2_y, option_w, option_h)
    intro_p2_weak_rect = pygame.Rect(intro_p2_human_rect.right + option_gap, p2_y, option_w, option_h)
    intro_p2_baseline_rect = pygame.Rect(intro_config_rect.x, intro_p2_human_rect.bottom + option_gap, option_w, option_h)
    intro_p2_master_rect = pygame.Rect(
        intro_p2_baseline_rect.right + option_gap,
        intro_p2_human_rect.bottom + option_gap,
        option_w,
        option_h,
    )

    intro_close_rect = pygame.Rect(intro_rect.centerx - 96, intro_rect.bottom - 52, 192, 40)

    end_w = min(620, max(420, int(screen_w * 0.48)))
    end_h = min(340, max(220, int(screen_h * 0.34)))
    end_rect = pygame.Rect((screen_w - end_w) // 2, (screen_h - end_h) // 2, end_w, end_h)
    end_close_rect = pygame.Rect(end_rect.centerx - 76, end_rect.bottom - 58, 152, 40)

    return UILayout(
        screen_size=(screen_w, screen_h),
        pad=pad,
        top_rect=top_rect,
        board_plate_rect=board_plate_rect,
        board_rect=board_rect,
        sidebar_rect=sidebar_rect,
        player_rect=player_rect,
        mode_rect=mode_rect,
        fast_mode_rect=fast_mode_rect,
        slow_mode_rect=slow_mode_rect,
        exposure_rect=exposure_rect,
        undo_rect=undo_rect,
        restart_rect=restart_rect,
        intro_rect=intro_rect,
        intro_rules_rect=intro_rules_rect,
        intro_config_rect=intro_config_rect,
        intro_p1_human_rect=intro_p1_human_rect,
        intro_p1_weak_rect=intro_p1_weak_rect,
        intro_p1_baseline_rect=intro_p1_baseline_rect,
        intro_p1_master_rect=intro_p1_master_rect,
        intro_p2_human_rect=intro_p2_human_rect,
        intro_p2_weak_rect=intro_p2_weak_rect,
        intro_p2_baseline_rect=intro_p2_baseline_rect,
        intro_p2_master_rect=intro_p2_master_rect,
        intro_close_rect=intro_close_rect,
        end_rect=end_rect,
        end_close_rect=end_close_rect,
        grid_origin=(grid_x, grid_y),
        cell=cell,
        grid_span=grid_span,
    )


def board_coords_from_pos(mouse_pos: tuple[int, int], layout: UILayout) -> tuple[int, int] | None:
    gx, gy = layout.grid_origin
    cell = layout.cell
    x, y = mouse_pos

    col_f = (x - gx) / cell
    row_f = (y - gy) / cell
    col = round(col_f)
    row = round(row_f)

    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        return None
    if abs(col_f - col) > 0.45 or abs(row_f - row) > 0.45:
        return None
    return row, col
