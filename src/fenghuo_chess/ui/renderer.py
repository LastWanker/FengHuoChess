"""Responsive renderer using picture assets and a clean glass UI."""

from pathlib import Path

import pygame

from src.fenghuo_chess.constants.gameplay import BOARD_SIZE, INTRO_TEXT, STAGE_DESCRIPTIONS
from src.fenghuo_chess.domain.models import GameState
from src.fenghuo_chess.ui.broadcast import BroadcastOverlay
from src.fenghuo_chess.ui.input_mapper import UILayout


def _font(size: int, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont(["Microsoft YaHei UI", "SimHei", "Segoe UI"], size, bold=bold)


def _cover_size(img_w: int, img_h: int, dst_w: int, dst_h: int) -> tuple[int, int]:
    scale = max(dst_w / max(1, img_w), dst_h / max(1, img_h))
    return int(img_w * scale), int(img_h * scale)


class GameRenderer:
    def __init__(self) -> None:
        root = Path(__file__).resolve().parents[3]
        pic = root / "pictures"
        self.bg_img = pygame.image.load(str(pic / "背景图.jpg")).convert()
        self.board_img = pygame.image.load(str(pic / "木头棋盘.png")).convert()
        self.black_piece_img = pygame.image.load(str(pic / "黑子.png")).convert_alpha()
        self.white_piece_img = pygame.image.load(str(pic / "白子.png")).convert_alpha()

        self._bg_cache: dict[tuple[int, int], pygame.Surface] = {}
        self._board_cache: dict[tuple[int, int], pygame.Surface] = {}
        self._piece_cache: dict[tuple[str, int], pygame.Surface] = {}

    def draw(
        self,
        surface: pygame.Surface,
        state: GameState,
        layout: UILayout,
        mouse_pos: tuple[int, int],
        broadcasts: BroadcastOverlay,
    ) -> None:
        self._draw_background(surface, layout)
        self._draw_top_bar(surface, state, layout)
        self._draw_board(surface, state, layout)
        self._draw_sidebar(surface, state, layout, mouse_pos)

        if state.game_over and not state.end_overlay_closed:
            self._draw_end_overlay(surface, state, layout, mouse_pos)
        if state.show_intro:
            self._draw_intro_overlay(surface, state, layout, mouse_pos)

        broadcasts.draw(surface, *layout.screen_size)

    def _draw_background(self, surface: pygame.Surface, layout: UILayout) -> None:
        w, h = layout.screen_size
        key = (w, h)
        bg = self._bg_cache.get(key)
        if bg is None:
            tw, th = _cover_size(self.bg_img.get_width(), self.bg_img.get_height(), w, h)
            scaled = pygame.transform.smoothscale(self.bg_img, (tw, th))
            bg = pygame.Surface((w, h))
            bg.blit(scaled, ((w - tw) // 2, (h - th) // 2))
            tint = pygame.Surface((w, h), pygame.SRCALPHA)
            tint.fill((14, 14, 14, 106))
            bg.blit(tint, (0, 0))
            self._bg_cache[key] = bg
        surface.blit(bg, (0, 0))

    def _draw_top_bar(self, surface: pygame.Surface, state: GameState, layout: UILayout) -> None:
        rect = layout.top_rect
        self._glass_card(surface, rect, alpha=168, border_alpha=78, radius=16)

        title_font = _font(max(18, int(rect.h * 0.34)), bold=True)
        desc_font = _font(max(12, int(rect.h * 0.22)))
        title = title_font.render(state.message, True, (239, 236, 231))
        desc = desc_font.render(STAGE_DESCRIPTIONS.get(state.stage, ""), True, (198, 192, 184))
        surface.blit(title, (rect.x + 16, rect.y + 8))
        surface.blit(desc, (rect.x + 16, rect.bottom - desc.get_height() - 10))

    def _draw_board(self, surface: pygame.Surface, state: GameState, layout: UILayout) -> None:
        self._draw_board_plate(surface, layout)
        self._draw_stage_highlight(surface, state, layout)
        self._draw_grid(surface, layout)
        self._draw_exposure_marks(surface, state, layout)
        self._draw_pieces(surface, state, layout)

    def _draw_board_plate(self, surface: pygame.Surface, layout: UILayout) -> None:
        rect = layout.board_plate_rect
        key = (rect.w, rect.h)
        board_tex = self._board_cache.get(key)
        if board_tex is None:
            tw, th = _cover_size(self.board_img.get_width(), self.board_img.get_height(), rect.w, rect.h)
            scaled = pygame.transform.smoothscale(self.board_img, (tw, th))
            board_tex = pygame.Surface((rect.w, rect.h))
            board_tex.blit(scaled, ((rect.w - tw) // 2, (rect.h - th) // 2))
            shade = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
            shade.fill((245, 238, 225, 40))
            board_tex.blit(shade, (0, 0))
            self._board_cache[key] = board_tex

        radius = 14
        shadow = pygame.Surface((rect.w + 8, rect.h + 8), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, 72), shadow.get_rect(), border_radius=radius + 2)
        surface.blit(shadow, (rect.x - 4, rect.y - 1))

        clipped = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        clipped.blit(board_tex, (0, 0))
        mask = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=radius)
        clipped.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        surface.blit(clipped, rect.topleft)
        pygame.draw.rect(surface, (108, 90, 62), rect, 2, border_radius=radius)

    def _draw_stage_highlight(self, surface: pygame.Surface, state: GameState, layout: UILayout) -> None:
        center = BOARD_SIZE // 2
        overlays: list[tuple[int, int, int, int, tuple[int, int, int, int]]] = []
        if state.stage == 1:
            overlays.append((center - 1, center - 1, 3, 3, (40, 40, 40, 34)))
        elif state.stage == 2:
            overlays.append((center - 2, center - 2, 5, 5, (40, 40, 40, 24)))
            overlays.append((center - 1, center - 1, 3, 3, (22, 22, 22, 36)))
        elif state.stage == 3:
            overlays.append((center - 4, center - 4, 9, 9, (40, 40, 40, 20)))
            overlays.append((center - 2, center - 2, 5, 5, (22, 22, 22, 34)))
        else:
            overlays.append((center - 4, center - 4, 9, 9, (22, 22, 22, 32)))

        gx, gy = layout.grid_origin
        cell = layout.cell
        for sr, sc, w, h, color in overlays:
            x = gx + sc * cell - cell // 2
            y = gy + sr * cell - cell // 2
            mask = pygame.Surface((w * cell, h * cell), pygame.SRCALPHA)
            pygame.draw.rect(mask, color, mask.get_rect(), border_radius=6)
            surface.blit(mask, (x, y))

    def _draw_grid(self, surface: pygame.Surface, layout: UILayout) -> None:
        gx, gy = layout.grid_origin
        span = layout.grid_span
        cell = layout.cell
        light = (124, 93, 57)
        deep = (94, 67, 39)

        for i in range(BOARD_SIZE):
            width = 2 if i in {0, BOARD_SIZE - 1, BOARD_SIZE // 2} else 1
            color = deep if width == 2 else light
            pygame.draw.line(surface, color, (gx, gy + i * cell), (gx + span, gy + i * cell), width)
            pygame.draw.line(surface, color, (gx + i * cell, gy), (gx + i * cell, gy + span), width)

        star_idx = [3, 7, 11]
        radius = max(2, cell // 8)
        for r in star_idx:
            for c in star_idx:
                pygame.draw.circle(surface, (88, 62, 38), (gx + c * cell, gy + r * cell), radius)

    def _draw_pieces(self, surface: pygame.Surface, state: GameState, layout: UILayout) -> None:
        gx, gy = layout.grid_origin
        cell = layout.cell
        piece_size = max(14, cell - 6)
        offset = piece_size // 2

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = int(state.board[r, c])
                if piece == 0:
                    continue

                center = (gx + c * cell, gy + r * cell)
                shadow_size = piece_size + 6
                shadow = pygame.Surface((shadow_size, shadow_size), pygame.SRCALPHA)
                for i in range(4):
                    pad = i * 2
                    alpha = max(12, 56 - i * 12)
                    pygame.draw.ellipse(
                        shadow,
                        (0, 0, 0, alpha),
                        pygame.Rect(pad, pad + 1, shadow_size - pad * 2, shadow_size - pad * 2 - 2),
                    )
                surface.blit(
                    shadow,
                    (center[0] - shadow_size // 2 + 1, center[1] - shadow_size // 2 + 1),
                )

                key = ("b" if piece == 1 else "w", piece_size)
                sprite = self._piece_cache.get(key)
                if sprite is None:
                    src = self.black_piece_img if piece == 1 else self.white_piece_img
                    sprite = pygame.transform.smoothscale(src, (piece_size, piece_size))
                    # Fine outline only; no extra gloss.
                    outline = pygame.Surface((piece_size, piece_size), pygame.SRCALPHA)
                    border_color = (48, 48, 48, 180) if piece == 1 else (150, 142, 132, 170)
                    pygame.draw.circle(outline, border_color, (piece_size // 2, piece_size // 2), piece_size // 2 - 1, 1)
                    sprite.blit(outline, (0, 0))
                    self._piece_cache[key] = sprite

                surface.blit(sprite, (center[0] - offset, center[1] - offset))

    def _draw_exposure_marks(self, surface: pygame.Surface, state: GameState, layout: UILayout) -> None:
        gx, gy = layout.grid_origin
        cell = layout.cell
        for r, c in state.exposure_positions:
            pygame.draw.circle(surface, (196, 86, 84), (gx + c * cell, gy + r * cell), max(5, cell // 3), 2)
        for r, c in state.exposure_markers:
            pygame.draw.circle(surface, (213, 113, 98), (gx + c * cell, gy + r * cell), max(4, cell // 5), 2)

    def _draw_sidebar(self, surface: pygame.Surface, state: GameState, layout: UILayout, mouse_pos: tuple[int, int]) -> None:
        self._glass_card(surface, layout.sidebar_rect, alpha=176, border_alpha=84, radius=16)

        self._draw_player_card(surface, state, layout)
        self._draw_mode_card(surface, state, layout, mouse_pos)
        self._draw_exposure_banner(surface, state, layout)
        self._draw_actions(surface, state, layout, mouse_pos)

    def _draw_player_card(self, surface: pygame.Surface, state: GameState, layout: UILayout) -> None:
        rect = layout.player_rect
        self._glass_card(surface, rect, alpha=140, border_alpha=62, radius=12)
        label = "黑棋执子" if state.current_player == 1 else "白棋执子"
        font = _font(max(16, rect.h // 3), bold=True)
        text = font.render(label, True, (236, 232, 226))
        surface.blit(text, (rect.x + 14, rect.centery - text.get_height() // 2))

    def _draw_mode_card(self, surface: pygame.Surface, state: GameState, layout: UILayout, mouse_pos: tuple[int, int]) -> None:
        rect = layout.mode_rect
        self._glass_card(surface, rect, alpha=140, border_alpha=62, radius=12)

        title_font = _font(max(15, rect.h // 10), bold=True)
        body_font = _font(max(13, rect.h // 12))
        title = title_font.render("游戏模式", True, (235, 231, 225))
        surface.blit(title, (rect.x + 12, rect.y + 10))
        if state.mode_locked:
            lock = body_font.render("阶段3后锁定", True, (205, 138, 132))
            surface.blit(lock, (rect.x + 12 + title.get_width() + 10, rect.y + 12))

        self._draw_mode_option(
            surface=surface,
            rect=layout.fast_mode_rect,
            hovered=layout.fast_mode_rect.collidepoint(mouse_pos),
            selected=state.game_mode == "fast",
            title="剑拔弩张",
            desc="阶段3：四连获胜",
            title_font=title_font,
            desc_font=body_font,
        )
        self._draw_mode_option(
            surface=surface,
            rect=layout.slow_mode_rect,
            hovered=layout.slow_mode_rect.collidepoint(mouse_pos),
            selected=state.game_mode == "slow",
            title="步步为营",
            desc="阶段3：五连获胜",
            title_font=title_font,
            desc_font=body_font,
        )

    def _draw_mode_option(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        hovered: bool,
        selected: bool,
        title: str,
        desc: str,
        title_font: pygame.font.Font,
        desc_font: pygame.font.Font,
    ) -> None:
        if selected:
            bg = (120, 108, 92, 156)
            border = (196, 177, 150, 120)
        elif hovered:
            bg = (103, 96, 86, 132)
            border = (166, 154, 136, 90)
        else:
            bg = (86, 82, 76, 118)
            border = (148, 138, 124, 78)

        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(panel, bg, panel.get_rect(), border_radius=10)
        pygame.draw.rect(panel, border, panel.get_rect(), 1, border_radius=10)
        surface.blit(panel, rect.topleft)

        dot = (189, 173, 146) if selected else (134, 128, 118)
        pygame.draw.circle(surface, dot, (rect.x + 12, rect.centery), 4)

        title_text = title_font.render(title, True, (232, 227, 220))
        desc_text = desc_font.render(desc, True, (200, 194, 186))
        surface.blit(title_text, (rect.x + 22, rect.y + 6))
        surface.blit(desc_text, (rect.x + 22, rect.bottom - desc_text.get_height() - 6))

    def _draw_exposure_banner(self, surface: pygame.Surface, state: GameState, layout: UILayout) -> None:
        rect = layout.exposure_rect
        self._glass_card(surface, rect, alpha=124, border_alpha=56, radius=10)
        font = _font(max(12, rect.h // 4))
        text = state.exposure_message if state.exposure_message else "—"
        color = (210, 147, 142) if state.exposure_message else (180, 176, 169)
        lines = self._wrap_text(text, font, rect.w - 20, max_lines=2)
        total_h = len(lines) * font.get_height() + (len(lines) - 1) * 2
        y = rect.y + (rect.h - total_h) // 2
        for line in lines:
            rendered = font.render(line, True, color)
            surface.blit(rendered, (rect.x + 10, y))
            y += font.get_height() + 2

    def _draw_actions(self, surface: pygame.Surface, state: GameState, layout: UILayout, mouse_pos: tuple[int, int]) -> None:
        btn_font = _font(max(15, layout.undo_rect.h // 2), bold=True)
        self._draw_button(surface, layout.undo_rect, "悔 棋", btn_font, layout.undo_rect.collidepoint(mouse_pos), danger=False)
        if state.restart_button_state == "confirm":
            label = "真的重开？"
            danger = True
        else:
            label = "重 开"
            danger = False
        self._draw_button(
            surface, layout.restart_rect, label, btn_font, layout.restart_rect.collidepoint(mouse_pos), danger=danger
        )

    def _draw_button(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        text: str,
        font: pygame.font.Font,
        hovered: bool,
        danger: bool,
    ) -> None:
        if danger:
            bg = (145, 100, 96, 210) if hovered else (127, 91, 88, 198)
            border = (192, 144, 139, 112)
        else:
            bg = (112, 108, 102, 198) if hovered else (98, 94, 90, 186)
            border = (163, 153, 138, 94)

        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(panel, bg, panel.get_rect(), border_radius=11)
        pygame.draw.rect(panel, border, panel.get_rect(), 1, border_radius=11)
        surface.blit(panel, rect.topleft)

        txt = font.render(text, True, (241, 236, 228))
        surface.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

    def _draw_intro_overlay(
        self, surface: pygame.Surface, state: GameState, layout: UILayout, mouse_pos: tuple[int, int]
    ) -> None:
        w, h = layout.screen_size
        veil = pygame.Surface((w, h), pygame.SRCALPHA)
        veil.fill((12, 12, 12, 150))
        surface.blit(veil, (0, 0))

        rect = layout.intro_rect
        self._glass_card(surface, rect, alpha=220, border_alpha=88, radius=18, fill=(45, 43, 40))

        title_font = _font(max(24, rect.h // 14), bold=True)
        body_font = _font(max(14, rect.h // 30))
        title = title_font.render("锋火 三阶段战略棋", True, (236, 232, 226))
        surface.blit(title, (rect.x + 24, rect.y + 20))

        y = rect.y + 72
        line_h = max(20, int(rect.h * 0.048))
        for line in INTRO_TEXT:
            t = body_font.render(line, True, (208, 202, 194))
            if y + line_h > layout.intro_rules_rect.bottom:
                break
            surface.blit(t, (layout.intro_rules_rect.x + 8, y))
            y += line_h

        self._glass_card(surface, layout.intro_config_rect, alpha=154, border_alpha=72, radius=12, fill=(52, 49, 46))
        option_font = _font(max(13, rect.h // 30), bold=True)
        side_font = _font(max(14, rect.h // 28), bold=True)

        p1_title = side_font.render("P1 黑子", True, (232, 227, 220))
        p2_title = side_font.render("P2 白子", True, (232, 227, 220))
        surface.blit(p1_title, (layout.intro_p1_human_rect.x + 2, layout.intro_p1_human_rect.y - p1_title.get_height() - 6))
        surface.blit(p2_title, (layout.intro_p2_human_rect.x + 2, layout.intro_p2_human_rect.y - p2_title.get_height() - 6))

        self._draw_intro_option(
            surface,
            layout.intro_p1_human_rect,
            "人类",
            selected=state.intro_p1_source == "human",
            hovered=layout.intro_p1_human_rect.collidepoint(mouse_pos),
            enabled=True,
            font=option_font,
        )
        self._draw_intro_option(
            surface,
            layout.intro_p1_weak_rect,
            "菜鸟AI",
            selected=state.intro_p1_source == "weak",
            hovered=layout.intro_p1_weak_rect.collidepoint(mouse_pos),
            enabled=True,
            font=option_font,
        )
        self._draw_intro_option(
            surface,
            layout.intro_p1_baseline_rect,
            "baseline",
            selected=state.intro_p1_source == "baseline",
            hovered=layout.intro_p1_baseline_rect.collidepoint(mouse_pos),
            enabled=True,
            font=option_font,
        )
        self._draw_intro_option(
            surface,
            layout.intro_p1_master_rect,
            "大师AI",
            selected=state.intro_p1_source == "master",
            hovered=layout.intro_p1_master_rect.collidepoint(mouse_pos),
            enabled=True,
            font=option_font,
        )

        self._draw_intro_option(
            surface,
            layout.intro_p2_human_rect,
            "人类",
            selected=state.intro_p2_source == "human",
            hovered=layout.intro_p2_human_rect.collidepoint(mouse_pos),
            enabled=True,
            font=option_font,
        )
        self._draw_intro_option(
            surface,
            layout.intro_p2_weak_rect,
            "菜鸟AI",
            selected=state.intro_p2_source == "weak",
            hovered=layout.intro_p2_weak_rect.collidepoint(mouse_pos),
            enabled=True,
            font=option_font,
        )
        self._draw_intro_option(
            surface,
            layout.intro_p2_baseline_rect,
            "baseline",
            selected=state.intro_p2_source == "baseline",
            hovered=layout.intro_p2_baseline_rect.collidepoint(mouse_pos),
            enabled=True,
            font=option_font,
        )
        self._draw_intro_option(
            surface,
            layout.intro_p2_master_rect,
            "大师AI",
            selected=state.intro_p2_source == "master",
            hovered=layout.intro_p2_master_rect.collidepoint(mouse_pos),
            enabled=True,
            font=option_font,
        )

        btn_font = _font(max(16, rect.h // 18), bold=True)
        self._draw_button(
            surface,
            layout.intro_close_rect,
            "开始游戏",
            btn_font,
            layout.intro_close_rect.collidepoint(mouse_pos),
            danger=False,
        )

    @staticmethod
    def _draw_intro_option(
        surface: pygame.Surface,
        rect: pygame.Rect,
        label: str,
        *,
        selected: bool,
        hovered: bool,
        enabled: bool,
        font: pygame.font.Font,
    ) -> None:
        if not enabled:
            bg = (86, 84, 80, 96)
            border = (132, 126, 118, 80)
            color = (154, 149, 142)
        elif selected:
            bg = (124, 112, 96, 180)
            border = (204, 186, 160, 130)
            color = (243, 238, 230)
        elif hovered:
            bg = (100, 94, 86, 150)
            border = (171, 160, 142, 102)
            color = (228, 223, 215)
        else:
            bg = (88, 84, 80, 130)
            border = (154, 145, 130, 92)
            color = (218, 212, 202)

        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(panel, bg, panel.get_rect(), border_radius=9)
        pygame.draw.rect(panel, border, panel.get_rect(), 1, border_radius=9)
        surface.blit(panel, rect.topleft)

        text = font.render(label, True, color)
        surface.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))

    def _draw_end_overlay(self, surface: pygame.Surface, state: GameState, layout: UILayout, mouse_pos: tuple[int, int]) -> None:
        w, h = layout.screen_size
        veil = pygame.Surface((w, h), pygame.SRCALPHA)
        veil.fill((10, 10, 10, 132))
        surface.blit(veil, (0, 0))

        rect = layout.end_rect
        self._glass_card(surface, rect, alpha=220, border_alpha=88, radius=16, fill=(45, 43, 40))

        font = _font(max(24, rect.h // 4), bold=True)
        if state.winner == 0:
            msg = "游戏结束，平局"
        else:
            side = "黑棋" if state.winner == 1 else "白棋"
            msg = f"玩家{state.winner}（{side}）获胜"
        text = font.render(msg, True, (238, 232, 226))
        surface.blit(text, (rect.centerx - text.get_width() // 2, rect.y + 50))

        btn_font = _font(max(16, rect.h // 7), bold=True)
        self._draw_button(
            surface,
            layout.end_close_rect,
            "关闭",
            btn_font,
            layout.end_close_rect.collidepoint(mouse_pos),
            danger=False,
        )

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1] + "…"

    @staticmethod
    def _wrap_text(text: str, font: pygame.font.Font, max_width: int, max_lines: int = 2) -> list[str]:
        if not text:
            return [""]
        lines: list[str] = []
        current = ""
        for ch in text:
            candidate = current + ch
            if font.size(candidate)[0] <= max_width:
                current = candidate
                continue
            if len(lines) == max_lines - 1:
                # Last line: clamp and append ellipsis.
                tail = current
                while tail and font.size(tail + "…")[0] > max_width:
                    tail = tail[:-1]
                lines.append((tail if tail else "") + "…")
                return lines

            lines.append(current if current else ch)
            current = ch if current else ""
            while current and font.size(current)[0] > max_width:
                current = current[:-1]

        if current:
            lines.append(current)
        if not lines:
            return [text[:1]]
        return lines[:max_lines]

    @staticmethod
    def _glass_card(
        surface: pygame.Surface,
        rect: pygame.Rect,
        alpha: int,
        border_alpha: int,
        radius: int,
        fill: tuple[int, int, int] = (56, 53, 49),
    ) -> None:
        card = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(card, (*fill, alpha), card.get_rect(), border_radius=radius)
        edge_alpha = min(220, border_alpha + 42)
        pygame.draw.rect(card, (252, 247, 238, edge_alpha), card.get_rect(), 1, border_radius=radius)
        surface.blit(card, rect.topleft)
