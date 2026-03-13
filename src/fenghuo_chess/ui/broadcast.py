"""Smooth side-in broadcast animation."""

import pygame

from src.fenghuo_chess.constants.gameplay import MAX_BROADCAST_QUEUE


def _font(size: int, bold: bool = True) -> pygame.font.Font:
    return pygame.font.SysFont(["Microsoft YaHei UI", "SimHei", "Segoe UI"], size, bold=bold)


def _ease_out_cubic(t: float) -> float:
    return 1 - (1 - t) ** 3


def _ease_in_out(t: float) -> float:
    if t < 0.5:
        return 4 * t * t * t
    return 1 - ((-2 * t + 2) ** 3) / 2


class TextBroadcast:
    def __init__(self, text: str, color: tuple[int, int, int]):
        self.text = text
        self.color = color
        self.active = False
        self.start_time_ms = 0

        self.in_ms = 360
        self.hold_ms = 900
        self.out_ms = 420
        self.total_ms = self.in_ms + self.hold_ms + self.out_ms

    def start(self) -> None:
        self.start_time_ms = pygame.time.get_ticks()
        self.active = True

    def update(self) -> None:
        if not self.active:
            return
        elapsed = pygame.time.get_ticks() - self.start_time_ms
        if elapsed >= self.total_ms:
            self.active = False

    def draw(self, surface: pygame.Surface, screen_w: int, screen_h: int) -> None:
        if not self.active:
            return

        elapsed = pygame.time.get_ticks() - self.start_time_ms
        font_size = max(20, min(34, int(screen_h * 0.04)))
        font = _font(font_size)
        text_surface = font.render(self.text, True, self.color)
        tw, th = text_surface.get_size()

        card_w = tw + 56
        card_h = th + 22
        y = max(16, int(screen_h * 0.06))
        target_x = (screen_w - card_w) // 2
        start_x = -card_w - 24
        end_x = screen_w + 24

        if elapsed <= self.in_ms:
            t = _ease_out_cubic(elapsed / self.in_ms)
            x = int(start_x + (target_x - start_x) * t)
            alpha = int(220 * t)
        elif elapsed <= self.in_ms + self.hold_ms:
            x = target_x
            alpha = 220
        else:
            t = (elapsed - self.in_ms - self.hold_ms) / self.out_ms
            t = _ease_in_out(max(0.0, min(1.0, t)))
            x = int(target_x + (end_x - target_x) * t)
            alpha = int(220 * (1 - t))

        card = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        pygame.draw.rect(card, (54, 51, 48, alpha), card.get_rect(), border_radius=13)
        pygame.draw.rect(card, (250, 245, 236, min(alpha, 170)), card.get_rect(), 1, border_radius=13)
        surface.blit(card, (x, y))

        text_surface.set_alpha(alpha)
        surface.blit(text_surface, (x + (card_w - tw) // 2, y + (card_h - th) // 2))


class BroadcastOverlay:
    def __init__(self) -> None:
        self.queue: list[tuple[str, tuple[int, int, int]]] = []
        self.current: TextBroadcast | None = None

    def enqueue(self, text: str, color: tuple[int, int, int]) -> None:
        if len(self.queue) < MAX_BROADCAST_QUEUE:
            self.queue.append((text, color))

    def enqueue_many(self, items: list[tuple[str, tuple[int, int, int]]]) -> None:
        for text, color in items:
            self.enqueue(text, color)

    def update(self, _screen_w: int, _screen_h: int) -> None:
        if self.current is None and self.queue:
            text, color = self.queue.pop(0)
            self.current = TextBroadcast(text, color)
            self.current.start()

        if self.current is None:
            return

        self.current.update()
        if not self.current.active:
            self.current = None

    def draw(self, surface: pygame.Surface, screen_w: int, screen_h: int) -> None:
        if self.current is not None:
            self.current.draw(surface, screen_w, screen_h)
