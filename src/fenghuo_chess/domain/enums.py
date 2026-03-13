"""Domain enums."""

from enum import IntEnum, StrEnum


class Player(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2


class Stage(IntEnum):
    PHASE_1 = 1
    PHASE_2 = 2
    PHASE_3 = 3
    PHASE_4 = 4


class GameMode(StrEnum):
    FAST = "fast"
    SLOW = "slow"

