"""Game orchestration controller."""

import time

from src.fenghuo_chess.application.logic_api import Action, apply_action, legal_actions
from src.fenghuo_chess.application.match_runner import MatchMode, MatchRunner, SourceKind, build_match_runner
from src.fenghuo_chess.constants.colors import WHITE
from src.fenghuo_chess.constants.gameplay import CONFIRMATION_DELAY, MAX_BROADCAST_QUEUE, MODE_DESCRIPTIONS
from src.fenghuo_chess.domain.events import LogicEvent
from src.fenghuo_chess.domain.models import GameState, create_initial_state
from src.fenghuo_chess.services.exposure_service import ExposureService
from src.fenghuo_chess.services.history_service import HistoryService
from src.fenghuo_chess.services.mode_service import set_mode


class GameController:
    def __init__(self, match_mode: MatchMode = "pvp", with_ui: bool = True, human_player: int = 1) -> None:
        self.state: GameState = create_initial_state()
        self.history = HistoryService()
        self.history.reset(self.state)
        self.exposure_service = ExposureService()
        self.match_runner: MatchRunner = build_match_runner(match_mode, with_ui=with_ui, human_player=human_player)
        self._pending_events: list[LogicEvent] = []
        self.state.intro_p1_source = "human"
        self.state.intro_p2_source = "baseline"

    @property
    def match_mode(self) -> MatchMode:
        return self.match_runner.config.match_mode

    @property
    def p1_source_kind(self) -> SourceKind:
        return self.match_runner.source_kind_for_player(1)

    @property
    def p2_source_kind(self) -> SourceKind:
        return self.match_runner.source_kind_for_player(2)

    @property
    def human_player(self) -> int:
        return self.match_runner.config.human_player

    def configure_match(self, match_mode: MatchMode, with_ui: bool = True, human_player: int = 1) -> None:
        self.match_runner = build_match_runner(match_mode, with_ui=with_ui, human_player=human_player)
        self.state.intro_p1_source = self.p1_source_kind
        self.state.intro_p2_source = self.p2_source_kind

    def configure_players(self, p1_source_kind: SourceKind, p2_source_kind: SourceKind, with_ui: bool = True) -> None:
        self.match_runner = build_match_runner(
            with_ui=with_ui,
            p1_source_kind=p1_source_kind,
            p2_source_kind=p2_source_kind,
        )
        self.state.intro_p1_source = p1_source_kind
        self.state.intro_p2_source = p2_source_kind

    def broadcast_text(self, text: str, color: tuple[int, int, int]) -> None:
        self._push_event(LogicEvent.broadcast(text, color))

    def consume_events(self) -> list[LogicEvent]:
        events = self._pending_events[:]
        self._pending_events.clear()
        return events

    def consume_broadcasts(self) -> list[tuple[str, tuple[int, int, int]]]:
        out: list[tuple[str, tuple[int, int, int]]] = []
        for event in self.consume_events():
            if event.kind != "BROADCAST":
                continue
            text = str(event.payload.get("text", ""))
            raw_color = event.payload.get("color", WHITE)
            if isinstance(raw_color, (tuple, list)) and len(raw_color) == 3:
                color = (int(raw_color[0]), int(raw_color[1]), int(raw_color[2]))
            else:
                color = WHITE
            out.append((text, color))
        return out

    def reset_game(self) -> None:
        with_ui = self.match_runner.config.with_ui
        p1_kind = self.p1_source_kind
        p2_kind = self.p2_source_kind
        self.state = create_initial_state()
        self.history.reset(self.state)
        self._pending_events.clear()
        self.match_runner.reset_sources()
        self.state.intro_p1_source = p1_kind
        self.state.intro_p2_source = p2_kind
        self.configure_players(p1_source_kind=p1_kind, p2_source_kind=p2_kind, with_ui=with_ui)

    def undo(self) -> bool:
        snapshot = self.history.undo()
        if snapshot is None:
            return False
        self.state = snapshot
        return True

    def change_mode(self, mode: str) -> bool:
        ok = set_mode(self.state, mode)
        if ok:
            self.broadcast_text(MODE_DESCRIPTIONS[mode], WHITE)
        return ok

    def update_restart_confirmation_timeout(self) -> None:
        if self.state.restart_button_state != "confirm":
            return
        if time.time() - self.state.restart_button_press_time > CONFIRMATION_DELAY:
            self.state.restart_button_state = "normal"

    def click_restart(self) -> None:
        now = time.time()
        if self.state.restart_button_state == "normal":
            self.state.restart_button_state = "confirm"
            self.state.restart_button_press_time = now
            return

        if now - self.state.restart_button_press_time <= CONFIRMATION_DELAY:
            self.reset_game()
        else:
            self.state.restart_button_press_time = now

    def make_move(self, row: int, col: int) -> bool:
        if self.state.game_over:
            return False

        accepted = self.match_runner.submit_human_action(self.state.current_player, Action(row, col))
        if not accepted:
            return False
        return self.step_turn()

    def step_turn(self) -> bool:
        if self.state.game_over:
            return False

        legal = legal_actions(self.state)
        if not legal:
            return False

        action = self.match_runner.select_action(self.state, legal)
        if action is None:
            return False

        step = apply_action(self.state, action, exposure_service=self.exposure_service)
        self._push_events(step.events)
        if not step.ok:
            return False

        self.history.push(self.state)
        return True

    def _push_events(self, events: list[LogicEvent]) -> None:
        for event in events:
            self._push_event(event)

    def _push_event(self, event: LogicEvent) -> None:
        if event.kind == "BROADCAST":
            pending_broadcasts = sum(1 for item in self._pending_events if item.kind == "BROADCAST")
            if pending_broadcasts >= MAX_BROADCAST_QUEUE:
                return
        self._pending_events.append(event)
