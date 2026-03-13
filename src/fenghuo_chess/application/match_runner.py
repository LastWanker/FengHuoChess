"""Match orchestration for pluggable player sources."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.fenghuo_chess.ai.baselines.heuristic_ai import HeuristicAISource, MasterAISource
from src.fenghuo_chess.ai.baselines.random_ai import RandomAISource
from src.fenghuo_chess.ai.model_ai import ModelAISource
from src.fenghuo_chess.application.logic_api import Action
from src.fenghuo_chess.application.player_source import HumanPlayerSource, PlayerSource
from src.fenghuo_chess.domain.models import GameState

MatchMode = Literal["pvp", "pve", "eve"]
SourceKind = Literal["human", "weak", "baseline", "master"]


@dataclass
class MatchConfig:
    match_mode: MatchMode = "pvp"
    black_source: PlayerSource | None = None
    white_source: PlayerSource | None = None
    human_player: int = 1
    p1_source_kind: SourceKind | None = None
    p2_source_kind: SourceKind | None = None
    with_ui: bool = True


class MatchRunner:
    def __init__(self, config: MatchConfig) -> None:
        self.config = config
        self.black_source, self.white_source = _resolve_sources(config)
        self.config.match_mode = _infer_match_mode(self.black_source, self.white_source)
        if self.config.p1_source_kind is None:
            self.config.p1_source_kind = source_kind_of(self.black_source)
        if self.config.p2_source_kind is None:
            self.config.p2_source_kind = source_kind_of(self.white_source)

    def source_for_player(self, player: int) -> PlayerSource:
        return self.black_source if player == 1 else self.white_source

    def source_kind_for_player(self, player: int) -> SourceKind:
        source = self.source_for_player(player)
        return source_kind_of(source)

    def is_human_turn(self, player: int) -> bool:
        return isinstance(self.source_for_player(player), HumanPlayerSource)

    def has_human_players(self) -> bool:
        return isinstance(self.black_source, HumanPlayerSource) or isinstance(self.white_source, HumanPlayerSource)

    def submit_human_action(self, player: int, action: Action) -> bool:
        source = self.source_for_player(player)
        if not isinstance(source, HumanPlayerSource):
            return False
        source.submit_action(action)
        return True

    def select_action(self, state: GameState, legal_actions: list[Action]) -> Action | None:
        source = self.source_for_player(state.current_player)
        return source.select_action(state, legal_actions)

    def reset_sources(self) -> None:
        for source in (self.black_source, self.white_source):
            if isinstance(source, HumanPlayerSource):
                source.clear_pending()


def build_match_runner(
    match_mode: MatchMode = "pvp",
    *,
    with_ui: bool = True,
    black_source: PlayerSource | None = None,
    white_source: PlayerSource | None = None,
    human_player: int = 1,
    p1_source_kind: SourceKind | None = None,
    p2_source_kind: SourceKind | None = None,
) -> MatchRunner:
    return MatchRunner(
        MatchConfig(
            match_mode=match_mode,
            black_source=black_source,
            white_source=white_source,
            human_player=human_player,
            p1_source_kind=p1_source_kind,
            p2_source_kind=p2_source_kind,
            with_ui=with_ui,
        )
    )


def source_kind_of(source: PlayerSource) -> SourceKind:
    if isinstance(source, HumanPlayerSource):
        return "human"
    if isinstance(source, RandomAISource):
        return "weak"
    if isinstance(source, ModelAISource) or isinstance(source, MasterAISource):
        return "master"
    if isinstance(source, HeuristicAISource):
        return "baseline"
    return "baseline"


def build_source_by_kind(kind: SourceKind) -> PlayerSource:
    if kind == "human":
        return HumanPlayerSource()
    if kind == "weak":
        return RandomAISource()
    if kind == "baseline":
        return HeuristicAISource()
    if kind == "master":
        return _build_master_source()
    raise ValueError(f"Unsupported source kind: {kind}")


def _build_master_source() -> PlayerSource:
    repo_root = Path(__file__).resolve().parents[3]
    default_slow_candidates = [
        repo_root / "artifacts/models/model_tiny_policy_slow_best_black.pt",
        repo_root / "artifacts/models/model_tiny_policy_slow_best.pt",
        repo_root / "artifacts/models/model_tiny_policy_slow_pretrain_v0.pt",
        repo_root / "artifacts/models/model_tiny_policy_slow.pt",
    ]
    default_slow_white_candidates = [
        repo_root / "artifacts/models/model_tiny_policy_slow_best_white.pt",
        repo_root / "artifacts/models/model_tiny_policy_slow_best.pt",
        repo_root / "artifacts/models/model_tiny_policy_slow_pretrain_v0.pt",
        repo_root / "artifacts/models/model_tiny_policy_slow.pt",
    ]
    default_fast_candidates = [
        repo_root / "artifacts/models/model_tiny_policy_fast_best.pt",
        repo_root / "artifacts/models/model_tiny_policy_fast.pt",
    ]
    default_fast_white_candidates = [
        repo_root / "artifacts/models/model_tiny_policy_fast_best_white.pt",
        repo_root / "artifacts/models/model_tiny_policy_fast_best.pt",
        repo_root / "artifacts/models/model_tiny_policy_fast.pt",
    ]
    default_slow = next((str(p) for p in default_slow_candidates if p.exists()), "")
    default_slow_white = next((str(p) for p in default_slow_white_candidates if p.exists()), default_slow)
    default_fast = next((str(p) for p in default_fast_candidates if p.exists()), default_slow)
    default_fast_white = next((str(p) for p in default_fast_white_candidates if p.exists()), default_fast)

    model_path = os.getenv("FENGHUO_MASTER_MODEL_PATH", "").strip() or default_slow
    model_fast_path = os.getenv("FENGHUO_MASTER_MODEL_FAST_PATH", "").strip() or default_fast or model_path
    model_slow_path = os.getenv("FENGHUO_MASTER_MODEL_SLOW_PATH", "").strip() or default_slow or model_path
    model_black_path = os.getenv("FENGHUO_MASTER_MODEL_BLACK_PATH", "").strip() or default_slow
    model_white_path = os.getenv("FENGHUO_MASTER_MODEL_WHITE_PATH", "").strip() or default_slow_white
    model_black_fast_path = os.getenv("FENGHUO_MASTER_MODEL_BLACK_FAST_PATH", "").strip() or default_fast
    model_white_fast_path = os.getenv("FENGHUO_MASTER_MODEL_WHITE_FAST_PATH", "").strip() or default_fast_white
    model_black_slow_path = os.getenv("FENGHUO_MASTER_MODEL_BLACK_SLOW_PATH", "").strip() or model_black_path
    model_white_slow_path = os.getenv("FENGHUO_MASTER_MODEL_WHITE_SLOW_PATH", "").strip() or model_white_path
    device = os.getenv("FENGHUO_MASTER_DEVICE", "cuda").strip() or "cuda"
    try:
        return ModelAISource(
            model_path=model_path,
            model_fast_path=model_fast_path,
            model_slow_path=model_slow_path,
            model_black_path=model_black_path,
            model_white_path=model_white_path,
            model_black_fast_path=model_black_fast_path,
            model_white_fast_path=model_white_fast_path,
            model_black_slow_path=model_black_slow_path,
            model_white_slow_path=model_white_slow_path,
            device=device,
        )
    except Exception as exc:
        print(f"[master-ai] fallback to baseline: {exc}")
        return HeuristicAISource()


def _resolve_sources(config: MatchConfig) -> tuple[PlayerSource, PlayerSource]:
    if config.human_player not in {1, 2}:
        raise ValueError(f"Unsupported human_player: {config.human_player}")

    if config.black_source is not None and config.white_source is not None:
        return config.black_source, config.white_source

    if config.p1_source_kind is not None and config.p2_source_kind is not None:
        return build_source_by_kind(config.p1_source_kind), build_source_by_kind(config.p2_source_kind)

    if config.match_mode == "pvp":
        return HumanPlayerSource(), HumanPlayerSource()
    if config.match_mode == "pve":
        if config.human_player == 2:
            return HeuristicAISource(), HumanPlayerSource()
        return HumanPlayerSource(), HeuristicAISource()
    if config.match_mode == "eve":
        return HeuristicAISource(), HeuristicAISource()

    raise ValueError(f"Unsupported match mode: {config.match_mode}")


def _infer_match_mode(black: PlayerSource, white: PlayerSource) -> MatchMode:
    humans = int(isinstance(black, HumanPlayerSource)) + int(isinstance(white, HumanPlayerSource))
    if humans == 2:
        return "pvp"
    if humans == 1:
        return "pve"
    return "eve"
