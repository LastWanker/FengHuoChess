"""Self-play league round for model promotion decisions."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

from src.fenghuo_chess.ai.baselines.heuristic_ai import HeuristicAISource
from src.fenghuo_chess.ai.baselines.random_ai import RandomAISource
from src.fenghuo_chess.ai.model_ai import ModelAISource
from src.fenghuo_chess.ai.model_v3 import ModelAISourceV3
from src.fenghuo_chess.ai.selfplay_runner import SelfPlaySummary, run_selfplay
from src.fenghuo_chess.application.match_runner import build_match_runner


def _supports_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    stream = getattr(sys, "stdout", None)
    return bool(stream and hasattr(stream, "isatty") and stream.isatty())


_COLOR_ENABLED = _supports_color()


def _paint(text: str, code: str) -> str:
    if not _COLOR_ENABLED:
        return text
    return f"\033[{code}m{text}\033[0m"


def _status_tag(name: str, passed: bool) -> str:
    return _paint(f"{name}=PASS", "32") if passed else _paint(f"{name}=FAIL", "31")


def _promotion_tag(promoted: bool) -> str:
    return _paint("promoted=True", "32") if promoted else _paint("promoted=False", "33")


def _first_player_dominant(stats: "ArenaStats") -> bool:
    first_player_wins = stats.black_round.p1_wins + stats.white_round.p1_wins
    return stats.draws == 0 and first_player_wins == stats.games


def _render_progress(prefix: str, done: int, total: int, extra: str = "") -> None:
    width = 28
    safe_total = max(1, total)
    ratio = min(1.0, max(0.0, done / safe_total))
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    text = f"\r{prefix} [{bar}] {done}/{total} ({ratio * 100:5.1f}%)"
    if extra:
        text += f" {extra}"
    sys.stdout.write(text)
    sys.stdout.flush()


def _end_progress_line() -> None:
    sys.stdout.write("\n")
    sys.stdout.flush()


@dataclass
class LeagueConfig:
    promotion_threshold: float = 0.50
    arena_games: int = 200
    selfplay_games: int = 400
    game_mode: str = "slow"
    device: str = "cuda"
    opponent_pool: dict[str, float] = field(default_factory=lambda: {"teacher": 0.5, "old_best": 0.5})
    trace_output_path: str = "artifacts/datasets/league_selfplay_round.jsonl"
    append_trace: bool = False
    auto_promote: bool = True
    archive_previous_best: bool = True
    random_seed: int = 42
    arena_seed: int | None = None
    selfplay_seed: int | None = None
    # Deprecated legacy exploration knobs; keep defaults disabled.
    arena_model_explore_second_prob: float = 0.0
    arena_model_explore_gap_threshold: float = 0.0
    # Unified stochastic control for model policies (softmax top-k).
    model_sample_top_k: int = 2
    model_sample_temperature: float = 1.0
    arena_workers: int = 8
    arena_chunk_size: int = 0
    arena_device: str = "cpu"
    selfplay_workers: int = 8
    selfplay_chunk_size: int = 0
    selfplay_device: str = "cpu"


@dataclass
class ArenaStats:
    games: int
    candidate_wins: int
    best_wins: int
    draws: int
    total_steps: int
    black_round: SelfPlaySummary
    white_round: SelfPlaySummary

    @property
    def candidate_win_rate(self) -> float:
        return self.candidate_wins / max(1, self.games)

    @property
    def avg_steps(self) -> float:
        return self.total_steps / max(1, self.games)


@dataclass
class SelfPlayGenStats:
    games: int
    rows_added: int
    total_steps: int
    opponent_counts: dict[str, int]


@dataclass
class RoundResult:
    promoted: bool
    promoted_black: bool
    promoted_white: bool
    promotion_threshold: float
    duel: ArenaStats
    candidate_vs_baseline: ArenaStats
    best_vs_baseline: ArenaStats
    duel_passed_black: bool
    duel_passed_white: bool
    baseline_passed_black: bool
    baseline_passed_white: bool
    generated: SelfPlayGenStats
    best_black_model_path: str
    best_white_model_path: str
    candidate_black_model_path: str
    candidate_white_model_path: str
    archived_best_black_path: str | None = None
    archived_best_white_path: str | None = None


@dataclass
class ArenaSourceSpec:
    kind: str
    seed: int = 0
    device: str = "cpu"
    model_path: str = ""
    model_fast_path: str = ""
    model_slow_path: str = ""
    model_black_path: str = ""
    model_white_path: str = ""
    model_black_fast_path: str = ""
    model_black_slow_path: str = ""
    model_white_fast_path: str = ""
    model_white_slow_path: str = ""
    explore_second_prob: float = 0.0
    explore_gap_threshold: float = 0.25
    sample_top_k: int = 0
    sample_temperature: float = 1.0
    decision_top_k: int = 5
    decision_alpha: float = 1.0
    decision_beta: float = 0.5
    max_window: int = 16


def _build_source_from_spec(spec: ArenaSourceSpec) -> object:
    if spec.kind == "baseline":
        return HeuristicAISource(rng=random.Random(spec.seed))
    if spec.kind == "random":
        return RandomAISource(rng=random.Random(spec.seed))
    if spec.kind == "model":
        return ModelAISource(
            model_path=spec.model_path,
            model_fast_path=spec.model_fast_path,
            model_slow_path=spec.model_slow_path,
            model_black_path=spec.model_black_path,
            model_white_path=spec.model_white_path,
            model_black_fast_path=spec.model_black_fast_path,
            model_black_slow_path=spec.model_black_slow_path,
            model_white_fast_path=spec.model_white_fast_path,
            model_white_slow_path=spec.model_white_slow_path,
            device=spec.device,
            rng=random.Random(spec.seed),
            explore_second_prob=spec.explore_second_prob,
            explore_gap_threshold=spec.explore_gap_threshold,
            sample_top_k=spec.sample_top_k,
            sample_temperature=spec.sample_temperature,
        )
    if spec.kind == "model_v3":
        return ModelAISourceV3(
            model_path=spec.model_path,
            model_fast_path=spec.model_fast_path,
            model_slow_path=spec.model_slow_path,
            model_black_path=spec.model_black_path,
            model_white_path=spec.model_white_path,
            model_black_fast_path=spec.model_black_fast_path,
            model_black_slow_path=spec.model_black_slow_path,
            model_white_fast_path=spec.model_white_fast_path,
            model_white_slow_path=spec.model_white_slow_path,
            device=spec.device,
            rng=random.Random(spec.seed),
            sample_top_k=spec.sample_top_k,
            sample_temperature=spec.sample_temperature,
            decision_top_k=max(1, int(spec.decision_top_k)),
            decision_alpha=float(spec.decision_alpha),
            decision_beta=float(spec.decision_beta),
            max_window=max(2, int(spec.max_window)),
        )
    raise ValueError(f"Unsupported spec.kind: {spec.kind}")


def _clone_spec_with_seed(spec: ArenaSourceSpec, seed: int) -> ArenaSourceSpec:
    return ArenaSourceSpec(
        kind=spec.kind,
        seed=seed,
        device=spec.device,
        model_path=spec.model_path,
        model_fast_path=spec.model_fast_path,
        model_slow_path=spec.model_slow_path,
        model_black_path=spec.model_black_path,
        model_white_path=spec.model_white_path,
        model_black_fast_path=spec.model_black_fast_path,
        model_black_slow_path=spec.model_black_slow_path,
        model_white_fast_path=spec.model_white_fast_path,
        model_white_slow_path=spec.model_white_slow_path,
        explore_second_prob=spec.explore_second_prob,
        explore_gap_threshold=spec.explore_gap_threshold,
        sample_top_k=spec.sample_top_k,
        sample_temperature=spec.sample_temperature,
        decision_top_k=spec.decision_top_k,
        decision_alpha=spec.decision_alpha,
        decision_beta=spec.decision_beta,
        max_window=spec.max_window,
    )


def _clone_spec_with_explore(
    spec: ArenaSourceSpec,
    *,
    explore_second_prob: float,
    explore_gap_threshold: float,
) -> ArenaSourceSpec:
    cloned = _clone_spec_with_seed(spec, spec.seed)
    cloned.explore_second_prob = max(0.0, min(1.0, float(explore_second_prob)))
    cloned.explore_gap_threshold = max(0.0, float(explore_gap_threshold))
    return cloned


def _build_game_chunks(total_games: int, workers: int, chunk_size: int) -> list[int]:
    if total_games <= 0:
        return []
    if chunk_size <= 0:
        target_jobs = max(1, workers * 16)
        chunk_size = max(1, min(16, math.ceil(total_games / target_jobs)))
    chunks: list[int] = []
    done = 0
    while done < total_games:
        size = min(chunk_size, total_games - done)
        chunks.append(size)
        done += size
    return chunks


def _run_orientation_chunk_worker(
    *,
    games: int,
    game_mode: str,
    black_spec: ArenaSourceSpec,
    white_spec: ArenaSourceSpec,
) -> tuple[int, int, int, int]:
    black_source = _build_source_from_spec(black_spec)
    white_source = _build_source_from_spec(white_spec)
    runner = build_match_runner("eve", with_ui=False, black_source=black_source, white_source=white_source)
    summary = run_selfplay(games=games, runner=runner, initial_game_mode=game_mode)
    return summary.p1_wins, summary.p2_wins, summary.draws, summary.steps


def _run_orientation_parallel(
    *,
    label: str,
    games: int,
    game_mode: str,
    black_spec: ArenaSourceSpec,
    white_spec: ArenaSourceSpec,
    workers: int,
    chunk_size: int,
    seed: int,
) -> SelfPlaySummary:
    if games <= 0:
        raise ValueError("games must be > 0")
    chunks = _build_game_chunks(games, workers, chunk_size)
    p1_wins = 0
    p2_wins = 0
    draws = 0
    steps = 0
    done_games = 0
    max_workers = max(1, min(int(workers), len(chunks)))
    _render_progress(label, 0, games, extra=f"workers={max_workers} preparing...")
    start_ts = time.monotonic()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for idx, chunk_games in enumerate(chunks):
            seed_offset = seed + idx * 1013
            future = pool.submit(
                _run_orientation_chunk_worker,
                games=chunk_games,
                game_mode=game_mode,
                black_spec=_clone_spec_with_seed(black_spec, black_spec.seed + seed_offset),
                white_spec=_clone_spec_with_seed(white_spec, white_spec.seed + seed_offset + 7),
            )
            futures[future] = chunk_games
        pending = set(futures.keys())
        while pending:
            done_now, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            if not done_now:
                elapsed = time.monotonic() - start_ts
                _render_progress(label, done_games, games, extra=f"running... {elapsed:.1f}s")
                continue
            for future in done_now:
                chunk_games = futures[future]
                p1, p2, d, s = future.result()
                p1_wins += p1
                p2_wins += p2
                draws += d
                steps += s
                done_games += chunk_games
                _render_progress(label, done_games, games, extra=f"w/d/l={p1_wins}/{draws}/{p2_wins}")
    _end_progress_line()
    return SelfPlaySummary(
        games=games,
        p1_wins=p1_wins,
        p2_wins=p2_wins,
        draws=draws,
        steps=steps,
        output_path=None,
    )


def run_arena(
    *,
    best_spec: ArenaSourceSpec,
    candidate_spec: ArenaSourceSpec,
    games: int,
    game_mode: str = "slow",
    label: str = "arena",
    workers: int = 8,
    chunk_size: int = 0,
    seed: int = 42,
) -> ArenaStats:
    if games <= 0:
        raise ValueError("games must be > 0")
    if game_mode not in {"fast", "slow"}:
        raise ValueError("game_mode must be fast or slow")

    black_round = _run_orientation_parallel(
        label=f"{label}-r1",
        games=games,
        game_mode=game_mode,
        black_spec=candidate_spec,
        white_spec=best_spec,
        workers=workers,
        chunk_size=chunk_size,
        seed=seed + 11,
    )
    white_round = _run_orientation_parallel(
        label=f"{label}-r2",
        games=games,
        game_mode=game_mode,
        black_spec=best_spec,
        white_spec=candidate_spec,
        workers=workers,
        chunk_size=chunk_size,
        seed=seed + 29,
    )

    candidate_wins = black_round.p1_wins + white_round.p2_wins
    best_wins = black_round.p2_wins + white_round.p1_wins
    draws = black_round.draws + white_round.draws
    total_games = games * 2
    total_steps = black_round.steps + white_round.steps

    return ArenaStats(
        games=total_games,
        candidate_wins=candidate_wins,
        best_wins=best_wins,
        draws=draws,
        total_steps=total_steps,
        black_round=black_round,
        white_round=white_round,
    )


def should_promote(
    *,
    duel: ArenaStats,
    candidate_vs_baseline: ArenaStats,
    best_vs_baseline: ArenaStats,
    threshold: float,
) -> tuple[bool, bool, bool, bool, bool, bool, bool]:
    games = max(1, duel.black_round.games)
    cand_black_rate = duel.black_round.p1_wins / games
    best_black_rate = duel.white_round.p1_wins / games
    cand_white_rate = duel.white_round.p2_wins / games
    best_white_rate = duel.black_round.p2_wins / games

    games_cb = max(1, candidate_vs_baseline.black_round.games)
    games_wb = max(1, candidate_vs_baseline.white_round.games)
    cand_black_base_rate = candidate_vs_baseline.black_round.p1_wins / games_cb
    best_black_base_rate = best_vs_baseline.black_round.p1_wins / max(1, best_vs_baseline.black_round.games)
    cand_white_base_rate = candidate_vs_baseline.white_round.p2_wins / games_wb
    best_white_base_rate = best_vs_baseline.white_round.p2_wins / max(1, best_vs_baseline.white_round.games)

    duel_passed_black = cand_black_rate > max(best_black_rate, float(threshold))
    duel_passed_white = cand_white_rate > max(best_white_rate, float(threshold))
    baseline_passed_black = cand_black_base_rate > best_black_base_rate
    baseline_passed_white = cand_white_base_rate > best_white_base_rate
    promoted_black = duel_passed_black and baseline_passed_black
    promoted_white = duel_passed_white and baseline_passed_white
    promoted = promoted_black or promoted_white
    return (
        promoted,
        promoted_black,
        promoted_white,
        duel_passed_black,
        duel_passed_white,
        baseline_passed_black,
        baseline_passed_white,
    )


def run_league_round(
    *,
    config: LeagueConfig,
    best_model_path: str = "",
    candidate_model_path: str = "",
    best_black_model_path: str | None = None,
    best_white_model_path: str | None = None,
    candidate_black_model_path: str | None = None,
    candidate_white_model_path: str | None = None,
    best_fast_model_path: str | None = None,
    candidate_fast_model_path: str | None = None,
    best_black_fast_model_path: str | None = None,
    best_white_fast_model_path: str | None = None,
    candidate_black_fast_model_path: str | None = None,
    candidate_white_fast_model_path: str | None = None,
) -> RoundResult:
    _validate_config(config)
    arena_seed_base = int(config.arena_seed if config.arena_seed is not None else config.random_seed)
    selfplay_seed = int(config.selfplay_seed if config.selfplay_seed is not None else config.random_seed)

    best_black_slow = _must_exist(best_black_model_path or best_model_path, "best_black_model_path")
    best_white_slow = _must_exist(best_white_model_path or best_model_path, "best_white_model_path")
    cand_black_slow = _must_exist(candidate_black_model_path or candidate_model_path, "candidate_black_model_path")
    cand_white_slow = _must_exist(candidate_white_model_path or candidate_model_path, "candidate_white_model_path")

    best_black_fast = (
        _must_exist(best_black_fast_model_path, "best_black_fast_model_path")
        if best_black_fast_model_path
        else _must_exist(best_fast_model_path, "best_fast_model_path") if best_fast_model_path else best_black_slow
    )
    best_white_fast = (
        _must_exist(best_white_fast_model_path, "best_white_fast_model_path")
        if best_white_fast_model_path
        else _must_exist(best_fast_model_path, "best_fast_model_path") if best_fast_model_path else best_white_slow
    )
    cand_black_fast = (
        _must_exist(candidate_black_fast_model_path, "candidate_black_fast_model_path")
        if candidate_black_fast_model_path
        else _must_exist(candidate_fast_model_path, "candidate_fast_model_path")
        if candidate_fast_model_path
        else cand_black_slow
    )
    cand_white_fast = (
        _must_exist(candidate_white_fast_model_path, "candidate_white_fast_model_path")
        if candidate_white_fast_model_path
        else _must_exist(candidate_fast_model_path, "candidate_fast_model_path")
        if candidate_fast_model_path
        else cand_white_slow
    )

    generated = _generate_selfplay_data(
        config=config,
        selfplay_seed=selfplay_seed,
        best_black_slow=best_black_slow,
        best_white_slow=best_white_slow,
        best_black_fast=best_black_fast,
        best_white_fast=best_white_fast,
    )

    best_spec = ArenaSourceSpec(
        kind="model",
        seed=arena_seed_base + 101,
        device=config.arena_device,
        model_black_slow_path=str(best_black_slow),
        model_white_slow_path=str(best_white_slow),
        model_black_fast_path=str(best_black_fast),
        model_white_fast_path=str(best_white_fast),
        explore_second_prob=config.arena_model_explore_second_prob,
        explore_gap_threshold=config.arena_model_explore_gap_threshold,
        sample_top_k=config.model_sample_top_k,
        sample_temperature=config.model_sample_temperature,
    )
    cand_spec = ArenaSourceSpec(
        kind="model",
        seed=arena_seed_base + 211,
        device=config.arena_device,
        model_black_slow_path=str(cand_black_slow),
        model_white_slow_path=str(cand_white_slow),
        model_black_fast_path=str(cand_black_fast),
        model_white_fast_path=str(cand_white_fast),
        explore_second_prob=config.arena_model_explore_second_prob,
        explore_gap_threshold=config.arena_model_explore_gap_threshold,
        sample_top_k=config.model_sample_top_k,
        sample_temperature=config.model_sample_temperature,
    )
    baseline_spec = ArenaSourceSpec(kind="baseline", seed=arena_seed_base + 307, device=config.arena_device)
    eval_best_spec = _clone_spec_with_explore(best_spec, explore_second_prob=0.0, explore_gap_threshold=0.0)
    eval_cand_spec = _clone_spec_with_explore(cand_spec, explore_second_prob=0.0, explore_gap_threshold=0.0)
    baseline_eval_seed = arena_seed_base + 2000

    duel = run_arena(
        best_spec=best_spec,
        candidate_spec=cand_spec,
        games=config.arena_games,
        game_mode=config.game_mode,
        label="duel",
        workers=config.arena_workers,
        chunk_size=config.arena_chunk_size,
        seed=arena_seed_base + 1000,
    )
    candidate_vs_baseline = run_arena(
        best_spec=baseline_spec,
        candidate_spec=eval_cand_spec,
        games=config.arena_games,
        game_mode=config.game_mode,
        label="cand-vs-base",
        workers=config.arena_workers,
        chunk_size=config.arena_chunk_size,
        seed=baseline_eval_seed,
    )
    best_vs_baseline = run_arena(
        best_spec=baseline_spec,
        candidate_spec=eval_best_spec,
        games=config.arena_games,
        game_mode=config.game_mode,
        label="best-vs-base",
        workers=config.arena_workers,
        chunk_size=config.arena_chunk_size,
        seed=baseline_eval_seed,
    )
    (
        promoted,
        promoted_black,
        promoted_white,
        duel_passed_black,
        duel_passed_white,
        baseline_passed_black,
        baseline_passed_white,
    ) = should_promote(
        duel=duel,
        candidate_vs_baseline=candidate_vs_baseline,
        best_vs_baseline=best_vs_baseline,
        threshold=config.promotion_threshold,
    )

    archived_best_black_path: str | None = None
    archived_best_white_path: str | None = None
    if config.auto_promote:
        if config.archive_previous_best:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if promoted_black:
                archived_black = best_black_slow.with_name(
                    f"{best_black_slow.stem}.before_round_{ts}{best_black_slow.suffix}"
                )
                shutil.copy2(best_black_slow, archived_black)
                archived_best_black_path = str(archived_black)
                if best_black_fast != best_black_slow:
                    archived_black_fast = best_black_fast.with_name(
                        f"{best_black_fast.stem}.before_round_{ts}{best_black_fast.suffix}"
                    )
                    shutil.copy2(best_black_fast, archived_black_fast)
            if promoted_white:
                archived_white = best_white_slow.with_name(
                    f"{best_white_slow.stem}.before_round_{ts}{best_white_slow.suffix}"
                )
                shutil.copy2(best_white_slow, archived_white)
                archived_best_white_path = str(archived_white)
                if best_white_fast != best_white_slow:
                    archived_white_fast = best_white_fast.with_name(
                        f"{best_white_fast.stem}.before_round_{ts}{best_white_fast.suffix}"
                    )
                    shutil.copy2(best_white_fast, archived_white_fast)
        if promoted_black:
            shutil.copy2(cand_black_slow, best_black_slow)
            if best_black_fast != best_black_slow:
                shutil.copy2(cand_black_fast, best_black_fast)
        if promoted_white:
            shutil.copy2(cand_white_slow, best_white_slow)
            if best_white_fast != best_white_slow:
                shutil.copy2(cand_white_fast, best_white_fast)

    return RoundResult(
        promoted=promoted,
        promoted_black=promoted_black,
        promoted_white=promoted_white,
        promotion_threshold=config.promotion_threshold,
        duel=duel,
        candidate_vs_baseline=candidate_vs_baseline,
        best_vs_baseline=best_vs_baseline,
        duel_passed_black=duel_passed_black,
        duel_passed_white=duel_passed_white,
        baseline_passed_black=baseline_passed_black,
        baseline_passed_white=baseline_passed_white,
        generated=generated,
        best_black_model_path=str(best_black_slow),
        best_white_model_path=str(best_white_slow),
        candidate_black_model_path=str(cand_black_slow),
        candidate_white_model_path=str(cand_white_slow),
        archived_best_black_path=archived_best_black_path,
        archived_best_white_path=archived_best_white_path,
    )


def _validate_config(config: LeagueConfig) -> None:
    if config.arena_games <= 0:
        raise ValueError("arena_games must be > 0")
    if config.selfplay_games < 0:
        raise ValueError("selfplay_games must be >= 0")
    if config.game_mode not in {"fast", "slow"}:
        raise ValueError("game_mode must be fast or slow")
    if not (0.0 <= config.promotion_threshold <= 1.0):
        raise ValueError("promotion_threshold must be in [0,1]")
    if config.arena_workers <= 0:
        raise ValueError("arena_workers must be > 0")
    if config.arena_chunk_size < 0:
        raise ValueError("arena_chunk_size must be >= 0")
    if config.selfplay_workers <= 0:
        raise ValueError("selfplay_workers must be > 0")
    if config.selfplay_chunk_size < 0:
        raise ValueError("selfplay_chunk_size must be >= 0")
    if config.model_sample_top_k < 0:
        raise ValueError("model_sample_top_k must be >= 0")
    if config.model_sample_temperature <= 0:
        raise ValueError("model_sample_temperature must be > 0")
    _normalize_opponent_pool(config.opponent_pool)


def _must_exist(path: str | None, label: str) -> Path:
    if path is None or not str(path).strip():
        raise ValueError(f"{label} is required")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    rows = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows += 1
    return rows


def _normalize_opponent_pool(pool: dict[str, float]) -> dict[str, float]:
    allowed = {"teacher", "old_best", "random"}
    unknown = [k for k in pool.keys() if k not in allowed]
    if unknown:
        raise ValueError(f"Unsupported opponent_pool keys: {unknown}")
    filtered = {k: float(v) for k, v in pool.items() if float(v) > 0}
    if not filtered:
        raise ValueError("opponent_pool must contain at least one positive weight")
    total = sum(filtered.values())
    return {k: v / total for k, v in filtered.items()}


def _pick_opponent_kind(rng: random.Random, normalized_pool: dict[str, float]) -> str:
    x = rng.random()
    acc = 0.0
    last = "teacher"
    for kind, prob in normalized_pool.items():
        acc += prob
        last = kind
        if x <= acc:
            return kind
    return last


def _build_model_source(
    *,
    black_slow_path: Path,
    white_slow_path: Path,
    black_fast_path: Path,
    white_fast_path: Path,
    device: str,
    sample_top_k: int,
    sample_temperature: float,
    explore_second_prob: float,
    explore_gap_threshold: float,
) -> ModelAISource:
    return ModelAISource(
        model_path=str(black_slow_path),
        model_black_path=str(black_slow_path),
        model_white_path=str(white_slow_path),
        model_black_slow_path=str(black_slow_path),
        model_white_slow_path=str(white_slow_path),
        model_black_fast_path=str(black_fast_path),
        model_white_fast_path=str(white_fast_path),
        model_fast_path=str(black_fast_path),
        model_slow_path=str(black_slow_path),
        device=device,
        sample_top_k=max(0, int(sample_top_k)),
        sample_temperature=max(1e-6, float(sample_temperature)),
        explore_second_prob=max(0.0, min(1.0, float(explore_second_prob))),
        explore_gap_threshold=max(0.0, float(explore_gap_threshold)),
    )


def _build_opponent_source(
    kind: str,
    *,
    best_black_slow: Path,
    best_white_slow: Path,
    best_black_fast: Path,
    best_white_fast: Path,
    rng_seed: int,
    device: str,
    sample_top_k: int,
    sample_temperature: float,
    explore_second_prob: float,
    explore_gap_threshold: float,
) -> object:
    if kind == "teacher":
        return HeuristicAISource(rng=random.Random(rng_seed))
    if kind == "random":
        return RandomAISource(rng=random.Random(rng_seed))
    if kind == "old_best":
        return _build_model_source(
            black_slow_path=best_black_slow,
            white_slow_path=best_white_slow,
            black_fast_path=best_black_fast,
            white_fast_path=best_white_fast,
            device=device,
            sample_top_k=sample_top_k,
            sample_temperature=sample_temperature,
            explore_second_prob=explore_second_prob,
            explore_gap_threshold=explore_gap_threshold,
        )
    raise ValueError(f"Unsupported opponent kind: {kind}")


def _build_selfplay_jobs(
    *,
    games: int,
    normalized_pool: dict[str, float],
    random_seed: int,
) -> list[tuple[int, str, bool, int]]:
    rng = random.Random(random_seed)
    jobs: list[tuple[int, str, bool, int]] = []
    for game_idx in range(games):
        kind = _pick_opponent_kind(rng, normalized_pool)
        best_is_black = rng.random() < 0.5
        job_seed = random_seed + game_idx * 37
        jobs.append((game_idx, kind, best_is_black, job_seed))
    return jobs


def _chunk_jobs(
    jobs: list[tuple[int, str, bool, int]],
    *,
    workers: int,
    chunk_size: int,
) -> list[list[tuple[int, str, bool, int]]]:
    if not jobs:
        return []
    if chunk_size <= 0:
        target_jobs = max(1, workers * 16)
        chunk_size = max(1, min(16, math.ceil(len(jobs) / target_jobs)))
    out: list[list[tuple[int, str, bool, int]]] = []
    start = 0
    while start < len(jobs):
        out.append(jobs[start : start + chunk_size])
        start += chunk_size
    return out


def _run_selfplay_chunk_worker(
    *,
    jobs: list[tuple[int, str, bool, int]],
    best_black_slow_path: str,
    best_white_slow_path: str,
    best_black_fast_path: str,
    best_white_fast_path: str,
    game_mode: str,
    device: str,
    output_path: str,
    model_sample_top_k: int,
    model_sample_temperature: float,
    model_explore_second_prob: float,
    model_explore_gap_threshold: float,
) -> tuple[int, int, int, dict[str, int]]:
    best_black_slow = Path(best_black_slow_path)
    best_white_slow = Path(best_white_slow_path)
    best_black_fast = Path(best_black_fast_path)
    best_white_fast = Path(best_white_fast_path)
    out = Path(output_path)
    if out.exists():
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)

    opponent_counts = {"teacher": 0, "old_best": 0, "random": 0}
    total_steps = 0
    for _, kind, best_is_black, rng_seed in jobs:
        best_source = _build_model_source(
            black_slow_path=best_black_slow,
            white_slow_path=best_white_slow,
            black_fast_path=best_black_fast,
            white_fast_path=best_white_fast,
            device=device,
            sample_top_k=model_sample_top_k,
            sample_temperature=model_sample_temperature,
            explore_second_prob=model_explore_second_prob,
            explore_gap_threshold=model_explore_gap_threshold,
        )
        opponent_source = _build_opponent_source(
            kind,
            best_black_slow=best_black_slow,
            best_white_slow=best_white_slow,
            best_black_fast=best_black_fast,
            best_white_fast=best_white_fast,
            rng_seed=rng_seed,
            device=device,
            sample_top_k=model_sample_top_k,
            sample_temperature=model_sample_temperature,
            explore_second_prob=model_explore_second_prob,
            explore_gap_threshold=model_explore_gap_threshold,
        )
        if best_is_black:
            black_source, white_source = best_source, opponent_source
        else:
            black_source, white_source = opponent_source, best_source
        runner = build_match_runner("eve", with_ui=False, black_source=black_source, white_source=white_source)
        summary = run_selfplay(
            games=1,
            output_path=str(out),
            append_output=True,
            runner=runner,
            initial_game_mode=game_mode,
        )
        total_steps += summary.steps
        opponent_counts[kind] += 1

    rows_added = _count_jsonl_rows(out)
    return len(jobs), rows_added, total_steps, opponent_counts


def _generate_selfplay_data(
    *,
    config: LeagueConfig,
    selfplay_seed: int,
    best_black_slow: Path,
    best_white_slow: Path,
    best_black_fast: Path,
    best_white_fast: Path,
) -> SelfPlayGenStats:
    if config.selfplay_games == 0:
        return SelfPlayGenStats(
            games=0,
            rows_added=0,
            total_steps=0,
            opponent_counts={"teacher": 0, "old_best": 0, "random": 0},
        )

    trace_path = Path(config.trace_output_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    append = config.append_trace
    if not append and trace_path.exists():
        trace_path.unlink()

    normalized_pool = _normalize_opponent_pool(config.opponent_pool)
    opponent_counts = {"teacher": 0, "old_best": 0, "random": 0}
    total_steps = 0
    total_rows_added = 0
    selfplay_jobs = _build_selfplay_jobs(
        games=config.selfplay_games,
        normalized_pool=normalized_pool,
        random_seed=selfplay_seed,
    )
    workers = max(1, min(int(config.selfplay_workers), max(1, os.cpu_count() or 1), len(selfplay_jobs)))
    chunks = _chunk_jobs(selfplay_jobs, workers=workers, chunk_size=int(config.selfplay_chunk_size))
    start_ts = time.monotonic()
    done_games = 0
    _render_progress("selfplay", 0, config.selfplay_games, extra=f"workers={workers} preparing...")

    if workers <= 1:
        for _, kind, best_is_black, rng_seed in selfplay_jobs:
            best_source = _build_model_source(
                black_slow_path=best_black_slow,
                white_slow_path=best_white_slow,
                black_fast_path=best_black_fast,
                white_fast_path=best_white_fast,
                device=config.selfplay_device,
                sample_top_k=config.model_sample_top_k,
                sample_temperature=config.model_sample_temperature,
                explore_second_prob=config.arena_model_explore_second_prob,
                explore_gap_threshold=config.arena_model_explore_gap_threshold,
            )
            opponent_source = _build_opponent_source(
                kind,
                best_black_slow=best_black_slow,
                best_white_slow=best_white_slow,
                best_black_fast=best_black_fast,
                best_white_fast=best_white_fast,
                rng_seed=rng_seed,
                device=config.selfplay_device,
                sample_top_k=config.model_sample_top_k,
                sample_temperature=config.model_sample_temperature,
                explore_second_prob=config.arena_model_explore_second_prob,
                explore_gap_threshold=config.arena_model_explore_gap_threshold,
            )
            if best_is_black:
                black_source, white_source = best_source, opponent_source
            else:
                black_source, white_source = opponent_source, best_source

            runner = build_match_runner("eve", with_ui=False, black_source=black_source, white_source=white_source)
            summary = run_selfplay(
                games=1,
                output_path=str(trace_path),
                append_output=True,
                runner=runner,
                initial_game_mode=config.game_mode,
            )
            total_steps += summary.steps
            opponent_counts[kind] += 1
            done_games += 1
            total_rows_added += summary.steps
            _render_progress(
                "selfplay",
                done_games,
                config.selfplay_games,
                extra=(
                    f"rows={total_rows_added} pool={opponent_counts['teacher']}/"
                    f"{opponent_counts['old_best']}/{opponent_counts['random']}"
                ),
            )
    else:
        with tempfile.TemporaryDirectory(prefix="league_selfplay_", dir=str(trace_path.parent)) as tmp_dir:
            tmp_root = Path(tmp_dir)
            with trace_path.open("a", encoding="utf-8") as out_fp:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = {}
                    for idx, chunk in enumerate(chunks):
                        tmp_out = tmp_root / f"chunk_{idx:04d}.jsonl"
                        future = pool.submit(
                            _run_selfplay_chunk_worker,
                            jobs=chunk,
                            best_black_slow_path=str(best_black_slow),
                            best_white_slow_path=str(best_white_slow),
                            best_black_fast_path=str(best_black_fast),
                            best_white_fast_path=str(best_white_fast),
                            game_mode=config.game_mode,
                            device=config.selfplay_device,
                            output_path=str(tmp_out),
                            model_sample_top_k=config.model_sample_top_k,
                            model_sample_temperature=config.model_sample_temperature,
                            model_explore_second_prob=config.arena_model_explore_second_prob,
                            model_explore_gap_threshold=config.arena_model_explore_gap_threshold,
                        )
                        futures[future] = (len(chunk), tmp_out)

                    pending = set(futures.keys())
                    while pending:
                        done_now, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                        if not done_now:
                            elapsed = time.monotonic() - start_ts
                            _render_progress(
                                "selfplay",
                                done_games,
                                config.selfplay_games,
                                extra=f"running... {elapsed:.1f}s",
                            )
                            continue
                        for future in done_now:
                            chunk_games, tmp_out = futures[future]
                            got_games, rows_added, chunk_steps, chunk_counts = future.result()
                            done_games += got_games
                            total_rows_added += rows_added
                            total_steps += chunk_steps
                            for key in opponent_counts:
                                opponent_counts[key] += int(chunk_counts.get(key, 0))
                            if tmp_out.exists():
                                with tmp_out.open("r", encoding="utf-8") as chunk_fp:
                                    for line in chunk_fp:
                                        if line.strip():
                                            out_fp.write(line)
                                tmp_out.unlink(missing_ok=True)
                            _render_progress(
                                "selfplay",
                                done_games,
                                config.selfplay_games,
                                extra=(
                                    f"rows={total_rows_added} pool={opponent_counts['teacher']}/"
                                    f"{opponent_counts['old_best']}/{opponent_counts['random']}"
                                ),
                            )

    _end_progress_line()
    return SelfPlayGenStats(
        games=config.selfplay_games,
        rows_added=max(0, total_rows_added),
        total_steps=total_steps,
        opponent_counts=opponent_counts,
    )


def _parse_opponent_pool(text: str) -> dict[str, float]:
    if not text.strip():
        return {"teacher": 0.5, "old_best": 0.5}
    out: dict[str, float] = {}
    parts = [part.strip() for part in text.split(",") if part.strip()]
    for part in parts:
        if ":" not in part:
            raise ValueError(f"Invalid opponent pool item: {part}")
        key, value = part.split(":", 1)
        out[key.strip()] = float(value.strip())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one self-play league round")
    parser.add_argument("--best-model", default="", help="Shared fallback best slow-model path")
    parser.add_argument("--candidate-model", default="", help="Shared fallback candidate slow-model path")
    parser.add_argument("--best-model-black", default="", help="Best black-model path")
    parser.add_argument("--best-model-white", default="", help="Best white-model path")
    parser.add_argument("--candidate-model-black", default="", help="Candidate black-model path")
    parser.add_argument("--candidate-model-white", default="", help="Candidate white-model path")
    parser.add_argument("--best-fast-model", default="", help="Optional best fast-model path")
    parser.add_argument("--candidate-fast-model", default="", help="Optional candidate fast-model path")
    parser.add_argument("--best-fast-model-black", default="", help="Optional best black fast-model path")
    parser.add_argument("--best-fast-model-white", default="", help="Optional best white fast-model path")
    parser.add_argument("--candidate-fast-model-black", default="", help="Optional candidate black fast-model path")
    parser.add_argument("--candidate-fast-model-white", default="", help="Optional candidate white fast-model path")
    parser.add_argument("--promotion-threshold", type=float, default=0.50)
    parser.add_argument("--arena-games", type=int, default=200, help="Games per color assignment")
    parser.add_argument("--arena-workers", type=int, default=8, help="Worker processes for arena evaluations")
    parser.add_argument("--arena-chunk-size", type=int, default=0, help="Arena games per worker chunk (0=auto)")
    parser.add_argument("--arena-device", default="cpu", help="Device for arena workers (default cpu)")
    parser.add_argument("--selfplay-games", type=int, default=400, help="Generated games from best vs pool")
    parser.add_argument("--game-mode", choices=["fast", "slow"], default="slow")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opponent-pool", default="teacher:0.5,old_best:0.5")
    parser.add_argument("--trace-output", default="artifacts/datasets/league_selfplay_round.jsonl")
    parser.add_argument("--append-trace", action="store_true")
    parser.add_argument("--selfplay-workers", type=int, default=8, help="Worker processes for selfplay generation")
    parser.add_argument("--selfplay-chunk-size", type=int, default=0, help="Selfplay games per worker chunk (0=auto)")
    parser.add_argument(
        "--selfplay-device",
        default="cpu",
        help="Device for selfplay generation workers (default cpu for stable multiprocessing)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--arena-seed",
        type=int,
        default=None,
        help="Optional fixed seed for arena evaluations (if unset, uses --seed)",
    )
    parser.add_argument(
        "--selfplay-seed",
        type=int,
        default=None,
        help="Optional seed for selfplay generation (if unset, uses --seed)",
    )
    parser.add_argument(
        "--arena-model-explore-second-prob",
        type=float,
        default=0.0,
        help="Legacy top2 exploration probability for model policy (keep 0 when using softmax-only)",
    )
    parser.add_argument(
        "--arena-model-explore-gap-threshold",
        type=float,
        default=0.0,
        help="Legacy top2 exploration gap threshold (keep 0 when using softmax-only)",
    )
    parser.add_argument(
        "--model-sample-top-k",
        type=int,
        default=2,
        help="Unified model stochasticity: softmax sample top-k (0 disables sampling)",
    )
    parser.add_argument(
        "--model-sample-temperature",
        type=float,
        default=1.0,
        help="Unified model stochasticity: softmax temperature",
    )
    parser.add_argument("--no-auto-promote", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()

    config = LeagueConfig(
        promotion_threshold=args.promotion_threshold,
        arena_games=args.arena_games,
        selfplay_games=args.selfplay_games,
        game_mode=args.game_mode,
        device=args.device,
        opponent_pool=_parse_opponent_pool(args.opponent_pool),
        trace_output_path=args.trace_output,
        append_trace=args.append_trace,
        auto_promote=not args.no_auto_promote,
        archive_previous_best=not args.no_archive,
        random_seed=args.seed,
        arena_seed=args.arena_seed,
        selfplay_seed=args.selfplay_seed,
        arena_model_explore_second_prob=args.arena_model_explore_second_prob,
        arena_model_explore_gap_threshold=args.arena_model_explore_gap_threshold,
        model_sample_top_k=args.model_sample_top_k,
        model_sample_temperature=args.model_sample_temperature,
        arena_workers=args.arena_workers,
        arena_chunk_size=args.arena_chunk_size,
        arena_device=args.arena_device,
        selfplay_workers=args.selfplay_workers,
        selfplay_chunk_size=args.selfplay_chunk_size,
        selfplay_device=args.selfplay_device,
    )
    result = run_league_round(
        config=config,
        best_model_path=args.best_model,
        candidate_model_path=args.candidate_model,
        best_black_model_path=(args.best_model_black or None),
        best_white_model_path=(args.best_model_white or None),
        candidate_black_model_path=(args.candidate_model_black or None),
        candidate_white_model_path=(args.candidate_model_white or None),
        best_fast_model_path=(args.best_fast_model or None),
        candidate_fast_model_path=(args.candidate_fast_model or None),
        best_black_fast_model_path=(args.best_fast_model_black or None),
        best_white_fast_model_path=(args.best_fast_model_white or None),
        candidate_black_fast_model_path=(args.candidate_fast_model_black or None),
        candidate_white_fast_model_path=(args.candidate_fast_model_white or None),
    )

    print(_paint("--- League Round ---", "36"))
    print(f"best_black={result.best_black_model_path}")
    print(f"best_white={result.best_white_model_path}")
    print(f"candidate_black={result.candidate_black_model_path}")
    print(f"candidate_white={result.candidate_white_model_path}")
    print(
        f"generated_games={result.generated.games} rows_added={result.generated.rows_added} "
        f"steps={result.generated.total_steps} pool={result.generated.opponent_counts}"
    )
    print(
        f"duel_games={result.duel.games} candidate_wins={result.duel.candidate_wins} "
        f"best_wins={result.duel.best_wins} draws={result.duel.draws} "
        f"candidate_win_rate={result.duel.candidate_win_rate:.3f} avg_steps={result.duel.avg_steps:.2f}"
    )
    print(
        f"candidate_vs_baseline_win_rate={result.candidate_vs_baseline.candidate_win_rate:.3f} "
        f"best_vs_baseline_win_rate={result.best_vs_baseline.candidate_win_rate:.3f}"
    )
    print(f"promotion_threshold={result.promotion_threshold:.3f}")
    print(
        " ".join(
            [
                _status_tag("duel_black", result.duel_passed_black),
                _status_tag("base_black", result.baseline_passed_black),
                _status_tag("promote_black", result.promoted_black),
            ]
        )
    )
    print(
        " ".join(
            [
                _status_tag("duel_white", result.duel_passed_white),
                _status_tag("base_white", result.baseline_passed_white),
                _status_tag("promote_white", result.promoted_white),
                _promotion_tag(result.promoted),
            ]
        )
    )
    if _first_player_dominant(result.duel):
        print(
            _paint(
                "warning=first_player_dominant_in_duel (increase arena exploration if duel is too rigid)",
                "33",
            )
        )
    if result.archived_best_black_path:
        print(_paint(f"archived_best_black={result.archived_best_black_path}", "36"))
    if result.archived_best_white_path:
        print(_paint(f"archived_best_white={result.archived_best_white_path}", "36"))


if __name__ == "__main__":
    main()
