"""Parallel exporter for teacher traces (baseline vs baseline, slow mode only)."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fenghuo_chess.ai.baselines.heuristic_ai import HeuristicAISource
from src.fenghuo_chess.application.logic_api import apply_action, legal_actions
from src.fenghuo_chess.domain.models import create_initial_state
from src.fenghuo_chess.domain.serialization import state_to_dict
from src.fenghuo_chess.services.exposure_service import ExposureService


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


def _outcome_for_player(player: int, winner: int) -> int:
    if winner == 0:
        return 0
    return 1 if int(player) == int(winner) else -1


def _run_single_game(
    *,
    game_id: int,
    seed: int,
    explore_second_prob: float,
    explore_gap_threshold: float,
) -> tuple[list[str], str, int, int]:
    black = HeuristicAISource(
        rng=random.Random(seed + 11),
        explore_second_prob=explore_second_prob,
        explore_gap_threshold=explore_gap_threshold,
    )
    white = HeuristicAISource(
        rng=random.Random(seed + 29),
        explore_second_prob=explore_second_prob,
        explore_gap_threshold=explore_gap_threshold,
    )

    state = create_initial_state()
    state.show_intro = False
    state.game_mode = "slow"
    exposure = ExposureService()

    rows: list[dict[str, Any]] = []
    step_idx = 0
    while not state.game_over:
        legal = legal_actions(state)
        if not legal:
            break
        source = black if state.current_player == 1 else white
        action = source.select_action(state, legal)
        if action is None:
            break

        row: dict[str, Any] = {
            "game_id": game_id,
            "step_index": step_idx,
            "seed": seed,
            "mode": "slow",
            "state": state_to_dict(state, core_only=True),
            "legal_actions": [[a.row, a.col] for a in legal],
            "action": [action.row, action.col],
            "player": int(state.current_player),
            "events": [],
            "outcome": None,
        }
        step = apply_action(state, action, exposure_service=exposure)
        row["events"] = [event.kind for event in step.events]
        rows.append(row)

        if not step.ok:
            break
        step_idx += 1

    winner = int(state.winner)
    for row in rows:
        row["outcome"] = _outcome_for_player(int(row["player"]), winner)

    summary = {
        "game_id": game_id,
        "seed": seed,
        "mode": "slow",
        "winner": winner,
        "num_steps": len(rows),
    }
    trace_lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    summary_line = json.dumps(summary, ensure_ascii=False)
    return trace_lines, summary_line, winner, len(rows)


def _run_chunk(
    *,
    start_game_id: int,
    num_games: int,
    base_seed: int,
    explore_second_prob: float,
    explore_gap_threshold: float,
) -> tuple[list[str], list[str], int, int, int, int]:
    trace_lines: list[str] = []
    summary_lines: list[str] = []
    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_steps = 0

    for offset in range(num_games):
        game_id = start_game_id + offset
        seed = base_seed + game_id * 7919
        game_traces, game_summary, winner, steps = _run_single_game(
            game_id=game_id,
            seed=seed,
            explore_second_prob=explore_second_prob,
            explore_gap_threshold=explore_gap_threshold,
        )
        trace_lines.extend(game_traces)
        summary_lines.append(game_summary)
        total_steps += steps
        if winner == 1:
            p1_wins += 1
        elif winner == 2:
            p2_wins += 1
        else:
            draws += 1

    return trace_lines, summary_lines, p1_wins, p2_wins, draws, total_steps


def _build_jobs(games: int, workers: int, chunk_size: int) -> list[tuple[int, int]]:
    if games <= 0:
        return []
    if chunk_size <= 0:
        # Use smaller auto chunks so progress appears early while still amortizing IPC overhead.
        target_jobs = max(1, workers * 32)
        chunk_size = max(1, min(8, math.ceil(games / target_jobs)))
    jobs: list[tuple[int, int]] = []
    start = 0
    while start < games:
        size = min(chunk_size, games - start)
        jobs.append((start, size))
        start += size
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel teacher trace export (baseline-vs-baseline, slow only)")
    parser.add_argument("--games", type=int, default=1000, help="Number of games to export")
    parser.add_argument("--workers", type=int, default=8, help="Worker process count")
    parser.add_argument("--chunk-size", type=int, default=0, help="Games per job chunk (0 = auto)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument(
        "--trace-output",
        type=str,
        default="artifacts/datasets/teacher_trace_raw.jsonl",
        help="Output trace JSONL path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Deprecated alias of --trace-output",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="artifacts/datasets/teacher_games_summary.jsonl",
        help="Output game-summary JSONL path",
    )
    parser.add_argument(
        "--explore-second-prob",
        type=float,
        default=0.25,
        help="Probability to choose second-best action in non-urgent positions",
    )
    parser.add_argument(
        "--explore-gap-threshold",
        type=float,
        default=320.0,
        help="Top1-top2 score-gap threshold for enabling second-best exploration",
    )
    args = parser.parse_args()

    if args.games <= 0:
        raise ValueError("--games must be > 0")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if args.chunk_size < 0:
        raise ValueError("--chunk-size must be >= 0")

    workers = min(args.workers, max(1, os.cpu_count() or 1))
    jobs = _build_jobs(args.games, workers, args.chunk_size)

    trace_output = Path(args.output.strip() or args.trace_output)
    summary_output = Path(args.summary_output)
    trace_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    print("--- Parallel Teacher Trace Export ---")
    print(
        f"mode=slow games={args.games} workers={workers} jobs={len(jobs)} "
        f"chunk_size={jobs[0][1] if jobs else 0} "
        f"explore_second_prob={args.explore_second_prob:.2f} gap_threshold={args.explore_gap_threshold:.1f}"
    )
    print(f"trace_output={trace_output}")
    print(f"summary_output={summary_output}")

    done_games = 0
    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_steps = 0
    total_rows = 0

    with trace_output.open("w", encoding="utf-8") as trace_fp, summary_output.open("w", encoding="utf-8") as summary_fp:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _run_chunk,
                    start_game_id=start,
                    num_games=size,
                    base_seed=args.seed,
                    explore_second_prob=args.explore_second_prob,
                    explore_gap_threshold=args.explore_gap_threshold,
                )
                for start, size in jobs
            ]
            future_to_size = {future: size for future, (_, size) in zip(futures, jobs, strict=True)}
            pending = set(futures)
            start_ts = time.monotonic()
            _render_progress("export", 0, args.games, extra="starting workers...")
            while pending:
                done_now, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                if not done_now:
                    elapsed = time.monotonic() - start_ts
                    _render_progress("export", done_games, args.games, extra=f"starting... {elapsed:.1f}s")
                    continue
                for future in done_now:
                    size = future_to_size[future]
                    trace_lines, summary_lines, p1, p2, d, steps = future.result()
                    for line in trace_lines:
                        trace_fp.write(line)
                        trace_fp.write("\n")
                    for line in summary_lines:
                        summary_fp.write(line)
                        summary_fp.write("\n")

                    done_games += size
                    p1_wins += p1
                    p2_wins += p2
                    draws += d
                    total_steps += steps
                    total_rows += len(trace_lines)
                    avg_steps = total_steps / max(1, done_games)
                    _render_progress(
                        "export",
                        done_games,
                        args.games,
                        extra=f"rows={total_rows} w/d/l={p1_wins}/{draws}/{p2_wins} avg_steps={avg_steps:.2f}",
                    )
    sys.stdout.write("\n")

    print("--- Export Summary ---")
    print(f"games={done_games} rows={total_rows} p1_wins={p1_wins} p2_wins={p2_wins} draws={draws}")
    print(f"avg_steps={total_steps / max(1, done_games):.2f}")


if __name__ == "__main__":
    main()
