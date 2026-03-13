"""Parallel exporter for V3 model selfplay traces (mirror or mixed opponent pool)."""

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
from src.fenghuo_chess.ai.baselines.random_ai import RandomAISource
from src.fenghuo_chess.ai.model_ai import ModelAISource
from src.fenghuo_chess.ai.model_v3 import ModelAISourceV3
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


def _canonical_pool_key(key: str) -> str:
    k = key.strip().lower().replace("-", "_")
    aliases = {
        "random": "random",
        "teacher": "v2_best",
        "teacherai": "v2_best",
        "teacher_ai": "v2_best",
        "v2best": "v2_best",
        "v2_best": "v2_best",
        "baseline": "baseline",
        "rule_baseline": "baseline",
        "rules_baseline": "baseline",
        "oldbest": "old_best",
        "old_best": "old_best",
        "best": "old_best",
    }
    return aliases.get(k, k)


def _parse_opponent_pool(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    parts = [part.strip() for part in text.split(",") if part.strip()]
    for part in parts:
        if ":" not in part:
            raise ValueError(f"Invalid opponent pool item: {part}")
        key, value = part.split(":", 1)
        canon = _canonical_pool_key(key)
        out[canon] = out.get(canon, 0.0) + float(value.strip())
    return out


def _normalize_opponent_pool(pool: dict[str, float]) -> dict[str, float]:
    allowed = {"random", "v2_best", "baseline", "old_best"}
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
    last = "old_best"
    for kind, prob in normalized_pool.items():
        acc += prob
        last = kind
        if x <= acc:
            return kind
    return last


def _seed_source(source: object, seed: int) -> None:
    rng = getattr(source, "_rng", None)
    if isinstance(rng, random.Random):
        rng.seed(seed)


def _build_model_source(
    *,
    model_black_slow: str,
    model_white_slow: str,
    model_black_fast: str,
    model_white_fast: str,
    device: str,
    seed: int,
    sample_top_k: int,
    sample_temperature: float,
    decision_top_k: int,
    decision_alpha: float,
    decision_beta: float,
    max_window: int,
) -> ModelAISourceV3:
    return ModelAISourceV3(
        model_path=model_black_slow,
        model_black_path=model_black_slow,
        model_white_path=model_white_slow,
        model_black_slow_path=model_black_slow,
        model_white_slow_path=model_white_slow,
        model_black_fast_path=model_black_fast,
        model_white_fast_path=model_white_fast,
        model_fast_path=model_black_fast,
        model_slow_path=model_black_slow,
        device=device,
        rng=random.Random(seed),
        sample_top_k=sample_top_k,
        sample_temperature=sample_temperature,
        decision_top_k=max(1, int(decision_top_k)),
        decision_alpha=float(decision_alpha),
        decision_beta=float(decision_beta),
        max_window=max(2, int(max_window)),
    )


def _outcome_for_player(player: int, winner: int) -> int:
    if winner == 0:
        return 0
    return 1 if int(player) == int(winner) else -1


def _run_single_game(
    *,
    game_id: int,
    seed: int,
    game_mode: str,
    black_source: object,
    white_source: object,
) -> tuple[list[str], dict[str, Any], int, int]:
    state = create_initial_state()
    state.show_intro = False
    state.game_mode = game_mode
    exposure = ExposureService()

    rows: list[dict[str, Any]] = []
    step_idx = 0
    while not state.game_over:
        legal = legal_actions(state)
        if not legal:
            break
        source = black_source if state.current_player == 1 else white_source
        action = source.select_action(state, legal)
        if action is None:
            break
        row: dict[str, Any] = {
            "game_id": game_id,
            "step_index": step_idx,
            "seed": seed,
            "mode": game_mode,
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
        "mode": game_mode,
        "winner": winner,
        "num_steps": len(rows),
    }
    trace_lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    return trace_lines, summary, winner, len(rows)


def _run_chunk(
    *,
    start_game_id: int,
    num_games: int,
    base_seed: int,
    game_mode: str,
    device: str,
    model_black_slow: str,
    model_white_slow: str,
    model_black_fast: str,
    model_white_fast: str,
    v2_model_black_slow: str,
    v2_model_white_slow: str,
    v2_model_black_fast: str,
    v2_model_white_fast: str,
    sample_top_k: int,
    sample_temperature: float,
    decision_top_k: int,
    decision_alpha: float,
    decision_beta: float,
    max_window: int,
    pool_mode: str,
    opponent_pool: dict[str, float],
) -> tuple[list[str], list[str], int, int, int, int, dict[str, int]]:
    chunk_seed = base_seed + start_game_id * 1009
    best_source = _build_model_source(
        model_black_slow=model_black_slow,
        model_white_slow=model_white_slow,
        model_black_fast=model_black_fast,
        model_white_fast=model_white_fast,
        device=device,
        seed=chunk_seed + 11,
        sample_top_k=sample_top_k,
        sample_temperature=sample_temperature,
        decision_top_k=decision_top_k,
        decision_alpha=decision_alpha,
        decision_beta=decision_beta,
        max_window=max_window,
    )
    old_best_source = _build_model_source(
        model_black_slow=model_black_slow,
        model_white_slow=model_white_slow,
        model_black_fast=model_black_fast,
        model_white_fast=model_white_fast,
        device=device,
        seed=chunk_seed + 29,
        sample_top_k=sample_top_k,
        sample_temperature=sample_temperature,
        decision_top_k=decision_top_k,
        decision_alpha=decision_alpha,
        decision_beta=decision_beta,
        max_window=max_window,
    )
    v2_best_source = ModelAISource(
        model_path=v2_model_black_slow,
        model_black_path=v2_model_black_slow,
        model_white_path=v2_model_white_slow,
        model_black_slow_path=v2_model_black_slow,
        model_white_slow_path=v2_model_white_slow,
        model_black_fast_path=v2_model_black_fast,
        model_white_fast_path=v2_model_white_fast,
        model_fast_path=v2_model_black_fast,
        model_slow_path=v2_model_black_slow,
        device=device,
        rng=random.Random(chunk_seed + 53),
        sample_top_k=sample_top_k,
        sample_temperature=sample_temperature,
    )
    random_source = RandomAISource(rng=random.Random(chunk_seed + 41))
    baseline_source = HeuristicAISource(rng=random.Random(chunk_seed + 67))

    normalized_pool = dict(opponent_pool)
    opponent_counts = {"random": 0, "v2_best": 0, "baseline": 0, "old_best": 0}
    opponents: dict[str, object] = {
        "random": random_source,
        "v2_best": v2_best_source,
        "baseline": baseline_source,
        "old_best": old_best_source,
    }

    trace_lines: list[str] = []
    summary_lines: list[str] = []
    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_steps = 0

    for offset in range(num_games):
        game_id = start_game_id + offset
        seed = base_seed + game_id * 7919
        choose_rng = random.Random(seed + 101)
        kind = "old_best" if pool_mode == "mirror" else _pick_opponent_kind(choose_rng, normalized_pool)
        best_is_black = choose_rng.random() < 0.5
        best_player = 1 if best_is_black else 2
        opponent_counts[kind] += 1

        opponent_source = opponents[kind]
        black_source = best_source if best_is_black else opponent_source
        white_source = opponent_source if best_is_black else best_source

        _seed_source(best_source, seed + 11)
        _seed_source(old_best_source, seed + 29)
        _seed_source(random_source, seed + 41)
        _seed_source(v2_best_source, seed + 53)
        _seed_source(baseline_source, seed + 67)
        game_traces, game_summary, winner, steps = _run_single_game(
            game_id=game_id,
            seed=seed,
            game_mode=game_mode,
            black_source=black_source,
            white_source=white_source,
        )
        game_summary["opponent_kind"] = kind
        game_summary["best_player"] = best_player
        game_summary["best_color"] = "black" if best_player == 1 else "white"
        game_summary["opponent_player"] = 2 if best_player == 1 else 1
        game_summary["opponent_color"] = "white" if best_player == 1 else "black"
        game_summary["best_outcome"] = _outcome_for_player(best_player, winner)
        game_summary["pool_mode"] = pool_mode
        trace_lines.extend(game_traces)
        summary_lines.append(json.dumps(game_summary, ensure_ascii=False))
        total_steps += steps
        if winner == 1:
            p1_wins += 1
        elif winner == 2:
            p2_wins += 1
        else:
            draws += 1

    return trace_lines, summary_lines, p1_wins, p2_wins, draws, total_steps, opponent_counts


def _build_jobs(games: int, workers: int, chunk_size: int) -> list[tuple[int, int]]:
    if games <= 0:
        return []
    if chunk_size <= 0:
        # Auto mode: keep job count moderate to reduce process scheduling overhead.
        target_jobs = max(1, workers * 8)
        chunk_size = max(4, min(64, math.ceil(games / target_jobs)))
    jobs: list[tuple[int, int]] = []
    start = 0
    while start < games:
        size = min(chunk_size, games - start)
        jobs.append((start, size))
        start += size
    return jobs


def _next_game_id(summary_output: Path) -> int:
    if not summary_output.exists():
        return 0
    last_id = -1
    with summary_output.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            gid = int(row.get("game_id", -1))
            if gid > last_id:
                last_id = gid
    return max(0, last_id + 1)


def main() -> None:
    default_workers = max(1, min(8, os.cpu_count() or 1))
    parser = argparse.ArgumentParser(description="Parallel V3 model selfplay export (mirror or mixed pool)")
    parser.add_argument("--games", type=int, default=3000, help="Number of games to export")
    parser.add_argument("--workers", type=int, default=default_workers, help="Worker process count")
    parser.add_argument("--chunk-size", type=int, default=0, help="Games per job chunk (0 = auto)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--game-mode", choices=["fast", "slow"], default="slow")
    parser.add_argument("--device", type=str, default="cpu", help="cpu/cuda for worker inference")
    parser.add_argument(
        "--model-black-slow",
        type=str,
        default="artifacts/v3_transformer/models/model_v3_best_black.pt",
        help="Black slow checkpoint path",
    )
    parser.add_argument(
        "--model-white-slow",
        type=str,
        default="artifacts/v3_transformer/models/model_v3_best_white.pt",
        help="White slow checkpoint path",
    )
    parser.add_argument("--model-black-fast", type=str, default="", help="Optional black fast checkpoint path")
    parser.add_argument("--model-white-fast", type=str, default="", help="Optional white fast checkpoint path")
    parser.add_argument(
        "--v2-model-black-slow",
        type=str,
        default="artifacts/value_plus_models/model_tiny_policy_slow_best_black.pt",
        help="V2 black slow checkpoint path for v2_best opponent bucket",
    )
    parser.add_argument(
        "--v2-model-white-slow",
        type=str,
        default="artifacts/value_plus_models/model_tiny_policy_slow_best_white.pt",
        help="V2 white slow checkpoint path for v2_best opponent bucket",
    )
    parser.add_argument("--v2-model-black-fast", type=str, default="", help="Optional V2 black fast checkpoint path")
    parser.add_argument("--v2-model-white-fast", type=str, default="", help="Optional V2 white fast checkpoint path")
    parser.add_argument(
        "--trace-output",
        type=str,
        default="artifacts/v3_transformer/datasets/model_pool_raw_trace.jsonl",
        help="Output trace JSONL path",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="artifacts/v3_transformer/datasets/model_pool_games_summary.jsonl",
        help="Output game summary JSONL path",
    )
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Append to existing trace/summary instead of overwriting (for cross-round accumulation)",
    )
    parser.add_argument(
        "--pool-mode",
        choices=["mirror", "mixed"],
        default="mixed",
        help="mirror=model-vs-model only, mixed=best vs weighted opponent pool",
    )
    parser.add_argument(
        "--opponent-pool",
        type=str,
        default="random:0.2,v2best:0.1,baseline:0.3,oldbest:0.4",
        help="Used when --pool-mode=mixed; keys: random, v2best (teacher alias), baseline, oldbest",
    )
    parser.add_argument(
        "--sample-top-k",
        type=int,
        default=2,
        help="Sample from top-k model moves using softmax (0 disables sampling)",
    )
    parser.add_argument(
        "--sample-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for top-k sampling",
    )
    parser.add_argument("--decision-top-k", type=int, default=5, help="Value re-rank candidate top-k")
    parser.add_argument("--decision-alpha", type=float, default=1.0, help="Policy prior coefficient")
    parser.add_argument("--decision-beta", type=float, default=0.5, help="Value coefficient")
    parser.add_argument("--max-window", type=int, default=16, help="Temporal window length for inference")
    args = parser.parse_args()

    if args.games <= 0:
        raise ValueError("--games must be > 0")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if args.chunk_size < 0:
        raise ValueError("--chunk-size must be >= 0")
    if args.sample_top_k < 0:
        raise ValueError("--sample-top-k must be >= 0")
    if args.sample_temperature <= 0:
        raise ValueError("--sample-temperature must be > 0")
    if args.decision_top_k <= 0:
        raise ValueError("--decision-top-k must be > 0")
    if args.max_window < 2:
        raise ValueError("--max-window must be >= 2")

    model_black_slow = Path(args.model_black_slow)
    model_white_slow = Path(args.model_white_slow)
    if not model_black_slow.exists():
        raise FileNotFoundError(f"model black slow not found: {model_black_slow}")
    if not model_white_slow.exists():
        raise FileNotFoundError(f"model white slow not found: {model_white_slow}")
    model_black_fast = args.model_black_fast.strip() or str(model_black_slow)
    model_white_fast = args.model_white_fast.strip() or str(model_white_slow)
    v2_model_black_slow = Path(args.v2_model_black_slow)
    v2_model_white_slow = Path(args.v2_model_white_slow)
    if not v2_model_black_slow.exists():
        raise FileNotFoundError(f"v2 model black slow not found: {v2_model_black_slow}")
    if not v2_model_white_slow.exists():
        raise FileNotFoundError(f"v2 model white slow not found: {v2_model_white_slow}")
    v2_model_black_fast = args.v2_model_black_fast.strip() or str(v2_model_black_slow)
    v2_model_white_fast = args.v2_model_white_fast.strip() or str(v2_model_white_slow)

    workers = min(args.workers, max(1, os.cpu_count() or 1))
    jobs = _build_jobs(args.games, workers, args.chunk_size)

    trace_output = Path(args.trace_output)
    summary_output = Path(args.summary_output)
    trace_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    game_id_offset = _next_game_id(summary_output) if args.append_output else 0

    if args.pool_mode == "mirror":
        normalized_pool = {"old_best": 1.0}
    else:
        normalized_pool = _normalize_opponent_pool(_parse_opponent_pool(args.opponent_pool))

    print("--- Parallel V3 Model Selfplay Export ---")
    print(
        f"mode={args.game_mode} games={args.games} workers={workers} jobs={len(jobs)} "
        f"chunk_size={jobs[0][1] if jobs else 0} sample_top_k={args.sample_top_k} "
        f"sample_temperature={args.sample_temperature:.2f} "
        f"decision_top_k={args.decision_top_k} decision_beta={args.decision_beta:.2f} "
        f"pool_mode={args.pool_mode} start_game_id={game_id_offset}"
    )
    if args.pool_mode == "mixed":
        print(f"opponent_pool={normalized_pool}")
    print(f"model_black_slow={model_black_slow}")
    print(f"model_white_slow={model_white_slow}")
    print(f"v2_model_black_slow={v2_model_black_slow}")
    print(f"v2_model_white_slow={v2_model_white_slow}")
    print(f"trace_output={trace_output}")
    print(f"summary_output={summary_output}")

    done_games = 0
    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_steps = 0
    total_rows = 0
    opponent_counts = {"random": 0, "v2_best": 0, "baseline": 0, "old_best": 0}
    trace_mode = "a" if args.append_output else "w"
    summary_mode = "a" if args.append_output else "w"
    with trace_output.open(trace_mode, encoding="utf-8") as trace_fp, summary_output.open(
        summary_mode, encoding="utf-8"
    ) as summary_fp:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _run_chunk,
                    start_game_id=start + game_id_offset,
                    num_games=size,
                    base_seed=args.seed,
                    game_mode=args.game_mode,
                    device=args.device,
                    model_black_slow=str(model_black_slow),
                    model_white_slow=str(model_white_slow),
                    model_black_fast=model_black_fast,
                    model_white_fast=model_white_fast,
                    v2_model_black_slow=str(v2_model_black_slow),
                    v2_model_white_slow=str(v2_model_white_slow),
                    v2_model_black_fast=v2_model_black_fast,
                    v2_model_white_fast=v2_model_white_fast,
                    sample_top_k=args.sample_top_k,
                    sample_temperature=args.sample_temperature,
                    decision_top_k=args.decision_top_k,
                    decision_alpha=args.decision_alpha,
                    decision_beta=args.decision_beta,
                    max_window=args.max_window,
                    pool_mode=args.pool_mode,
                    opponent_pool=normalized_pool,
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
                    trace_lines, summary_lines, p1, p2, d, steps, chunk_counts = future.result()
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
                    for key in opponent_counts:
                        opponent_counts[key] += int(chunk_counts.get(key, 0))
                    avg_steps = total_steps / max(1, done_games)
                    _render_progress(
                        "export",
                        done_games,
                        args.games,
                        extra=(
                            f"rows={total_rows} w/d/l={p1_wins}/{draws}/{p2_wins} avg_steps={avg_steps:.2f} "
                            f"pool={opponent_counts['random']}/{opponent_counts['v2_best']}/"
                            f"{opponent_counts['baseline']}/{opponent_counts['old_best']}"
                        ),
                    )
    sys.stdout.write("\n")

    print("--- Export Summary ---")
    print(f"games={done_games} rows={total_rows} p1_wins={p1_wins} p2_wins={p2_wins} draws={draws}")
    print(f"avg_steps={total_steps / max(1, done_games):.2f}")
    print(f"opponent_counts={opponent_counts}")


if __name__ == "__main__":
    main()
