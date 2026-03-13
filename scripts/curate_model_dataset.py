"""Curate balanced model-selfplay dataset by (length x winner-color) buckets."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import random
import sys
from pathlib import Path
from typing import Any


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


def _length_bucket(num_steps: int, *, split_a: int, split_b: int) -> str:
    if num_steps <= split_a:
        return "steps_0_30"
    if num_steps <= split_b:
        return "steps_31_50"
    return "steps_51_plus"


def _count_nonempty_lines(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


def _canonical_opponent_kind(kind: str) -> str:
    key = str(kind).strip().lower().replace("-", "_")
    aliases = {
        "teacher": "teacher",
        "teacher_ai": "teacher",
        "teacherai": "teacher",
        "baseline": "baseline",
        "rule_baseline": "baseline",
        "rules_baseline": "baseline",
        "random": "random",
        "old_best": "old_best",
        "oldbest": "old_best",
        "best": "old_best",
    }
    return aliases.get(key, key)


def _parse_hard_case_opponents(text: str) -> set[str]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    out: set[str] = set()
    for part in parts:
        out.add(_canonical_opponent_kind(part))
    return out


def _is_hard_case_row(row: dict[str, Any], hard_case_opponents: set[str]) -> bool:
    if not hard_case_opponents:
        return False
    kind = _canonical_opponent_kind(str(row.get("opponent_kind", "")))
    if kind not in hard_case_opponents:
        return False
    try:
        best_outcome = int(row.get("best_outcome", 0))
    except Exception:
        return False
    return best_outcome < 0


def _sample_bucket_rows(
    *,
    rng: random.Random,
    rows: list[dict[str, Any]],
    target: int,
    hard_case_weight: float,
    hard_case_opponents: set[str],
) -> list[dict[str, Any]]:
    if target <= 0:
        return []
    if target >= len(rows):
        return list(rows)
    if hard_case_weight <= 1.0 or not hard_case_opponents:
        return rng.sample(rows, target)

    pool = list(rows)
    picked: list[dict[str, Any]] = []
    for _ in range(target):
        weights = [
            float(hard_case_weight) if _is_hard_case_row(row, hard_case_opponents) else 1.0
            for row in pool
        ]
        total = float(sum(weights))
        if total <= 0.0:
            idx = rng.randrange(len(pool))
        else:
            point = rng.random() * total
            acc = 0.0
            idx = len(pool) - 1
            for i, w in enumerate(weights):
                acc += float(w)
                if point <= acc:
                    idx = i
                    break
        picked.append(pool.pop(idx))
    return picked


def _sample_game_id_set(*, rng: random.Random, rows: list[dict[str, Any]], target: int) -> set[int]:
    if target <= 0 or not rows:
        return set()
    game_ids = [int(row.get("game_id", -1)) for row in rows]
    game_ids = [gid for gid in game_ids if gid >= 0]
    if target >= len(game_ids):
        return set(game_ids)
    return set(rng.sample(game_ids, target))


def _apply_hard_case_weight_to_row(
    *,
    row: dict[str, Any],
    meta: dict[str, Any] | None,
    hard_case_weight: float,
) -> tuple[dict[str, Any], bool]:
    out = dict(row)
    if not meta:
        return out, False
    if not bool(meta.get("hard_case", False)):
        return out, False
    player = int(out.get("player", 0))
    best_player = int(meta.get("best_player", 0))
    if best_player not in {1, 2} or player != best_player or float(hard_case_weight) <= 1.0:
        return out, False
    base_weight = out.get("sample_weight", 1.0)
    try:
        base_weight_value = float(base_weight)
    except Exception:
        base_weight_value = 1.0
    if base_weight_value <= 0.0:
        base_weight_value = 1.0
    out["sample_weight"] = base_weight_value * float(hard_case_weight)
    return out, True


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate balanced model selfplay dataset")
    parser.add_argument(
        "--trace-input",
        type=str,
        default="artifacts/datasets/model_pool_raw_trace.jsonl",
        help="Raw trace JSONL path",
    )
    parser.add_argument(
        "--summary-input",
        type=str,
        default="artifacts/datasets/model_pool_games_summary.jsonl",
        help="Per-game summary JSONL path",
    )
    parser.add_argument(
        "--trace-output",
        type=str,
        default="artifacts/datasets/model_pool_balanced_trace.jsonl",
        help="Curated balanced trace JSONL path",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="artifacts/datasets/model_pool_balanced_summary.jsonl",
        help="Curated balanced summary JSONL path",
    )
    parser.add_argument(
        "--trace-output-black",
        type=str,
        default="artifacts/datasets/model_pool_balanced_trace_black.jsonl",
        help="Curated training trace for black model (includes black counterexample bucket)",
    )
    parser.add_argument(
        "--trace-output-white",
        type=str,
        default="artifacts/datasets/model_pool_balanced_trace_white.jsonl",
        help="Curated training trace for white model (includes white counterexample bucket)",
    )
    parser.add_argument("--split-a", type=int, default=30, help="First split boundary (inclusive)")
    parser.add_argument("--split-b", type=int, default=50, help="Second split boundary (inclusive)")
    parser.add_argument("--target-per-bucket", type=int, default=0, help="0 means auto target from strategy")
    parser.add_argument(
        "--target-strategy",
        choices=["min", "max"],
        default="min",
        help="Auto strategy when target-per-bucket=0 (no oversampling; effective target is capped by min bucket)",
    )
    parser.add_argument("--mode", choices=["fast", "slow"], default="slow")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hard-case-weight",
        type=float,
        default=1.0,
        help="Boost factor for baseline/teacher hard-case games (best_outcome<0). 1.0 disables boost.",
    )
    parser.add_argument(
        "--hard-case-opponents",
        type=str,
        default="baseline,teacher",
        help="Comma-separated opponent kinds for hard-case detection",
    )
    parser.add_argument(
        "--counterexample-ratio-black",
        type=float,
        default=4.0,
        help=(
            "Counterexample game target for black as ratio x balanced target; "
            "source bucket is steps_0_30_white (black loses fast)"
        ),
    )
    parser.add_argument(
        "--counterexample-ratio-white",
        type=float,
        default=2.0,
        help=(
            "Counterexample game target for white as ratio x balanced target; "
            "source bucket is steps_0_30_black (white loses fast)"
        ),
    )
    args = parser.parse_args()

    if args.split_a <= 0:
        raise ValueError("--split-a must be > 0")
    if args.split_b <= args.split_a:
        raise ValueError("--split-b must be > --split-a")
    if args.target_per_bucket < 0:
        raise ValueError("--target-per-bucket must be >= 0")
    if args.hard_case_weight <= 0:
        raise ValueError("--hard-case-weight must be > 0")
    if args.counterexample_ratio_black < 0:
        raise ValueError("--counterexample-ratio-black must be >= 0")
    if args.counterexample_ratio_white < 0:
        raise ValueError("--counterexample-ratio-white must be >= 0")

    trace_input = Path(args.trace_input)
    summary_input = Path(args.summary_input)
    trace_output = Path(args.trace_output)
    summary_output = Path(args.summary_output)
    trace_output_black = Path(args.trace_output_black)
    trace_output_white = Path(args.trace_output_white)
    if not trace_input.exists():
        raise FileNotFoundError(f"trace input not found: {trace_input}")
    if not summary_input.exists():
        raise FileNotFoundError(f"summary input not found: {summary_input}")
    trace_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    trace_output_black.parent.mkdir(parents=True, exist_ok=True)
    trace_output_white.parent.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_summary_rows = 0
    with summary_input.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            total_summary_rows += 1
            row = json.loads(raw)
            if str(row.get("mode", "")) != args.mode:
                continue
            winner = int(row.get("winner", 0))
            if winner not in {1, 2}:
                continue
            winner_key = "black" if winner == 1 else "white"
            num_steps = int(row.get("num_steps", 0))
            length_key = _length_bucket(num_steps, split_a=args.split_a, split_b=args.split_b)
            bucket = f"{length_key}_{winner_key}"
            buckets[bucket].append(row)

    required_buckets = [
        "steps_0_30_black",
        "steps_0_30_white",
        "steps_31_50_black",
        "steps_31_50_white",
        "steps_51_plus_black",
        "steps_51_plus_white",
    ]
    missing = [name for name in required_buckets if len(buckets.get(name, [])) == 0]
    if missing:
        raise RuntimeError(f"Cannot balance dataset; empty buckets: {', '.join(missing)}")

    min_bucket = min(len(buckets[name]) for name in required_buckets)
    max_bucket = max(len(buckets[name]) for name in required_buckets)
    if args.target_per_bucket > 0:
        target = min(min_bucket, int(args.target_per_bucket))
    else:
        # Keep no-duplication curation: even with accumulated data, sampling remains capped by min bucket.
        target = min_bucket if args.target_strategy == "min" else min(min_bucket, max_bucket)
    if target <= 0:
        raise RuntimeError("No samples available for balancing.")

    rng = random.Random(args.seed)
    hard_case_opponents = _parse_hard_case_opponents(args.hard_case_opponents)
    selected_games: dict[int, dict[str, Any]] = {}
    for name in required_buckets:
        rows = buckets[name]
        picked = _sample_bucket_rows(
            rng=rng,
            rows=rows,
            target=target,
            hard_case_weight=float(args.hard_case_weight),
            hard_case_opponents=hard_case_opponents,
        )
        for row in picked:
            game_id = int(row["game_id"])
            selected_games[game_id] = {
                "summary": row,
                "hard_case": _is_hard_case_row(row, hard_case_opponents),
                "best_player": int(row.get("best_player", 0)),
                "bucket": name,
            }

    counterexample_black_source = "steps_0_30_white"
    counterexample_white_source = "steps_0_30_black"
    counterexample_black_target = min(
        len(buckets[counterexample_black_source]),
        max(0, int(round(float(args.counterexample_ratio_black) * target))),
    )
    counterexample_white_target = min(
        len(buckets[counterexample_white_source]),
        max(0, int(round(float(args.counterexample_ratio_white) * target))),
    )
    counterexample_black_game_ids = _sample_game_id_set(
        rng=rng,
        rows=buckets[counterexample_black_source],
        target=counterexample_black_target,
    )
    counterexample_white_game_ids = _sample_game_id_set(
        rng=rng,
        rows=buckets[counterexample_white_source],
        target=counterexample_white_target,
    )

    selected_ids = set(selected_games.keys())
    sorted_selected_ids = sorted(selected_ids)
    hard_case_games_selected = 0
    with summary_output.open("w", encoding="utf-8") as out:
        for game_id in sorted_selected_ids:
            meta = selected_games[game_id]
            summary_row = dict(meta["summary"])
            hard_case = bool(meta["hard_case"])
            if hard_case:
                hard_case_games_selected += 1
            summary_row["hard_case"] = hard_case
            summary_row["hard_case_weight"] = float(args.hard_case_weight) if hard_case else 1.0
            out.write(json.dumps(summary_row, ensure_ascii=False))
            out.write("\n")

    total_trace_rows = _count_nonempty_lines(trace_input)
    scanned_trace_rows = 0
    kept_trace_rows = 0
    kept_trace_rows_black = 0
    kept_trace_rows_white = 0
    hard_case_rows_weighted = 0
    black_counterexample_rows = 0
    white_counterexample_rows = 0
    with (
        trace_input.open("r", encoding="utf-8") as src,
        trace_output.open("w", encoding="utf-8") as dst,
        trace_output_black.open("w", encoding="utf-8") as dst_black,
        trace_output_white.open("w", encoding="utf-8") as dst_white,
    ):
        for line in src:
            raw = line.strip()
            if not raw:
                continue
            scanned_trace_rows += 1
            row = json.loads(raw)
            game_id = int(row.get("game_id", -1))
            player = int(row.get("player", 0))
            meta = selected_games.get(game_id)
            in_selected = meta is not None
            in_black_counterexample = game_id in counterexample_black_game_ids
            in_white_counterexample = game_id in counterexample_white_game_ids

            if not in_selected and not in_black_counterexample and not in_white_counterexample:
                if scanned_trace_rows % 2000 == 0 or scanned_trace_rows == total_trace_rows:
                    _render_progress(
                        "curate",
                        scanned_trace_rows,
                        total_trace_rows,
                        extra=(
                            f"kept_shared={kept_trace_rows} "
                            f"kept_black={kept_trace_rows_black} kept_white={kept_trace_rows_white}"
                        ),
                    )
                continue

            if in_selected:
                shared_row, shared_weighted = _apply_hard_case_weight_to_row(
                    row=row,
                    meta=meta,
                    hard_case_weight=float(args.hard_case_weight),
                )
                if shared_weighted:
                    hard_case_rows_weighted += 1
                dst.write(json.dumps(shared_row, ensure_ascii=False))
                dst.write("\n")
                kept_trace_rows += 1

            if player == 1 and (in_selected or in_black_counterexample):
                if in_black_counterexample:
                    black_row = dict(row)
                    black_row["counterexample"] = True
                    black_row["counterexample_for"] = "black"
                    black_row["counterexample_bucket"] = counterexample_black_source
                    black_row["counterexample_reason"] = "black_lost_early"
                    black_counterexample_rows += 1
                else:
                    black_row, _ = _apply_hard_case_weight_to_row(
                        row=row,
                        meta=meta,
                        hard_case_weight=float(args.hard_case_weight),
                    )
                dst_black.write(json.dumps(black_row, ensure_ascii=False))
                dst_black.write("\n")
                kept_trace_rows_black += 1

            if player == 2 and (in_selected or in_white_counterexample):
                if in_white_counterexample:
                    white_row = dict(row)
                    white_row["counterexample"] = True
                    white_row["counterexample_for"] = "white"
                    white_row["counterexample_bucket"] = counterexample_white_source
                    white_row["counterexample_reason"] = "white_lost_early"
                    white_counterexample_rows += 1
                else:
                    white_row, _ = _apply_hard_case_weight_to_row(
                        row=row,
                        meta=meta,
                        hard_case_weight=float(args.hard_case_weight),
                    )
                dst_white.write(json.dumps(white_row, ensure_ascii=False))
                dst_white.write("\n")
                kept_trace_rows_white += 1

            if scanned_trace_rows % 2000 == 0 or scanned_trace_rows == total_trace_rows:
                _render_progress(
                    "curate",
                    scanned_trace_rows,
                    total_trace_rows,
                    extra=(
                        f"kept_shared={kept_trace_rows} "
                        f"kept_black={kept_trace_rows_black} kept_white={kept_trace_rows_white}"
                    ),
                )
    sys.stdout.write("\n")

    print("--- Curate Summary ---")
    print(
        f"mode={args.mode} summary_rows_total={total_summary_rows} selected_games={len(selected_ids)} "
        f"target_strategy={args.target_strategy}"
    )
    print(
        f"hard_case_weight={args.hard_case_weight:.3f} hard_case_opponents={sorted(hard_case_opponents)} "
        f"hard_case_games_selected={hard_case_games_selected} hard_case_rows_weighted={hard_case_rows_weighted}"
    )
    print(
        f"counterexample_black_source={counterexample_black_source} ratio={args.counterexample_ratio_black:.2f} "
        f"target={counterexample_black_target} selected_games={len(counterexample_black_game_ids)} "
        f"rows_marked={black_counterexample_rows}"
    )
    print(
        f"counterexample_white_source={counterexample_white_source} ratio={args.counterexample_ratio_white:.2f} "
        f"target={counterexample_white_target} selected_games={len(counterexample_white_game_ids)} "
        f"rows_marked={white_counterexample_rows}"
    )
    print(f"bucket_min={min_bucket} bucket_max={max_bucket} target={target}")
    for name in required_buckets:
        print(f"bucket_{name}={len(buckets[name])} selected={target}")
    print(
        f"trace_rows_total={total_trace_rows} trace_rows_kept_shared={kept_trace_rows} "
        f"trace_rows_kept_black={kept_trace_rows_black} trace_rows_kept_white={kept_trace_rows_white}"
    )
    print(f"trace_output={trace_output}")
    print(f"trace_output_black={trace_output_black}")
    print(f"trace_output_white={trace_output_white}")
    print(f"summary_output={summary_output}")


if __name__ == "__main__":
    main()
