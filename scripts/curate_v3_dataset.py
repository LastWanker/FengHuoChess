"""Curate V3 balanced dataset while preserving full move sequences."""

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
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _canonical_opponent_kind(kind: str) -> str:
    key = str(kind).strip().lower().replace("-", "_")
    aliases = {
        "teacher": "v2_best",
        "teacher_ai": "v2_best",
        "teacherai": "v2_best",
        "v2best": "v2_best",
        "v2_best": "v2_best",
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
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    return {_canonical_opponent_kind(part) for part in parts}


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
    hard_case: bool,
    hard_case_weight: float,
) -> tuple[dict[str, Any], bool]:
    out = dict(row)
    if not hard_case or float(hard_case_weight) <= 1.0:
        return out, False
    base = out.get("sample_weight", 1.0)
    try:
        base_weight = float(base)
    except Exception:
        base_weight = 1.0
    if base_weight <= 0.0:
        base_weight = 1.0
    out["sample_weight"] = base_weight * float(hard_case_weight)
    return out, True


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate V3 balanced model selfplay dataset")
    parser.add_argument(
        "--trace-input",
        type=str,
        default="artifacts/v3_transformer/datasets/model_pool_raw_trace.jsonl",
    )
    parser.add_argument(
        "--summary-input",
        type=str,
        default="artifacts/v3_transformer/datasets/model_pool_games_summary.jsonl",
    )
    parser.add_argument(
        "--trace-output",
        type=str,
        default="artifacts/v3_transformer/datasets/model_pool_balanced_trace.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="artifacts/v3_transformer/datasets/model_pool_balanced_summary.jsonl",
    )
    parser.add_argument(
        "--index-output",
        type=str,
        default="artifacts/v3_transformer/cache/model_pool_balanced_index.jsonl",
        help="Per-game index cache output used by V3 temporal training.",
    )
    parser.add_argument("--split-a", type=int, default=30)
    parser.add_argument("--split-b", type=int, default=50)
    parser.add_argument("--target-per-bucket", type=int, default=0)
    parser.add_argument("--target-strategy", choices=["min", "max"], default="min")
    parser.add_argument(
        "--missing-bucket-policy",
        choices=["error", "drop"],
        default="drop",
        help="How to handle empty length/winner buckets: error=abort, drop=continue with non-empty buckets",
    )
    parser.add_argument("--mode", choices=["fast", "slow"], default="slow")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-case-weight", type=float, default=1.2)
    parser.add_argument("--hard-case-opponents", type=str, default="baseline,teacher")
    parser.add_argument("--counterexample-ratio-black", type=float, default=4.0)
    parser.add_argument("--counterexample-ratio-white", type=float, default=2.0)
    args = parser.parse_args()

    if args.split_a <= 0:
        raise ValueError("--split-a must be > 0")
    if args.split_b <= args.split_a:
        raise ValueError("--split-b must be > --split-a")
    if args.target_per_bucket < 0:
        raise ValueError("--target-per-bucket must be >= 0")
    if args.hard_case_weight <= 0:
        raise ValueError("--hard-case-weight must be > 0")
    if args.counterexample_ratio_black < 0 or args.counterexample_ratio_white < 0:
        raise ValueError("--counterexample ratios must be >= 0")

    trace_input = Path(args.trace_input)
    summary_input = Path(args.summary_input)
    trace_output = Path(args.trace_output)
    summary_output = Path(args.summary_output)
    index_output = Path(args.index_output)
    if not trace_input.exists():
        raise FileNotFoundError(f"trace input not found: {trace_input}")
    if not summary_input.exists():
        raise FileNotFoundError(f"summary input not found: {summary_input}")

    trace_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    index_output.parent.mkdir(parents=True, exist_ok=True)

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
            bucket_name = f"{length_key}_{winner_key}"
            buckets[bucket_name].append(row)

    required_buckets = [
        "steps_0_30_black",
        "steps_0_30_white",
        "steps_31_50_black",
        "steps_31_50_white",
        "steps_51_plus_black",
        "steps_51_plus_white",
    ]
    missing = [name for name in required_buckets if len(buckets.get(name, [])) == 0]
    active_buckets = list(required_buckets)
    if missing:
        if args.missing_bucket_policy == "error":
            raise RuntimeError(f"Cannot balance dataset; empty buckets: {', '.join(missing)}")
        active_buckets = [name for name in required_buckets if len(buckets.get(name, [])) > 0]
        if not active_buckets:
            raise RuntimeError("Cannot curate dataset; all buckets are empty.")
        active_black = any(name.endswith("_black") for name in active_buckets)
        active_white = any(name.endswith("_white") for name in active_buckets)
        if not active_black or not active_white:
            raise RuntimeError(
                "Cannot curate dataset safely: active buckets do not cover both colors. "
                f"missing={missing}"
            )
        print(
            f"warning=missing_buckets policy=drop missing={missing} active={active_buckets}",
            flush=True,
        )

    min_bucket = min(len(buckets[name]) for name in active_buckets)
    max_bucket = max(len(buckets[name]) for name in active_buckets)
    if args.target_per_bucket > 0:
        target = min(min_bucket, int(args.target_per_bucket))
    else:
        target = min_bucket if args.target_strategy == "min" else min(min_bucket, max_bucket)
    if target <= 0:
        raise RuntimeError("No samples available for balancing.")
    if target < 20:
        print(
            f"warning=small_target_per_bucket target={target} split={args.split_a}/{args.split_b} "
            "recommendation=lower_split_thresholds_or_more_pool_games",
            flush=True,
        )

    rng = random.Random(args.seed)
    hard_case_opponents = _parse_hard_case_opponents(args.hard_case_opponents)
    selected_games: dict[int, dict[str, Any]] = {}
    for name in active_buckets:
        picked = _sample_bucket_rows(
            rng=rng,
            rows=buckets[name],
            target=target,
            hard_case_weight=float(args.hard_case_weight),
            hard_case_opponents=hard_case_opponents,
        )
        for row in picked:
            gid = int(row["game_id"])
            selected_games[gid] = {
                "summary": row,
                "hard_case": _is_hard_case_row(row, hard_case_opponents),
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
        rng=rng, rows=buckets[counterexample_black_source], target=counterexample_black_target
    )
    counterexample_white_game_ids = _sample_game_id_set(
        rng=rng, rows=buckets[counterexample_white_source], target=counterexample_white_target
    )

    selected_ids = set(selected_games.keys()) | counterexample_black_game_ids | counterexample_white_game_ids
    sorted_ids = sorted(selected_ids)

    hard_case_games_selected = 0
    with summary_output.open("w", encoding="utf-8") as out_summary, index_output.open(
        "w", encoding="utf-8"
    ) as out_index:
        for gid in sorted_ids:
            meta = selected_games.get(gid)
            if meta is None:
                # Counterexample-only game: keep original summary shell from source bucket if available.
                summary_row = {"game_id": gid, "mode": args.mode}
                if gid in counterexample_black_game_ids:
                    summary_row["counterexample_source"] = counterexample_black_source
                if gid in counterexample_white_game_ids:
                    summary_row["counterexample_source"] = counterexample_white_source
                out_summary.write(json.dumps(summary_row, ensure_ascii=False))
                out_summary.write("\n")
                continue

            hard_case = bool(meta["hard_case"])
            if hard_case:
                hard_case_games_selected += 1
            summary_row = dict(meta["summary"])
            summary_row["hard_case"] = hard_case
            summary_row["hard_case_weight"] = float(args.hard_case_weight) if hard_case else 1.0
            summary_row["bucket"] = str(meta["bucket"])
            if gid in counterexample_black_game_ids:
                summary_row["counterexample_black_game"] = True
            if gid in counterexample_white_game_ids:
                summary_row["counterexample_white_game"] = True
            out_summary.write(json.dumps(summary_row, ensure_ascii=False))
            out_summary.write("\n")

            index_row = {
                "game_id": int(gid),
                "bucket": str(meta["bucket"]),
                "hard_case": hard_case,
                "num_steps": int(summary_row.get("num_steps", 0)),
                "winner": int(summary_row.get("winner", 0)),
            }
            out_index.write(json.dumps(index_row, ensure_ascii=False))
            out_index.write("\n")

    total_trace_rows = _count_nonempty_lines(trace_input)
    scanned_trace_rows = 0
    kept_trace_rows = 0
    weighted_rows = 0
    counterexample_rows = 0
    with trace_input.open("r", encoding="utf-8") as src, trace_output.open("w", encoding="utf-8") as dst:
        for line in src:
            raw = line.strip()
            if not raw:
                continue
            scanned_trace_rows += 1
            row = json.loads(raw)
            gid = int(row.get("game_id", -1))
            if gid not in selected_ids:
                if scanned_trace_rows % 2000 == 0 or scanned_trace_rows == total_trace_rows:
                    _render_progress(
                        "curate-v3",
                        scanned_trace_rows,
                        total_trace_rows,
                        extra=f"kept={kept_trace_rows}",
                    )
                continue

            meta = selected_games.get(gid)
            hard_case = bool(meta["hard_case"]) if meta is not None else False
            out_row, weighted = _apply_hard_case_weight_to_row(
                row=row,
                hard_case=hard_case,
                hard_case_weight=float(args.hard_case_weight),
            )
            if weighted:
                weighted_rows += 1

            player = int(out_row.get("player", 0))
            if gid in counterexample_black_game_ids and player == 1:
                out_row["counterexample"] = True
                out_row["counterexample_for"] = "black"
                out_row["counterexample_bucket"] = counterexample_black_source
                out_row["counterexample_reason"] = "black_lost_early"
                counterexample_rows += 1
            if gid in counterexample_white_game_ids and player == 2:
                out_row["counterexample"] = True
                out_row["counterexample_for"] = "white"
                out_row["counterexample_bucket"] = counterexample_white_source
                out_row["counterexample_reason"] = "white_lost_early"
                counterexample_rows += 1

            dst.write(json.dumps(out_row, ensure_ascii=False))
            dst.write("\n")
            kept_trace_rows += 1
            if scanned_trace_rows % 2000 == 0 or scanned_trace_rows == total_trace_rows:
                _render_progress(
                    "curate-v3",
                    scanned_trace_rows,
                    total_trace_rows,
                    extra=f"kept={kept_trace_rows} weighted={weighted_rows} counterexample={counterexample_rows}",
                )
    sys.stdout.write("\n")

    print("--- Curate V3 Summary ---")
    print(
        f"mode={args.mode} summary_rows_total={total_summary_rows} selected_games={len(sorted_ids)} "
        f"target_strategy={args.target_strategy} target={target}"
    )
    print(
        f"missing_bucket_policy={args.missing_bucket_policy} "
        f"active_bucket_count={len(active_buckets)}"
    )
    print(
        f"hard_case_weight={args.hard_case_weight:.3f} hard_case_opponents={sorted(hard_case_opponents)} "
        f"hard_case_games_selected={hard_case_games_selected} hard_case_rows_weighted={weighted_rows}"
    )
    print(
        f"counterexample_black_source={counterexample_black_source} ratio={args.counterexample_ratio_black:.2f} "
        f"selected_games={len(counterexample_black_game_ids)}"
    )
    print(
        f"counterexample_white_source={counterexample_white_source} ratio={args.counterexample_ratio_white:.2f} "
        f"selected_games={len(counterexample_white_game_ids)}"
    )
    active_set = set(active_buckets)
    for name in required_buckets:
        selected = target if name in active_set else 0
        print(f"bucket_{name}={len(buckets[name])} selected={selected}")
    print(f"trace_rows_total={total_trace_rows} trace_rows_kept={kept_trace_rows}")
    print(f"trace_output={trace_output}")
    print(f"summary_output={summary_output}")
    print(f"index_output={index_output}")


if __name__ == "__main__":
    main()
