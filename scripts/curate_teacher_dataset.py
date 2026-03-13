"""Curate balanced teacher dataset by game length and winner color."""

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


def _bucket_name(*, num_steps: int, winner: int, split_step: int) -> str | None:
    if winner not in {1, 2}:
        return None
    length_key = "short" if num_steps <= split_step else "long"
    color_key = "black" if winner == 1 else "white"
    return f"{length_key}_{color_key}"


def _count_nonempty_lines(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate balanced teacher dataset from raw export outputs")
    parser.add_argument(
        "--trace-input",
        type=str,
        default="artifacts/datasets/teacher_trace_raw.jsonl",
        help="Raw trace JSONL path",
    )
    parser.add_argument(
        "--summary-input",
        type=str,
        default="artifacts/datasets/teacher_games_summary.jsonl",
        help="Per-game summary JSONL path",
    )
    parser.add_argument(
        "--trace-output",
        type=str,
        default="artifacts/datasets/teacher_trace_balanced.jsonl",
        help="Curated balanced trace JSONL path",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="artifacts/datasets/teacher_games_balanced.jsonl",
        help="Curated balanced summary JSONL path",
    )
    parser.add_argument("--split-step", type=int, default=40, help="Length split boundary")
    parser.add_argument("--target-per-bucket", type=int, default=0, help="0 means auto=min bucket size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.split_step <= 0:
        raise ValueError("--split-step must be > 0")
    if args.target_per_bucket < 0:
        raise ValueError("--target-per-bucket must be >= 0")

    trace_input = Path(args.trace_input)
    summary_input = Path(args.summary_input)
    trace_output = Path(args.trace_output)
    summary_output = Path(args.summary_output)
    if not trace_input.exists():
        raise FileNotFoundError(f"trace input not found: {trace_input}")
    if not summary_input.exists():
        raise FileNotFoundError(f"summary input not found: {summary_input}")
    trace_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_summary_rows = 0
    with summary_input.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            total_summary_rows += 1
            row = json.loads(raw)
            if str(row.get("mode", "")) != "slow":
                continue
            winner = int(row.get("winner", 0))
            num_steps = int(row.get("num_steps", 0))
            bucket = _bucket_name(num_steps=num_steps, winner=winner, split_step=args.split_step)
            if bucket is None:
                continue
            buckets[bucket].append(row)

    required_buckets = ["short_black", "short_white", "long_black", "long_white"]
    missing = [name for name in required_buckets if len(buckets.get(name, [])) == 0]
    if missing:
        raise RuntimeError(f"Cannot balance dataset; empty buckets: {', '.join(missing)}")

    min_bucket = min(len(buckets[name]) for name in required_buckets)
    target = min_bucket if args.target_per_bucket == 0 else min(min_bucket, args.target_per_bucket)
    if target <= 0:
        raise RuntimeError("No samples available for balancing.")

    rng = random.Random(args.seed)
    selected_games: dict[int, dict[str, Any]] = {}
    for name in required_buckets:
        rows = buckets[name]
        picked = rng.sample(rows, target)
        for row in picked:
            selected_games[int(row["game_id"])] = row

    selected_ids = set(selected_games.keys())
    sorted_selected_ids = sorted(selected_ids)

    with summary_output.open("w", encoding="utf-8") as out:
        for game_id in sorted_selected_ids:
            out.write(json.dumps(selected_games[game_id], ensure_ascii=False))
            out.write("\n")

    total_trace_rows = _count_nonempty_lines(trace_input)
    scanned_trace_rows = 0
    kept_trace_rows = 0
    with trace_input.open("r", encoding="utf-8") as src, trace_output.open("w", encoding="utf-8") as dst:
        for line in src:
            raw = line.strip()
            if not raw:
                continue
            scanned_trace_rows += 1
            row = json.loads(raw)
            if int(row.get("game_id", -1)) not in selected_ids:
                if scanned_trace_rows % 2000 == 0 or scanned_trace_rows == total_trace_rows:
                    _render_progress(
                        "curate",
                        scanned_trace_rows,
                        total_trace_rows,
                        extra=f"kept_rows={kept_trace_rows}",
                    )
                continue
            dst.write(raw)
            dst.write("\n")
            kept_trace_rows += 1
            if scanned_trace_rows % 2000 == 0 or scanned_trace_rows == total_trace_rows:
                _render_progress(
                    "curate",
                    scanned_trace_rows,
                    total_trace_rows,
                    extra=f"kept_rows={kept_trace_rows}",
                )
    sys.stdout.write("\n")

    print("--- Curate Summary ---")
    print(f"summary_rows_total={total_summary_rows} selected_games={len(selected_ids)}")
    for name in required_buckets:
        print(f"bucket_{name}={len(buckets[name])} selected={target}")
    print(f"trace_rows_total={total_trace_rows} trace_rows_kept={kept_trace_rows}")
    print(f"trace_output={trace_output}")
    print(f"summary_output={summary_output}")


if __name__ == "__main__":
    main()
