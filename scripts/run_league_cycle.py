"""Run slow-mode upgrade cycles with teacher + optional selfplay accumulation."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys


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


def _c_info(text: str) -> str:
    return _paint(text, "36")


def _c_ok(text: str) -> str:
    return _paint(text, "32")


def _c_warn(text: str) -> str:
    return _paint(text, "33")


def _c_err(text: str) -> str:
    return _paint(text, "31")


def _run_step(repo_root: Path, title: str, args: list[str]) -> None:
    print()
    print(_c_info(f"=== {title} ==="), flush=True)
    cmd = [sys.executable, "-u", *args]
    print(" ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        stdin=sys.stdin,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"Step failed: {title} (exit code: {proc.returncode})")


def _print_step_title(title: str) -> None:
    print()
    print(_c_info(f"=== {title} ==="), flush=True)


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    rows = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows += 1
    return rows


def _compose_training_trace(*, teacher_trace: Path, selfplay_trace: Path, output_trace: Path) -> tuple[int, int]:
    output_trace.parent.mkdir(parents=True, exist_ok=True)
    teacher_rows = 0
    selfplay_rows = 0
    with output_trace.open("w", encoding="utf-8") as out:
        if teacher_trace.exists():
            with teacher_trace.open("r", encoding="utf-8") as src:
                for line in src:
                    if not line.strip():
                        continue
                    out.write(line)
                    teacher_rows += 1
        if selfplay_trace.exists():
            with selfplay_trace.open("r", encoding="utf-8") as src:
                for line in src:
                    if not line.strip():
                        continue
                    out.write(line)
                    selfplay_rows += 1
    return teacher_rows, selfplay_rows


def _backup_best_before_train(best_path: Path, round_no: int) -> Path | None:
    if not best_path.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = best_path.with_name(f"{best_path.stem}.before_train_r{round_no}_{ts}{best_path.suffix}")
    shutil.copy2(best_path, backup)
    return backup


def _ensure_model_file(target: Path, fallback_candidates: list[Path], label: str) -> Path:
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    for src in fallback_candidates:
        if src.exists():
            shutil.copy2(src, target)
            print(_c_warn(f"{label}_bootstrapped_from={src} -> {target}"), flush=True)
            return target
    raise FileNotFoundError(f"{label} missing and no fallback exists: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one or more league training cycles (slow mode)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of consecutive cycles to run")
    parser.add_argument("--games", type=int, default=1000, help="Raw export game count")
    parser.add_argument("--workers", type=int, default=8, help="Export worker processes")
    parser.add_argument("--split-step", type=int, default=40, help="Length split for balancing")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--arena-games", type=int, default=200, help="Arena games per color assignment")
    parser.add_argument("--arena-workers", type=int, default=8, help="Worker processes for arena evaluations")
    parser.add_argument("--arena-chunk-size", type=int, default=0, help="Arena games per worker chunk (0=auto)")
    parser.add_argument("--arena-device", default="cpu", help="Device for arena workers (default cpu)")
    parser.add_argument(
        "--arena-model-explore-second-prob",
        type=float,
        default=0.0,
        help="Legacy top2 exploration probability for model policy (use 0 with softmax-only)",
    )
    parser.add_argument(
        "--arena-model-explore-gap-threshold",
        type=float,
        default=0.0,
        help="Legacy top2 exploration gap threshold (use 0 with softmax-only)",
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
    parser.add_argument("--promotion-threshold", type=float, default=0.50)
    parser.add_argument("--selfplay-games", type=int, default=300, help="Generated selfplay games per round")
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
        help="Fixed arena evaluation seed across rounds (default: same as --seed)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--best-model",
        default="artifacts/models/model_tiny_policy_slow_best.pt",
        help="Shared fallback best model path used to bootstrap black/white best models",
    )
    parser.add_argument(
        "--candidate-model",
        default="artifacts/models/model_tiny_policy_slow_candidate.pt",
        help="Shared fallback candidate model path (legacy compatibility)",
    )
    parser.add_argument(
        "--best-model-black",
        default="artifacts/models/model_tiny_policy_slow_best_black.pt",
        help="Best black-model path",
    )
    parser.add_argument(
        "--best-model-white",
        default="artifacts/models/model_tiny_policy_slow_best_white.pt",
        help="Best white-model path",
    )
    parser.add_argument(
        "--candidate-model-black",
        default="artifacts/models/model_tiny_policy_slow_candidate_black.pt",
        help="Candidate black-model output path",
    )
    parser.add_argument(
        "--candidate-model-white",
        default="artifacts/models/model_tiny_policy_slow_candidate_white.pt",
        help="Candidate white-model output path",
    )
    parser.add_argument(
        "--trace-raw",
        default="artifacts/datasets/league_cycle_raw_trace.jsonl",
        help="Raw trace output path",
    )
    parser.add_argument(
        "--summary-raw",
        default="artifacts/datasets/league_cycle_raw_summary.jsonl",
        help="Raw game summary output path",
    )
    parser.add_argument(
        "--trace-balanced",
        default="artifacts/datasets/league_cycle_balanced_trace.jsonl",
        help="Balanced trace output path",
    )
    parser.add_argument(
        "--summary-balanced",
        default="artifacts/datasets/league_cycle_balanced_summary.jsonl",
        help="Balanced game summary output path",
    )
    parser.add_argument(
        "--trace-train",
        default="artifacts/datasets/league_cycle_train_trace.jsonl",
        help="Composed training trace path: balanced teacher + per-round selfplay (default resets each round)",
    )
    parser.add_argument(
        "--trace-selfplay",
        default="artifacts/datasets/league_cycle_selfplay_trace.jsonl",
        help="Selfplay trace path (default cleared before each round)",
    )
    parser.add_argument(
        "--accumulate-selfplay-trace",
        action="store_true",
        help="Keep and append selfplay trace across rounds (legacy behavior)",
    )
    parser.add_argument(
        "--reset-selfplay-trace",
        action="store_true",
        help="Clear selfplay trace before round 1 (kept for compatibility)",
    )
    parser.add_argument(
        "--init-model-fallback",
        default="artifacts/models/model_tiny_policy_slow.pt",
        help="Fallback init model when best model does not exist",
    )
    parser.add_argument(
        "--opponent-pool",
        default="teacher:0.5,old_best:0.5",
        help="Selfplay opponent pool, default disables random opponent",
    )
    args = parser.parse_args()
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    if args.selfplay_games < 0:
        raise SystemExit("--selfplay-games must be >= 0")
    if args.arena_workers <= 0:
        raise SystemExit("--arena-workers must be > 0")
    if args.arena_chunk_size < 0:
        raise SystemExit("--arena-chunk-size must be >= 0")
    if args.selfplay_workers <= 0:
        raise SystemExit("--selfplay-workers must be > 0")
    if args.selfplay_chunk_size < 0:
        raise SystemExit("--selfplay-chunk-size must be >= 0")
    if args.model_sample_top_k < 0:
        raise SystemExit("--model-sample-top-k must be >= 0")
    if args.model_sample_temperature <= 0:
        raise SystemExit("--model-sample-temperature must be > 0")
    arena_seed = args.seed if args.arena_seed is None else int(args.arena_seed)

    repo_root = Path(__file__).resolve().parents[1]
    selfplay_trace = repo_root / args.trace_selfplay
    best_model_shared_path = repo_root / args.best_model
    best_model_black_path = repo_root / args.best_model_black
    best_model_white_path = repo_root / args.best_model_white
    candidate_model_black_path = repo_root / args.candidate_model_black
    candidate_model_white_path = repo_root / args.candidate_model_white
    init_fallback_path = repo_root / args.init_model_fallback
    if args.reset_selfplay_trace and selfplay_trace.exists():
        selfplay_trace.unlink()

    for round_idx in range(args.rounds):
        round_no = round_idx + 1
        round_seed = args.seed + round_idx
        print()
        print(_paint(f"##### League Round {round_no}/{args.rounds} (seed={round_seed}) #####", "35"), flush=True)
        if not args.accumulate_selfplay_trace and selfplay_trace.exists():
            selfplay_trace.unlink()
            print(_c_warn(f"selfplay_trace_cleared_before_round={selfplay_trace}"), flush=True)
        _run_step(
            repo_root,
            f"[round {round_no}] 1/6 Export Raw Teacher Data (slow)",
            [
                "scripts/export_teacher_data.py",
                "--games",
                str(args.games),
                "--workers",
                str(args.workers),
                "--seed",
                str(round_seed),
                "--trace-output",
                args.trace_raw,
                "--summary-output",
                args.summary_raw,
            ],
        )
        _run_step(
            repo_root,
            f"[round {round_no}] 2/6 Curate Balanced Dataset",
            [
                "scripts/curate_teacher_dataset.py",
                "--trace-input",
                args.trace_raw,
                "--summary-input",
                args.summary_raw,
                "--trace-output",
                args.trace_balanced,
                "--summary-output",
                args.summary_balanced,
                "--split-step",
                str(args.split_step),
                "--seed",
                str(round_seed),
            ],
        )
        _print_step_title(f"[round {round_no}] 3/6 Compose Training Trace")
        teacher_trace = repo_root / args.trace_balanced
        train_trace = repo_root / args.trace_train
        teacher_rows, selfplay_rows = _compose_training_trace(
            teacher_trace=teacher_trace,
            selfplay_trace=selfplay_trace,
            output_trace=train_trace,
        )
        print(
            f"composed_train_trace={train_trace} teacher_rows={teacher_rows} "
            f"selfplay_rows={selfplay_rows} total_rows={teacher_rows + selfplay_rows}",
            flush=True,
        )
        best_black = _ensure_model_file(
            best_model_black_path,
            fallback_candidates=[best_model_black_path, best_model_shared_path, init_fallback_path],
            label="best_black",
        )
        best_white = _ensure_model_file(
            best_model_white_path,
            fallback_candidates=[best_model_white_path, best_model_shared_path, init_fallback_path],
            label="best_white",
        )

        archived_black = _backup_best_before_train(best_black, round_no)
        if archived_black is not None:
            print(_c_info(f"best_black_backup_before_train={archived_black}"), flush=True)
        archived_white = _backup_best_before_train(best_white, round_no)
        if archived_white is not None:
            print(_c_info(f"best_white_backup_before_train={archived_white}"), flush=True)

        train_black_cmd = [
            "scripts/train_policy_model.py",
            "--trace",
            args.trace_train,
            "--mode",
            "slow",
            "--player-filter",
            "black",
            "--output",
            args.candidate_model_black,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--seed",
            str(round_seed),
            "--init-model",
            args.best_model_black,
        ]
        _run_step(
            repo_root,
            f"[round {round_no}] 4/6 Train Candidate Black Model",
            train_black_cmd,
        )

        train_white_cmd = [
            "scripts/train_policy_model.py",
            "--trace",
            args.trace_train,
            "--mode",
            "slow",
            "--player-filter",
            "white",
            "--output",
            args.candidate_model_white,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--seed",
            str(round_seed + 10000),
            "--init-model",
            args.best_model_white,
        ]
        _run_step(
            repo_root,
            f"[round {round_no}] 5/6 Train Candidate White Model",
            train_white_cmd,
        )
        _run_step(
            repo_root,
            f"[round {round_no}] 6/6 Arena Promotion + Selfplay Generation",
            (
                [
                "-m",
                "src.fenghuo_chess.ai.selfplay_league",
                "--best-model",
                args.best_model,
                "--candidate-model",
                args.candidate_model,
                "--best-model-black",
                args.best_model_black,
                "--best-model-white",
                args.best_model_white,
                "--candidate-model-black",
                args.candidate_model_black,
                "--candidate-model-white",
                args.candidate_model_white,
                "--arena-games",
                str(args.arena_games),
                "--arena-workers",
                str(args.arena_workers),
                "--arena-chunk-size",
                str(args.arena_chunk_size),
                "--arena-device",
                args.arena_device,
                "--selfplay-games",
                str(args.selfplay_games),
                "--selfplay-workers",
                str(args.selfplay_workers),
                "--selfplay-chunk-size",
                str(args.selfplay_chunk_size),
                "--selfplay-device",
                args.selfplay_device,
                "--promotion-threshold",
                str(args.promotion_threshold),
                "--game-mode",
                "slow",
                "--device",
                args.device,
                "--seed",
                str(round_seed),
                "--arena-seed",
                str(arena_seed),
                "--selfplay-seed",
                str(round_seed),
                "--trace-output",
                args.trace_selfplay,
                "--opponent-pool",
                args.opponent_pool,
                "--arena-model-explore-second-prob",
                str(args.arena_model_explore_second_prob),
                "--arena-model-explore-gap-threshold",
                str(args.arena_model_explore_gap_threshold),
                "--model-sample-top-k",
                str(args.model_sample_top_k),
                "--model-sample-temperature",
                str(args.model_sample_temperature),
                ]
                + (["--append-trace"] if args.accumulate_selfplay_trace else [])
            ),
        )
        print(
            _c_info(f"selfplay_trace_rows_now={_count_jsonl_rows(selfplay_trace)} path={selfplay_trace}"),
            flush=True,
        )

    print()
    print(_c_ok(f"League cycles completed: {args.rounds}"))


if __name__ == "__main__":
    main()
