"""Run model-selfplay league cycles with robust pairwise promotion checks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fenghuo_chess.ai.selfplay_league import ArenaSourceSpec, run_arena

_SEED_MOD = 2_147_483_647  # Keep seeds in 31-bit positive range for downstream libs.


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


def _status_tag(name: str, passed: bool) -> str:
    return _c_ok(f"{name}=PASS") if passed else _c_err(f"{name}=FAIL")


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


def _run_step_code(repo_root: Path, title: str, args: list[str]) -> int:
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
    return int(proc.returncode)


def _replace_cli_arg(args: list[str], key: str, value: str) -> list[str]:
    out = list(args)
    for idx, token in enumerate(out):
        if token == key:
            if idx + 1 >= len(out):
                raise ValueError(f"Missing value for argument: {key}")
            out[idx + 1] = value
            return out
    out.extend([key, value])
    return out


def _run_train_step_with_fallback(repo_root: Path, title: str, args: list[str], preferred_device: str) -> None:
    requested_device = (preferred_device or "").strip() or "cpu"
    requested_lower = requested_device.lower()
    first_args = _replace_cli_arg(args, "--device", requested_device)

    rc1 = _run_step_code(repo_root, f"{title} [attempt 1 device={requested_device}]", first_args)
    if rc1 == 0:
        return
    print(_c_warn(f"{title} first_attempt_failed device={requested_device} exit_code={rc1}"), flush=True)

    rc2 = _run_step_code(repo_root, f"{title} [attempt 2 device={requested_device}]", first_args)
    if rc2 == 0:
        return
    print(_c_warn(f"{title} retry_failed device={requested_device} exit_code={rc2}"), flush=True)

    if requested_lower != "cpu":
        cpu_args = _replace_cli_arg(args, "--device", "cpu")
        rc_cpu = _run_step_code(repo_root, f"{title} [fallback device=cpu]", cpu_args)
        if rc_cpu == 0:
            print(_c_warn(f"{title} recovered_on_cpu=true"), flush=True)
            return
        raise SystemExit(
            f"Step failed: {title} (device={requested_device} retry exit={rc2}; cpu fallback exit={rc_cpu})"
        )

    raise SystemExit(f"Step failed: {title} (exit code: {rc2})")


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


def _copy_if_missing(target: Path, fallback_candidates: list[Path], label: str) -> Path | None:
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    for src in fallback_candidates:
        if src.exists():
            shutil.copy2(src, target)
            print(_c_warn(f"{label}_copied_from={src} -> {target}"), flush=True)
            return target
    return None


def _backup_file(path: Path, round_no: int, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}.{tag}_r{round_no}_{ts}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def _canonical_pool_key(key: str) -> str:
    k = key.strip().lower().replace("-", "_")
    aliases = {
        "random": "random",
        "teacher": "teacher",
        "teacherai": "teacher",
        "teacher_ai": "teacher",
        "baseline": "baseline",
        "rule_baseline": "baseline",
        "rules_baseline": "baseline",
        "oldbest": "old_best",
        "old_best": "old_best",
        "best": "old_best",
    }
    return aliases.get(k, k)


def _parse_opponent_pool_text(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    for part in parts:
        if ":" not in part:
            raise ValueError(f"Invalid opponent pool item: {part}")
        key, value = part.split(":", 1)
        canon = _canonical_pool_key(key)
        out[canon] = out.get(canon, 0.0) + float(value.strip())
    return out


def _normalize_opponent_pool(pool: dict[str, float]) -> dict[str, float]:
    allowed = {"random", "teacher", "baseline", "old_best"}
    unknown = [k for k in pool.keys() if k not in allowed]
    if unknown:
        raise ValueError(f"Unsupported opponent_pool keys: {unknown}")
    filtered = {k: float(v) for k, v in pool.items() if float(v) > 0.0}
    if not filtered:
        raise ValueError("opponent_pool must contain at least one positive weight")
    total = float(sum(filtered.values()))
    return {k: (v / total) for k, v in filtered.items()}


def _format_opponent_pool(pool: dict[str, float]) -> str:
    ordered = ["random", "teacher", "baseline", "old_best"]
    chunks: list[str] = []
    for key in ordered:
        val = float(pool.get(key, 0.0))
        if val > 0.0:
            label = "teacherAI" if key == "teacher" else ("oldbest" if key == "old_best" else key)
            chunks.append(f"{label}:{val:.6f}")
    return ",".join(chunks)


def _lerp(start: float, end: float, progress: float) -> float:
    t = max(0.0, min(1.0, float(progress)))
    return float(start) + (float(end) - float(start)) * t


def _blend_pools(start_pool: dict[str, float], end_pool: dict[str, float], progress: float) -> dict[str, float]:
    keys = sorted(set(start_pool.keys()) | set(end_pool.keys()))
    mixed = {key: _lerp(start_pool.get(key, 0.0), end_pool.get(key, 0.0), progress) for key in keys}
    return _normalize_opponent_pool(mixed)


def _normalize_seed(seed: int) -> int:
    s = int(seed) % _SEED_MOD
    return s if s > 0 else 1


def _secondary_entropy_seed() -> int:
    # Mix OS randomness and process-level jitter as a secondary entropy source.
    sys_rand = random.SystemRandom()
    entropy_pool = [
        int.from_bytes(os.urandom(8), "little"),
        int(datetime.now().timestamp() * 1_000_000),
        os.getpid(),
        sys_rand.randrange(1, _SEED_MOD),
    ]
    pick_a = sys_rand.choice(entropy_pool)
    pick_b = sys_rand.choice(entropy_pool)
    mixed = (pick_a ^ ((pick_b << 13) | (pick_b >> 7))) & 0xFFFFFFFF
    mixed ^= sys_rand.randrange(1, _SEED_MOD)
    return _normalize_seed(mixed)


def _mix_seed(primary: int, secondary: int, salt: int = 0) -> int:
    a = (int(primary) * 1_103_515_245 + 12_345) & 0xFFFFFFFF
    b = (int(secondary) * 2_654_435_761 + int(salt)) & 0xFFFFFFFF
    x = a ^ b
    x ^= (x >> 16)
    x = (x * 2_246_822_519) & 0xFFFFFFFF
    x ^= (x >> 13)
    return _normalize_seed(x)


@dataclass
class DuelPromotionResult:
    duel_candidate_win_rate: float
    duel_avg_steps: float
    cand_black_rate: float
    best_black_rate: float
    cand_white_rate: float
    best_white_rate: float
    candidate_vs_baseline_rate: float
    best_vs_baseline_rate: float
    best_vs_baseline_source: str
    duel_passed_total: bool
    duel_passed_black: bool
    duel_passed_white: bool
    baseline_passed: bool
    promote_pair: bool
    archived_best_black: str | None
    archived_best_white: str | None


def _run_duel_and_promote(
    *,
    best_black_model: Path,
    best_white_model: Path,
    candidate_black_model: Path,
    candidate_white_model: Path,
    arena_games: int,
    arena_workers: int,
    arena_chunk_size: int,
    arena_device: str,
    arena_seed: int,
    promotion_threshold: float,
    promotion_margin: float,
    color_margin: float,
    baseline_margin: float,
    duel_sample_top_k: int,
    duel_sample_temperature: float,
    round_no: int,
    cached_best_vs_baseline_rate: float | None = None,
    cached_best_vs_baseline_source: str | None = None,
) -> DuelPromotionResult:
    best_spec = ArenaSourceSpec(
        kind="model",
        seed=arena_seed + 101,
        device=arena_device,
        model_black_slow_path=str(best_black_model),
        model_white_slow_path=str(best_white_model),
        model_black_fast_path=str(best_black_model),
        model_white_fast_path=str(best_white_model),
        explore_second_prob=0.0,
        explore_gap_threshold=0.0,
        sample_top_k=max(0, int(duel_sample_top_k)),
        sample_temperature=max(1e-6, float(duel_sample_temperature)),
    )
    cand_spec = ArenaSourceSpec(
        kind="model",
        seed=arena_seed + 211,
        device=arena_device,
        model_black_slow_path=str(candidate_black_model),
        model_white_slow_path=str(candidate_white_model),
        model_black_fast_path=str(candidate_black_model),
        model_white_fast_path=str(candidate_white_model),
        explore_second_prob=0.0,
        explore_gap_threshold=0.0,
        sample_top_k=max(0, int(duel_sample_top_k)),
        sample_temperature=max(1e-6, float(duel_sample_temperature)),
    )
    baseline_spec = ArenaSourceSpec(kind="baseline", seed=arena_seed + 307, device=arena_device)
    duel = run_arena(
        best_spec=best_spec,
        candidate_spec=cand_spec,
        games=arena_games,
        game_mode="slow",
        label="duel",
        workers=arena_workers,
        chunk_size=arena_chunk_size,
        seed=arena_seed + 1000,
    )
    candidate_vs_baseline = run_arena(
        best_spec=baseline_spec,
        candidate_spec=cand_spec,
        games=arena_games,
        game_mode="slow",
        label="cand-vs-base",
        workers=arena_workers,
        chunk_size=arena_chunk_size,
        seed=arena_seed + 2000,
    )
    if cached_best_vs_baseline_rate is None:
        best_vs_baseline = run_arena(
            best_spec=baseline_spec,
            candidate_spec=best_spec,
            games=arena_games,
            game_mode="slow",
            label="best-vs-base",
            workers=arena_workers,
            chunk_size=arena_chunk_size,
            seed=arena_seed + 2000,
        )
        best_vs_baseline_rate = float(best_vs_baseline.candidate_win_rate)
        best_vs_baseline_source = f"round{round_no}:best-vs-base(computed)"
    else:
        best_vs_baseline_rate = float(cached_best_vs_baseline_rate)
        best_vs_baseline_source = (cached_best_vs_baseline_source or "cached_best_vs_baseline").strip()
        print(_c_info(f"best-vs-base reused source={best_vs_baseline_source}"), flush=True)

    games = max(1, duel.black_round.games)
    cand_black_rate = duel.black_round.p1_wins / games
    best_black_rate = duel.white_round.p1_wins / games
    cand_white_rate = duel.white_round.p2_wins / games
    best_white_rate = duel.black_round.p2_wins / games

    duel_passed_total = duel.candidate_win_rate > (max(0.5, float(promotion_threshold)) + float(promotion_margin))
    duel_passed_black = cand_black_rate > (best_black_rate + float(color_margin))
    duel_passed_white = cand_white_rate > (best_white_rate + float(color_margin))
    baseline_passed = candidate_vs_baseline.candidate_win_rate > (
        best_vs_baseline_rate + float(baseline_margin)
    )
    promote_pair = duel_passed_total and duel_passed_black and duel_passed_white and baseline_passed

    archived_best_black: str | None = None
    archived_best_white: str | None = None
    if promote_pair:
        backup_black = _backup_file(best_black_model, round_no, "before_duel_promote_black")
        archived_best_black = str(backup_black)
        shutil.copy2(candidate_black_model, best_black_model)
        backup_white = _backup_file(best_white_model, round_no, "before_duel_promote_white")
        archived_best_white = str(backup_white)
        shutil.copy2(candidate_white_model, best_white_model)

    return DuelPromotionResult(
        duel_candidate_win_rate=duel.candidate_win_rate,
        duel_avg_steps=duel.avg_steps,
        cand_black_rate=cand_black_rate,
        best_black_rate=best_black_rate,
        cand_white_rate=cand_white_rate,
        best_white_rate=best_white_rate,
        candidate_vs_baseline_rate=candidate_vs_baseline.candidate_win_rate,
        best_vs_baseline_rate=best_vs_baseline_rate,
        best_vs_baseline_source=best_vs_baseline_source,
        duel_passed_total=duel_passed_total,
        duel_passed_black=duel_passed_black,
        duel_passed_white=duel_passed_white,
        baseline_passed=baseline_passed,
        promote_pair=promote_pair,
        archived_best_black=archived_best_black,
        archived_best_white=archived_best_white,
    )


def main() -> None:
    default_workers = max(1, min(8, os.cpu_count() or 1))
    parser = argparse.ArgumentParser(description="Run model-selfplay league cycles with robust pairwise promotion")
    parser.add_argument("--rounds", type=int, default=10, help="Number of consecutive cycles to run")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument(
        "--arena-seed",
        type=int,
        default=None,
        help="Fixed arena seed across rounds (default: same as --seed)",
    )
    parser.add_argument("--pool-games", type=int, default=2000, help="Model selfplay games per round")
    parser.add_argument("--pool-workers", type=int, default=default_workers, help="Worker processes for model selfplay export")
    parser.add_argument("--pool-chunk-size", type=int, default=0, help="Selfplay export chunk size (0=auto heuristic)")
    parser.add_argument("--pool-device", default="cpu", help="Inference device used by selfplay export workers")
    parser.add_argument("--pool-sample-top-k", type=int, default=2, help="Top-k for softmax move sampling")
    parser.add_argument(
        "--pool-sample-temperature",
        type=float,
        default=1.0,
        help="Fixed softmax temperature for pool sampling (used when temperature curriculum is off)",
    )
    parser.add_argument(
        "--pool-mode",
        choices=["mirror", "mixed"],
        default="mixed",
        help="Selfplay pool mode: mirror=model-vs-model, mixed=best vs weighted opponents",
    )
    parser.add_argument(
        "--pool-opponent-pool",
        default="random:0.2,teacherAI:0.1,baseline:0.3,oldbest:0.4",
        help="Fixed pool weights used when opponent curriculum is off",
    )
    parser.add_argument(
        "--pool-opponent-curriculum",
        choices=["on", "off"],
        default="on",
        help="Round curriculum for opponent pool weights",
    )
    parser.add_argument(
        "--pool-opponent-pool-start",
        default="random:0.2,teacherAI:0.1,baseline:0.3,oldbest:0.4",
        help="Round-1 opponent pool when opponent curriculum is on",
    )
    parser.add_argument(
        "--pool-opponent-pool-end",
        default="random:0.1,teacherAI:0.05,baseline:0.3,oldbest:0.55",
        help="Final-round opponent pool when opponent curriculum is on",
    )
    parser.add_argument(
        "--pool-temperature-curriculum",
        choices=["on", "off"],
        default="on",
        help="Round curriculum for pool sample temperature",
    )
    parser.add_argument(
        "--pool-sample-temperature-start",
        type=float,
        default=1.2,
        help="Round-1 softmax temperature when temperature curriculum is on",
    )
    parser.add_argument(
        "--pool-sample-temperature-end",
        type=float,
        default=0.9,
        help="Final-round softmax temperature when temperature curriculum is on",
    )
    parser.add_argument("--split-a", type=int, default=30, help="Length split A for curation")
    parser.add_argument("--split-b", type=int, default=50, help="Length split B for curation")
    parser.add_argument("--target-per-bucket", type=int, default=0, help="0 means auto target from target strategy")
    parser.add_argument(
        "--target-strategy",
        choices=["min", "max"],
        default="min",
        help="Auto target strategy for bucket balancing when target-per-bucket=0 (no oversampling; capped by min bucket)",
    )
    parser.add_argument(
        "--hard-case-curriculum",
        choices=["on", "off"],
        default="on",
        help="Round curriculum for hard-case sample weighting in curation",
    )
    parser.add_argument(
        "--hard-case-weight",
        type=float,
        default=1.0,
        help="Fixed hard-case weight used when hard-case curriculum is off",
    )
    parser.add_argument(
        "--hard-case-weight-start",
        type=float,
        default=1.2,
        help="Round-1 hard-case weight when hard-case curriculum is on",
    )
    parser.add_argument(
        "--hard-case-weight-end",
        type=float,
        default=2.0,
        help="Final-round hard-case weight when hard-case curriculum is on",
    )
    parser.add_argument(
        "--hard-case-opponents",
        default="baseline,teacher",
        help="Opponent kinds treated as hard-case source in curation",
    )
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs per round")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--device", default="cuda", help="Training device")
    parser.add_argument("--arena-games", type=int, default=300, help="Duel games per color assignment")
    parser.add_argument("--arena-workers", type=int, default=default_workers, help="Worker processes for duel")
    parser.add_argument("--arena-chunk-size", type=int, default=0, help="Duel chunk size (0=auto)")
    parser.add_argument("--arena-device", default="cpu", help="Device for duel workers")
    parser.add_argument("--promotion-threshold", type=float, default=0.5, help="Minimum pair win-rate baseline")
    parser.add_argument("--promotion-margin", type=float, default=0.02, help="Required duel pair-rate margin over threshold")
    parser.add_argument("--color-margin", type=float, default=0.01, help="Required per-color margin over current best")
    parser.add_argument("--baseline-margin", type=float, default=0.00, help="Required margin vs baseline relative score")
    parser.add_argument(
        "--duel-sample-top-k",
        type=int,
        default=2,
        help="Top-k sampling for duel evaluation models (0 disables sampling)",
    )
    parser.add_argument(
        "--duel-sample-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for duel sampling",
    )
    parser.add_argument(
        "--best-model-black",
        default="artifacts/value_plus_models/model_tiny_policy_slow_best_black.pt",
        help="Best black model path",
    )
    parser.add_argument(
        "--best-model-white",
        default="artifacts/value_plus_models/model_tiny_policy_slow_best_white.pt",
        help="Best white model path",
    )
    parser.add_argument(
        "--candidate-model-black",
        default="artifacts/value_plus_models/model_tiny_policy_slow_candidate_black.pt",
        help="Candidate black model output path",
    )
    parser.add_argument(
        "--candidate-model-white",
        default="artifacts/value_plus_models/model_tiny_policy_slow_candidate_white.pt",
        help="Candidate white model output path",
    )
    parser.add_argument(
        "--fallback-shared-best",
        default="artifacts/value_plus_models/model_tiny_policy_slow_best.pt",
        help="Fallback shared best checkpoint path",
    )
    parser.add_argument(
        "--fallback-init-model",
        default="artifacts/value_plus_models/model_tiny_policy_slow.pt",
        help="Fallback initialization checkpoint path",
    )
    parser.add_argument(
        "--trace-raw",
        default="artifacts/datasets/model_pool_raw_trace.jsonl",
        help="Raw selfplay trace path",
    )
    parser.add_argument(
        "--summary-raw",
        default="artifacts/datasets/model_pool_games_summary.jsonl",
        help="Raw selfplay summary path",
    )
    parser.add_argument(
        "--trace-balanced",
        default="artifacts/datasets/model_pool_balanced_trace.jsonl",
        help="Balanced selfplay trace path",
    )
    parser.add_argument(
        "--summary-balanced",
        default="artifacts/datasets/model_pool_balanced_summary.jsonl",
        help="Balanced selfplay summary path",
    )
    parser.add_argument(
        "--trace-balanced-black",
        default="artifacts/datasets/model_pool_balanced_trace_black.jsonl",
        help="Balanced trace for black training (with black counterexample bucket)",
    )
    parser.add_argument(
        "--trace-balanced-white",
        default="artifacts/datasets/model_pool_balanced_trace_white.jsonl",
        help="Balanced trace for white training (with white counterexample bucket)",
    )
    parser.add_argument(
        "--counterexample-ratio-black",
        type=float,
        default=4.0,
        help="Counterexample bucket size ratio for black (source=steps_0_30_white)",
    )
    parser.add_argument(
        "--counterexample-ratio-white",
        type=float,
        default=2.0,
        help="Counterexample bucket size ratio for white (source=steps_0_30_black)",
    )
    parser.add_argument(
        "--reset-pool-data-on-start",
        dest="reset_pool_data_on_start",
        action="store_true",
        help="Clear raw pool trace/summary once at script start",
    )
    parser.add_argument(
        "--no-reset-pool-data-on-start",
        dest="reset_pool_data_on_start",
        action="store_false",
        help="Keep existing raw pool trace/summary and continue appending",
    )
    parser.set_defaults(reset_pool_data_on_start=True)
    args = parser.parse_args()

    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    if args.pool_games <= 0:
        raise SystemExit("--pool-games must be > 0")
    if args.pool_workers <= 0:
        raise SystemExit("--pool-workers must be > 0")
    if args.pool_chunk_size < 0:
        raise SystemExit("--pool-chunk-size must be >= 0")
    if args.pool_sample_top_k < 0:
        raise SystemExit("--pool-sample-top-k must be >= 0")
    if args.pool_sample_temperature <= 0:
        raise SystemExit("--pool-sample-temperature must be > 0")
    if args.pool_sample_temperature_start <= 0:
        raise SystemExit("--pool-sample-temperature-start must be > 0")
    if args.pool_sample_temperature_end <= 0:
        raise SystemExit("--pool-sample-temperature-end must be > 0")
    if args.arena_workers <= 0:
        raise SystemExit("--arena-workers must be > 0")
    if args.arena_chunk_size < 0:
        raise SystemExit("--arena-chunk-size must be >= 0")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be > 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if not (0.0 <= args.promotion_threshold <= 1.0):
        raise SystemExit("--promotion-threshold must be in [0,1]")
    if args.promotion_margin < 0.0:
        raise SystemExit("--promotion-margin must be >= 0")
    if args.color_margin < 0.0:
        raise SystemExit("--color-margin must be >= 0")
    if args.baseline_margin < 0.0:
        raise SystemExit("--baseline-margin must be >= 0")
    if args.duel_sample_top_k < 0:
        raise SystemExit("--duel-sample-top-k must be >= 0")
    if args.duel_sample_temperature <= 0:
        raise SystemExit("--duel-sample-temperature must be > 0")
    if args.hard_case_weight <= 0:
        raise SystemExit("--hard-case-weight must be > 0")
    if args.hard_case_weight_start <= 0:
        raise SystemExit("--hard-case-weight-start must be > 0")
    if args.hard_case_weight_end <= 0:
        raise SystemExit("--hard-case-weight-end must be > 0")
    if args.counterexample_ratio_black < 0:
        raise SystemExit("--counterexample-ratio-black must be >= 0")
    if args.counterexample_ratio_white < 0:
        raise SystemExit("--counterexample-ratio-white must be >= 0")

    try:
        fixed_pool = _normalize_opponent_pool(_parse_opponent_pool_text(args.pool_opponent_pool))
        if args.pool_opponent_curriculum == "on":
            start_pool = _normalize_opponent_pool(_parse_opponent_pool_text(args.pool_opponent_pool_start))
            end_pool = _normalize_opponent_pool(_parse_opponent_pool_text(args.pool_opponent_pool_end))
        else:
            start_pool = dict(fixed_pool)
            end_pool = dict(fixed_pool)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    repo_root = ROOT
    primary_seed_base = _normalize_seed(args.seed)
    arena_primary_seed_base = primary_seed_base if args.arena_seed is None else _normalize_seed(int(args.arena_seed))
    secondary_seed_base = _secondary_entropy_seed()
    arena_secondary_seed_base = _secondary_entropy_seed()
    print(
        _c_info(
            f"seed_mode=primary+secondary "
            f"primary_seed_base={primary_seed_base} arena_primary_seed_base={arena_primary_seed_base} "
            f"secondary_seed_base={secondary_seed_base} arena_secondary_seed_base={arena_secondary_seed_base}"
        ),
        flush=True,
    )

    best_black_model = repo_root / args.best_model_black
    best_white_model = repo_root / args.best_model_white
    candidate_black_model = repo_root / args.candidate_model_black
    candidate_white_model = repo_root / args.candidate_model_white
    fallback_shared_best = repo_root / args.fallback_shared_best
    fallback_init_model = repo_root / args.fallback_init_model
    trace_raw = repo_root / args.trace_raw
    summary_raw = repo_root / args.summary_raw

    legacy_dir = repo_root / "artifacts" / "models"
    legacy_best_black = legacy_dir / "model_tiny_policy_slow_best_black.pt"
    legacy_best_white = legacy_dir / "model_tiny_policy_slow_best_white.pt"
    legacy_shared_best = legacy_dir / "model_tiny_policy_slow_best.pt"
    legacy_init_model = legacy_dir / "model_tiny_policy_slow.pt"

    # Bootstrap value-plus model directory from legacy checkpoints when needed.
    _copy_if_missing(
        fallback_init_model,
        [legacy_init_model, legacy_shared_best, legacy_best_black, legacy_best_white],
        "value_plus_init_model",
    )
    _copy_if_missing(
        fallback_shared_best,
        [legacy_shared_best, legacy_best_black, legacy_best_white, fallback_init_model],
        "value_plus_shared_best",
    )

    if args.reset_pool_data_on_start:
        if trace_raw.exists():
            trace_raw.unlink()
            print(_c_warn(f"pool_trace_reset={trace_raw}"), flush=True)
        if summary_raw.exists():
            summary_raw.unlink()
            print(_c_warn(f"pool_summary_reset={summary_raw}"), flush=True)

    _ensure_model_file(
        best_black_model,
        fallback_candidates=[
            best_black_model,
            best_white_model,
            fallback_shared_best,
            fallback_init_model,
            legacy_best_black,
            legacy_best_white,
            legacy_shared_best,
            legacy_init_model,
        ],
        label="best_black",
    )
    _ensure_model_file(
        best_white_model,
        fallback_candidates=[
            best_white_model,
            best_black_model,
            fallback_shared_best,
            fallback_init_model,
            legacy_best_white,
            legacy_best_black,
            legacy_shared_best,
            legacy_init_model,
        ],
        label="best_white",
    )
    best_vs_baseline_rate_cache: float | None = None
    best_vs_baseline_source_cache: str | None = None

    for round_idx in range(args.rounds):
        round_no = round_idx + 1
        round_primary_seed = _normalize_seed(primary_seed_base + round_idx)
        round_seed = _mix_seed(round_primary_seed, secondary_seed_base, salt=round_no)
        round_curate_seed = _mix_seed(round_primary_seed, secondary_seed_base, salt=10_000 + round_no)
        round_black_train_seed = _mix_seed(round_primary_seed, secondary_seed_base, salt=20_000 + round_no)
        round_white_train_seed = _mix_seed(round_primary_seed, secondary_seed_base, salt=30_000 + round_no)
        round_arena_primary_seed = _normalize_seed(arena_primary_seed_base + round_idx)
        round_arena_seed = _mix_seed(round_arena_primary_seed, arena_secondary_seed_base, salt=40_000 + round_no)
        round_progress = 0.0 if args.rounds <= 1 else (round_idx / float(args.rounds - 1))
        round_epochs = args.epochs + ((round_no - 1) // 2)
        round_pool = _blend_pools(start_pool, end_pool, round_progress) if args.pool_opponent_curriculum == "on" else fixed_pool
        round_pool_text = _format_opponent_pool(round_pool)
        round_pool_temperature = (
            _lerp(args.pool_sample_temperature_start, args.pool_sample_temperature_end, round_progress)
            if args.pool_temperature_curriculum == "on"
            else float(args.pool_sample_temperature)
        )
        round_hard_case_weight = (
            _lerp(args.hard_case_weight_start, args.hard_case_weight_end, round_progress)
            if args.hard_case_curriculum == "on"
            else float(args.hard_case_weight)
        )

        print()
        print(
            _paint(
                (
                    f"##### Model League Round {round_no}/{args.rounds} "
                    f"(round_seed={round_seed} arena_seed={round_arena_seed}) #####"
                ),
                "35",
            ),
            flush=True,
        )
        print(
            _c_info(
                f"progress={round_progress:.2f} pool_mode={args.pool_mode} pool={round_pool_text} "
                f"pool_temp={round_pool_temperature:.3f} hard_case_weight={round_hard_case_weight:.3f} "
                f"target_strategy={args.target_strategy} round_epochs={round_epochs} "
                f"counterexample_ratio_black={args.counterexample_ratio_black:.2f} "
                f"counterexample_ratio_white={args.counterexample_ratio_white:.2f} "
                f"seed_primary={round_primary_seed} seed_curate={round_curate_seed} "
                f"seed_train_black={round_black_train_seed} seed_train_white={round_white_train_seed}"
            ),
            flush=True,
        )

        _run_step(
            repo_root,
            f"[round {round_no}] 1/5 Export Model Selfplay Pool",
            [
                "scripts/export_model_selfplay_data.py",
                "--games",
                str(args.pool_games),
                "--workers",
                str(args.pool_workers),
                "--chunk-size",
                str(args.pool_chunk_size),
                "--seed",
                str(round_seed),
                "--game-mode",
                "slow",
                "--device",
                args.pool_device,
                "--model-black-slow",
                args.best_model_black,
                "--model-white-slow",
                args.best_model_white,
                "--trace-output",
                args.trace_raw,
                "--summary-output",
                args.summary_raw,
                "--append-output",
                "--sample-top-k",
                str(args.pool_sample_top_k),
                "--sample-temperature",
                str(round_pool_temperature),
                "--pool-mode",
                args.pool_mode,
                "--opponent-pool",
                round_pool_text,
            ],
        )
        _run_step(
            repo_root,
            f"[round {round_no}] 2/5 Curate Balanced Length Buckets",
            [
                "scripts/curate_model_dataset.py",
                "--trace-input",
                args.trace_raw,
                "--summary-input",
                args.summary_raw,
                "--trace-output",
                args.trace_balanced,
                "--summary-output",
                args.summary_balanced,
                "--trace-output-black",
                args.trace_balanced_black,
                "--trace-output-white",
                args.trace_balanced_white,
                "--split-a",
                str(args.split_a),
                "--split-b",
                str(args.split_b),
                "--target-per-bucket",
                str(args.target_per_bucket),
                "--target-strategy",
                args.target_strategy,
                "--mode",
                "slow",
                "--seed",
                str(round_curate_seed),
                "--hard-case-weight",
                str(round_hard_case_weight),
                "--hard-case-opponents",
                args.hard_case_opponents,
                "--counterexample-ratio-black",
                str(args.counterexample_ratio_black),
                "--counterexample-ratio-white",
                str(args.counterexample_ratio_white),
            ],
        )
        _run_train_step_with_fallback(
            repo_root,
            f"[round {round_no}] 3/5 Train Candidate Black",
            [
                "scripts/train_policy_model.py",
                "--trace",
                args.trace_balanced_black,
                "--mode",
                "slow",
                "--player-filter",
                "black",
                "--output",
                args.candidate_model_black,
                "--epochs",
                str(round_epochs),
                "--batch-size",
                str(args.batch_size),
                "--seed",
                str(round_black_train_seed),
                "--init-model",
                args.best_model_black,
            ],
            preferred_device=args.device,
        )
        _run_train_step_with_fallback(
            repo_root,
            f"[round {round_no}] 4/5 Train Candidate White",
            [
                "scripts/train_policy_model.py",
                "--trace",
                args.trace_balanced_white,
                "--mode",
                "slow",
                "--player-filter",
                "white",
                "--output",
                args.candidate_model_white,
                "--epochs",
                str(round_epochs),
                "--batch-size",
                str(args.batch_size),
                "--seed",
                str(round_white_train_seed),
                "--init-model",
                args.best_model_white,
            ],
            preferred_device=args.device,
        )

        print()
        print(_c_info(f"=== [round {round_no}] 5/5 Duel Promotion (candidate vs best) ==="), flush=True)
        result = _run_duel_and_promote(
            best_black_model=best_black_model,
            best_white_model=best_white_model,
            candidate_black_model=candidate_black_model,
            candidate_white_model=candidate_white_model,
            arena_games=args.arena_games,
            arena_workers=args.arena_workers,
            arena_chunk_size=args.arena_chunk_size,
            arena_device=args.arena_device,
            arena_seed=round_arena_seed,
            promotion_threshold=args.promotion_threshold,
            promotion_margin=args.promotion_margin,
            color_margin=args.color_margin,
            baseline_margin=args.baseline_margin,
            duel_sample_top_k=args.duel_sample_top_k,
            duel_sample_temperature=args.duel_sample_temperature,
            round_no=round_no,
            cached_best_vs_baseline_rate=best_vs_baseline_rate_cache,
            cached_best_vs_baseline_source=best_vs_baseline_source_cache,
        )
        print(
            f"duel_candidate_win_rate={result.duel_candidate_win_rate:.3f} avg_steps={result.duel_avg_steps:.2f} "
            f"threshold={args.promotion_threshold:.3f} margin={args.promotion_margin:.3f}"
        )
        print(
            f"black_candidate_rate={result.cand_black_rate:.3f} "
            f"black_best_rate={result.best_black_rate:.3f}"
        )
        print(
            f"white_candidate_rate={result.cand_white_rate:.3f} "
            f"white_best_rate={result.best_white_rate:.3f}"
        )
        print(
            f"candidate_vs_baseline_win_rate={result.candidate_vs_baseline_rate:.3f} "
            f"best_vs_baseline_win_rate={result.best_vs_baseline_rate:.3f} "
            f"(source={result.best_vs_baseline_source})"
        )
        print(
            " ".join(
                [
                    _status_tag("duel_total", result.duel_passed_total),
                    _status_tag("duel_black", result.duel_passed_black),
                    _status_tag("duel_white", result.duel_passed_white),
                    _status_tag("baseline", result.baseline_passed),
                    _status_tag("promote_pair", result.promote_pair),
                ]
            )
        )
        if result.archived_best_black:
            print(_c_info(f"archived_best_black={result.archived_best_black}"))
        if result.archived_best_white:
            print(_c_info(f"archived_best_white={result.archived_best_white}"))

        if result.promote_pair:
            # Next round's best is this round's candidate, so its baseline score is candidate-vs-base.
            best_vs_baseline_rate_cache = float(result.candidate_vs_baseline_rate)
            best_vs_baseline_source_cache = f"round{round_no}:cand-vs-base(promoted)"
        else:
            # Best model unchanged, keep/reuse best-vs-base score.
            best_vs_baseline_rate_cache = float(result.best_vs_baseline_rate)
            best_vs_baseline_source_cache = result.best_vs_baseline_source

    print()
    print(_c_ok(f"Model league cycles completed: {args.rounds}"))


if __name__ == "__main__":
    main()
