"""Bootstrap V3 best checkpoints from high-quality V2 game pool."""

from __future__ import annotations

import argparse
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

_SEED_MOD = 2_147_483_647


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


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _copy_if_missing(target: Path, fallback_candidates: list[Path], label: str) -> Path | None:
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    src = _first_existing(fallback_candidates)
    if src is None:
        return None
    shutil.copy2(src, target)
    print(_c_warn(f"{label}_copied_from={src} -> {target}"), flush=True)
    return target


def _train_v3_best(
    *,
    repo_root: Path,
    title: str,
    output_best: Path,
    init_candidates: list[Path],
    trace_path: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    value_loss_weight: float,
    val_ratio: float,
    mode: str,
    window: int,
    sampling_boundary_9_10_weight: float,
    sampling_boundary_25_26_weight: float,
    sampling_critical_weight: float,
    critical_boundary_step: int,
    critical_horizon: int,
    hidden_channels: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    ffn_dim: int,
    dropout: float,
    preferred_device: str,
) -> None:
    output_best.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_best.with_name(f"{output_best.stem}.tmp{output_best.suffix}")
    init_model = _first_existing(init_candidates)

    train_args = [
        "scripts/train_v3_model.py",
        "--trace",
        trace_path,
        "--mode",
        mode,
        "--output",
        str(tmp_output),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
        "--lr",
        str(lr),
        "--value-loss-weight",
        str(value_loss_weight),
        "--val-ratio",
        str(val_ratio),
        "--window",
        str(window),
        "--sampling-boundary-9-10-weight",
        str(sampling_boundary_9_10_weight),
        "--sampling-boundary-25-26-weight",
        str(sampling_boundary_25_26_weight),
        "--sampling-critical-weight",
        str(sampling_critical_weight),
        "--critical-boundary-step",
        str(critical_boundary_step),
        "--critical-horizon",
        str(critical_horizon),
        "--hidden-channels",
        str(hidden_channels),
        "--d-model",
        str(d_model),
        "--n-heads",
        str(n_heads),
        "--n-layers",
        str(n_layers),
        "--ffn-dim",
        str(ffn_dim),
        "--dropout",
        str(dropout),
    ]
    if init_model is not None:
        train_args.extend(["--init-model", str(init_model)])

    _run_train_step_with_fallback(repo_root, title, train_args, preferred_device=preferred_device)
    if not tmp_output.exists():
        raise FileNotFoundError(f"temp V3 checkpoint missing: {tmp_output}")
    tmp_output.replace(output_best)
    print(_c_ok(f"updated_best={output_best}"), flush=True)


def main() -> None:
    default_workers = max(1, min(8, os.cpu_count() or 1))
    parser = argparse.ArgumentParser(description="Bootstrap V3 best checkpoints from V2 pool games")
    parser.add_argument("--rounds", type=int, default=6, help="Number of bootstrap rounds")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--pool-games", type=int, default=2400, help="V2 pool games per round")
    parser.add_argument("--pool-workers", type=int, default=default_workers, help="Selfplay export workers")
    parser.add_argument("--pool-chunk-size", type=int, default=0, help="Selfplay export chunk size (0=auto)")
    parser.add_argument("--pool-device", default="cpu", help="Inference device used by export workers")
    parser.add_argument("--pool-sample-top-k", type=int, default=2, help="Top-k for softmax move sampling")
    parser.add_argument("--pool-sample-temperature", type=float, default=1.0, help="Fixed pool softmax temperature")
    parser.add_argument(
        "--pool-mode",
        choices=["mirror", "mixed"],
        default="mixed",
        help="Selfplay pool mode: mirror=model-vs-model, mixed=model-vs-opponent-pool",
    )
    parser.add_argument(
        "--pool-opponent-pool",
        default="baseline:0.6,oldbest:0.4",
        help="Fixed pool weights when opponent curriculum is off",
    )
    parser.add_argument(
        "--pool-opponent-curriculum",
        choices=["on", "off"],
        default="off",
        help="Round curriculum for pool weights",
    )
    parser.add_argument(
        "--pool-opponent-pool-start",
        default="baseline:0.7,oldbest:0.3",
        help="Round-1 opponent pool when curriculum is on",
    )
    parser.add_argument(
        "--pool-opponent-pool-end",
        default="baseline:0.4,oldbest:0.6",
        help="Final-round opponent pool when curriculum is on",
    )
    parser.add_argument(
        "--pool-temperature-curriculum",
        choices=["on", "off"],
        default="off",
        help="Round curriculum for pool sample temperature",
    )
    parser.add_argument("--pool-sample-temperature-start", type=float, default=1.05)
    parser.add_argument("--pool-sample-temperature-end", type=float, default=0.95)

    parser.add_argument("--split-a", type=int, default=30, help="Length split A for curation")
    parser.add_argument("--split-b", type=int, default=50, help="Length split B for curation")
    parser.add_argument("--target-per-bucket", type=int, default=0, help="0 means auto from target strategy")
    parser.add_argument("--target-strategy", choices=["min", "max"], default="min")
    parser.add_argument(
        "--missing-bucket-policy",
        choices=["error", "drop"],
        default="drop",
        help="How to handle empty buckets during curation",
    )
    parser.add_argument(
        "--hard-case-curriculum",
        choices=["on", "off"],
        default="on",
        help="Round curriculum for hard-case weighting",
    )
    parser.add_argument("--hard-case-weight", type=float, default=1.2, help="Fixed hard-case weight")
    parser.add_argument("--hard-case-weight-start", type=float, default=1.2)
    parser.add_argument("--hard-case-weight-end", type=float, default=2.0)
    parser.add_argument("--hard-case-opponents", default="baseline,teacher")
    parser.add_argument("--counterexample-ratio-black", type=float, default=4.0)
    parser.add_argument("--counterexample-ratio-white", type=float, default=2.0)

    parser.add_argument("--epochs", type=int, default=8, help="Base training epochs per round")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--device", default="cuda", help="Training device")
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--mode", choices=["auto", "fast", "slow"], default="slow")

    parser.add_argument("--v3-train-window", type=int, default=16)
    parser.add_argument("--sampling-boundary-9-10-weight", type=float, default=1.5)
    parser.add_argument("--sampling-boundary-25-26-weight", type=float, default=2.0)
    parser.add_argument("--sampling-critical-weight", type=float, default=1.5)
    parser.add_argument("--critical-boundary-step", type=int, default=25)
    parser.add_argument("--critical-horizon", type=int, default=6)

    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument(
        "--v2-model-black",
        default="artifacts/value_plus_models/model_tiny_policy_slow_best_black.pt",
        help="V2 black checkpoint for export pool source",
    )
    parser.add_argument(
        "--v2-model-white",
        default="artifacts/value_plus_models/model_tiny_policy_slow_best_white.pt",
        help="V2 white checkpoint for export pool source",
    )
    parser.add_argument("--v2-model-black-fast", default="", help="Optional V2 black fast checkpoint")
    parser.add_argument("--v2-model-white-fast", default="", help="Optional V2 white fast checkpoint")

    parser.add_argument(
        "--best-model-black",
        default="artifacts/v3_transformer/models/model_v3_best_black.pt",
        help="V3 best black checkpoint path",
    )
    parser.add_argument(
        "--best-model-white",
        default="artifacts/v3_transformer/models/model_v3_best_white.pt",
        help="V3 best white checkpoint path",
    )
    parser.add_argument(
        "--fallback-shared-bootstrap",
        default="artifacts/v3_transformer/models/model_v3_shared_bootstrap.pt",
        help="Fallback V3 shared bootstrap checkpoint",
    )
    parser.add_argument(
        "--trace-raw",
        default="artifacts/v3_transformer/bootstrap_from_v2/datasets/model_pool_raw_trace.jsonl",
        help="Raw export trace path",
    )
    parser.add_argument(
        "--summary-raw",
        default="artifacts/v3_transformer/bootstrap_from_v2/datasets/model_pool_games_summary.jsonl",
        help="Raw export summary path",
    )
    parser.add_argument(
        "--trace-balanced",
        default="artifacts/v3_transformer/bootstrap_from_v2/datasets/model_pool_balanced_trace.jsonl",
        help="Balanced trace path",
    )
    parser.add_argument(
        "--summary-balanced",
        default="artifacts/v3_transformer/bootstrap_from_v2/datasets/model_pool_balanced_summary.jsonl",
        help="Balanced summary path",
    )
    parser.add_argument(
        "--index-balanced",
        default="artifacts/v3_transformer/bootstrap_from_v2/cache/model_pool_balanced_index.jsonl",
        help="Balanced index cache path",
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
    parser.set_defaults(reset_pool_data_on_start=False)
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
    if args.epochs <= 0:
        raise SystemExit("--epochs must be > 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.lr <= 0:
        raise SystemExit("--lr must be > 0")
    if args.value_loss_weight < 0:
        raise SystemExit("--value-loss-weight must be >= 0")
    if not (0.0 <= args.val_ratio < 1.0):
        raise SystemExit("--val-ratio must be in [0,1)")
    if args.v3_train_window < 2:
        raise SystemExit("--v3-train-window must be >= 2")
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
    if args.sampling_boundary_9_10_weight <= 0:
        raise SystemExit("--sampling-boundary-9-10-weight must be > 0")
    if args.sampling_boundary_25_26_weight <= 0:
        raise SystemExit("--sampling-boundary-25-26-weight must be > 0")
    if args.sampling_critical_weight <= 0:
        raise SystemExit("--sampling-critical-weight must be > 0")
    if args.critical_horizon < 0:
        raise SystemExit("--critical-horizon must be >= 0")
    if args.hidden_channels <= 0 or args.d_model <= 0 or args.n_heads <= 0 or args.n_layers <= 0 or args.ffn_dim <= 0:
        raise SystemExit("model dimensions must be > 0")
    if not (0.0 <= args.dropout < 1.0):
        raise SystemExit("--dropout must be in [0,1)")

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
    v2_black = repo_root / args.v2_model_black
    v2_white = repo_root / args.v2_model_white
    if not v2_black.exists():
        raise FileNotFoundError(f"v2 model black not found: {v2_black}")
    if not v2_white.exists():
        raise FileNotFoundError(f"v2 model white not found: {v2_white}")

    best_black_model = repo_root / args.best_model_black
    best_white_model = repo_root / args.best_model_white
    fallback_shared_bootstrap = repo_root / args.fallback_shared_bootstrap
    trace_raw = repo_root / args.trace_raw
    summary_raw = repo_root / args.summary_raw

    _copy_if_missing(
        best_black_model,
        fallback_candidates=[best_white_model, fallback_shared_bootstrap],
        label="v3_best_black",
    )
    _copy_if_missing(
        best_white_model,
        fallback_candidates=[best_black_model, fallback_shared_bootstrap],
        label="v3_best_white",
    )

    if args.reset_pool_data_on_start:
        if trace_raw.exists():
            trace_raw.unlink()
            print(_c_warn(f"pool_trace_reset={trace_raw}"), flush=True)
        if summary_raw.exists():
            summary_raw.unlink()
            print(_c_warn(f"pool_summary_reset={summary_raw}"), flush=True)

    primary_seed_base = _normalize_seed(args.seed)
    secondary_seed_base = _secondary_entropy_seed()
    print(
        _c_info(
            f"seed_mode=primary+secondary "
            f"primary_seed_base={primary_seed_base} secondary_seed_base={secondary_seed_base}"
        ),
        flush=True,
    )

    v2_black_fast = args.v2_model_black_fast.strip() or args.v2_model_black
    v2_white_fast = args.v2_model_white_fast.strip() or args.v2_model_white

    for round_idx in range(args.rounds):
        round_no = round_idx + 1
        round_primary_seed = _normalize_seed(primary_seed_base + round_idx)
        round_export_seed = _mix_seed(round_primary_seed, secondary_seed_base, salt=round_no)
        round_curate_seed = _mix_seed(round_primary_seed, secondary_seed_base, salt=10_000 + round_no)
        round_black_train_seed = _mix_seed(round_primary_seed, secondary_seed_base, salt=20_000 + round_no)
        round_white_train_seed = _mix_seed(round_primary_seed, secondary_seed_base, salt=30_000 + round_no)
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
                    f"##### Bootstrap V3 From V2 Round {round_no}/{args.rounds} "
                    f"(export_seed={round_export_seed}) #####"
                ),
                "35",
            ),
            flush=True,
        )
        print(
            _c_info(
                f"progress={round_progress:.2f} pool_mode={args.pool_mode} pool={round_pool_text} "
                f"pool_temp={round_pool_temperature:.3f} hard_case_weight={round_hard_case_weight:.3f} "
                f"split=({args.split_a},{args.split_b}) target_strategy={args.target_strategy} "
                f"round_epochs={round_epochs} window={args.v3_train_window}"
            ),
            flush=True,
        )

        _run_step(
            repo_root,
            f"[round {round_no}] 1/4 Export V2 Pool Games",
            [
                "scripts/export_model_selfplay_data.py",
                "--games",
                str(args.pool_games),
                "--workers",
                str(args.pool_workers),
                "--chunk-size",
                str(args.pool_chunk_size),
                "--seed",
                str(round_export_seed),
                "--game-mode",
                "slow",
                "--device",
                args.pool_device,
                "--model-black-slow",
                args.v2_model_black,
                "--model-white-slow",
                args.v2_model_white,
                "--model-black-fast",
                v2_black_fast,
                "--model-white-fast",
                v2_white_fast,
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
            f"[round {round_no}] 2/4 Curate V3 Buckets (Keep Both Colors In Sequence)",
            [
                "scripts/curate_v3_dataset.py",
                "--trace-input",
                args.trace_raw,
                "--summary-input",
                args.summary_raw,
                "--trace-output",
                args.trace_balanced,
                "--summary-output",
                args.summary_balanced,
                "--index-output",
                args.index_balanced,
                "--split-a",
                str(args.split_a),
                "--split-b",
                str(args.split_b),
                "--target-per-bucket",
                str(args.target_per_bucket),
                "--target-strategy",
                args.target_strategy,
                "--missing-bucket-policy",
                args.missing_bucket_policy,
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

        _train_v3_best(
            repo_root=repo_root,
            title=f"[round {round_no}] 3/4 Train V3 Best Black",
            output_best=best_black_model,
            init_candidates=[best_black_model, best_white_model, fallback_shared_bootstrap],
            trace_path=args.trace_balanced,
            seed=round_black_train_seed,
            epochs=round_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            value_loss_weight=args.value_loss_weight,
            val_ratio=args.val_ratio,
            mode=args.mode,
            window=args.v3_train_window,
            sampling_boundary_9_10_weight=args.sampling_boundary_9_10_weight,
            sampling_boundary_25_26_weight=args.sampling_boundary_25_26_weight,
            sampling_critical_weight=args.sampling_critical_weight,
            critical_boundary_step=args.critical_boundary_step,
            critical_horizon=args.critical_horizon,
            hidden_channels=args.hidden_channels,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
            preferred_device=args.device,
        )

        _train_v3_best(
            repo_root=repo_root,
            title=f"[round {round_no}] 4/4 Train V3 Best White",
            output_best=best_white_model,
            init_candidates=[best_white_model, best_black_model, fallback_shared_bootstrap],
            trace_path=args.trace_balanced,
            seed=round_white_train_seed,
            epochs=round_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            value_loss_weight=args.value_loss_weight,
            val_ratio=args.val_ratio,
            mode=args.mode,
            window=args.v3_train_window,
            sampling_boundary_9_10_weight=args.sampling_boundary_9_10_weight,
            sampling_boundary_25_26_weight=args.sampling_boundary_25_26_weight,
            sampling_critical_weight=args.sampling_critical_weight,
            critical_boundary_step=args.critical_boundary_step,
            critical_horizon=args.critical_horizon,
            hidden_channels=args.hidden_channels,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
            preferred_device=args.device,
        )

    print()
    print(_c_ok(f"Bootstrap V3 from V2 completed: rounds={args.rounds}"), flush=True)
    print(_c_info(f"best_black={best_black_model}"), flush=True)
    print(_c_info(f"best_white={best_white_model}"), flush=True)


if __name__ == "__main__":
    main()
