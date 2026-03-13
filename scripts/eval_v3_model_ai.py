"""Evaluate V3 model AI against baseline/weak opponents in headless EVE."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fenghuo_chess.ai.baselines.heuristic_ai import HeuristicAISource
from src.fenghuo_chess.ai.baselines.random_ai import RandomAISource
from src.fenghuo_chess.ai.model_v3 import ModelAISourceV3
from src.fenghuo_chess.ai.selfplay_runner import SelfPlaySummary, run_selfplay
from src.fenghuo_chess.application.match_runner import build_match_runner


def _make_opponent(kind: str, seed: int) -> object:
    key = kind.strip().lower()
    if key == "weak":
        return RandomAISource(rng=random.Random(seed))
    if key == "baseline":
        return HeuristicAISource(rng=random.Random(seed))
    raise ValueError(f"Unsupported opponent: {kind}. Supported: weak, baseline")


def _print_round(label: str, summary: SelfPlaySummary) -> None:
    print(
        f"{label}: games={summary.games} "
        f"p1_wins={summary.p1_wins} p2_wins={summary.p2_wins} draws={summary.draws} steps={summary.steps}"
    )


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


def _first_existing(candidates: list[Path]) -> str:
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


def _resolve_paths(args: argparse.Namespace) -> dict[str, str]:
    repo_root = ROOT
    default_black_slow = _first_existing(
        [
            repo_root / "artifacts/v3_transformer/models/model_v3_best_black.pt",
            repo_root / "artifacts/v3_transformer/models/model_v3_shared_bootstrap.pt",
        ]
    )
    default_white_slow = _first_existing(
        [
            repo_root / "artifacts/v3_transformer/models/model_v3_best_white.pt",
            repo_root / "artifacts/v3_transformer/models/model_v3_shared_bootstrap.pt",
            repo_root / "artifacts/v3_transformer/models/model_v3_best_black.pt",
        ]
    )
    default_black_fast = _first_existing(
        [
            repo_root / "artifacts/v3_transformer/models/model_v3_best_black.pt",
            repo_root / "artifacts/v3_transformer/models/model_v3_shared_bootstrap.pt",
        ]
    )
    default_white_fast = _first_existing(
        [
            repo_root / "artifacts/v3_transformer/models/model_v3_best_white.pt",
            repo_root / "artifacts/v3_transformer/models/model_v3_shared_bootstrap.pt",
            repo_root / "artifacts/v3_transformer/models/model_v3_best_black.pt",
        ]
    )

    shared = args.model.strip()
    common_fast = args.model_fast.strip() or shared
    common_slow = args.model_slow.strip() or shared
    black_base = args.model_black.strip() or shared
    white_base = args.model_white.strip() or shared

    black_slow = args.model_black_slow.strip() or black_base or common_slow or default_black_slow
    white_slow = args.model_white_slow.strip() or white_base or common_slow or default_white_slow or default_black_slow
    black_fast = args.model_black_fast.strip() or black_base or common_fast or default_black_fast or black_slow
    white_fast = args.model_white_fast.strip() or white_base or common_fast or default_white_fast or white_slow

    return {
        "shared": shared,
        "common_fast": common_fast,
        "common_slow": common_slow,
        "black_fast": black_fast,
        "black_slow": black_slow,
        "white_fast": white_fast,
        "white_slow": white_slow,
    }


def _validate_model_paths(paths: dict[str, str]) -> None:
    chosen = [paths["black_fast"], paths["black_slow"], paths["white_fast"], paths["white_slow"]]
    if not any(chosen):
        raise ValueError(
            "No model checkpoint resolved. Provide --model or color-specific paths, "
            "or place checkpoints under artifacts/v3_transformer/models."
        )
    for key in ("black_fast", "black_slow", "white_fast", "white_slow"):
        path = paths[key].strip()
        if path and not Path(path).exists():
            raise FileNotFoundError(f"{key} not found: {path}")


def _build_model_source(
    paths: dict[str, str],
    device: str,
    *,
    sample_top_k: int,
    sample_temperature: float,
    decision_top_k: int,
    decision_alpha: float,
    decision_beta: float,
    max_window: int,
) -> ModelAISourceV3:
    anchor = paths["black_slow"] or paths["black_fast"] or paths["white_slow"] or paths["white_fast"]
    return ModelAISourceV3(
        model_path=anchor,
        model_fast_path=paths["common_fast"] or anchor,
        model_slow_path=paths["common_slow"] or anchor,
        model_black_fast_path=paths["black_fast"],
        model_black_slow_path=paths["black_slow"],
        model_white_fast_path=paths["white_fast"],
        model_white_slow_path=paths["white_slow"],
        device=device,
        sample_top_k=max(0, int(sample_top_k)),
        sample_temperature=max(1e-6, float(sample_temperature)),
        decision_top_k=max(1, int(decision_top_k)),
        decision_alpha=float(decision_alpha),
        decision_beta=float(decision_beta),
        max_window=max(2, int(max_window)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ModelAISourceV3 in headless EVE")
    parser.add_argument("--model", type=str, default="", help="Shared checkpoint fallback (.pt)")
    parser.add_argument("--model-fast", type=str, default="", help="Shared fast-mode checkpoint (.pt)")
    parser.add_argument("--model-slow", type=str, default="", help="Shared slow-mode checkpoint (.pt)")
    parser.add_argument("--model-black", type=str, default="", help="Shared black checkpoint fallback (.pt)")
    parser.add_argument("--model-white", type=str, default="", help="Shared white checkpoint fallback (.pt)")
    parser.add_argument("--model-black-fast", type=str, default="", help="Black fast-mode checkpoint (.pt)")
    parser.add_argument("--model-black-slow", type=str, default="", help="Black slow-mode checkpoint (.pt)")
    parser.add_argument("--model-white-fast", type=str, default="", help="White fast-mode checkpoint (.pt)")
    parser.add_argument("--model-white-slow", type=str, default="", help="White slow-mode checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    parser.add_argument("--opponent", choices=["weak", "baseline"], default="baseline")
    parser.add_argument("--game-mode", choices=["fast", "slow"], default="slow")
    parser.add_argument("--games", type=int, default=30, help="Games per color assignment")
    parser.add_argument("--sample-top-k", type=int, default=2)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--decision-top-k", type=int, default=5)
    parser.add_argument("--decision-alpha", type=float, default=1.0)
    parser.add_argument("--decision-beta", type=float, default=0.5)
    parser.add_argument("--max-window", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.games <= 0:
        raise ValueError("--games must be > 0")
    if args.sample_top_k < 0:
        raise ValueError("--sample-top-k must be >= 0")
    if args.sample_temperature <= 0:
        raise ValueError("--sample-temperature must be > 0")

    paths = _resolve_paths(args)
    _validate_model_paths(paths)

    candidate_black = _build_model_source(
        paths,
        args.device,
        sample_top_k=args.sample_top_k,
        sample_temperature=args.sample_temperature,
        decision_top_k=args.decision_top_k,
        decision_alpha=args.decision_alpha,
        decision_beta=args.decision_beta,
        max_window=args.max_window,
    )
    opponent_white = _make_opponent(args.opponent, args.seed + 23)
    runner_1 = build_match_runner("eve", with_ui=False, black_source=candidate_black, white_source=opponent_white)
    s1 = run_selfplay(
        games=args.games,
        runner=runner_1,
        initial_game_mode=args.game_mode,
        progress_callback=lambda done, total, p1, p2, d, steps: _render_progress(
            "eval-v3-r1", done, total, extra=f"w/d/l={p1}/{d}/{p2}"
        ),
    )
    sys.stdout.write("\n")

    opponent_black = _make_opponent(args.opponent, args.seed + 37)
    candidate_white = _build_model_source(
        paths,
        args.device,
        sample_top_k=args.sample_top_k,
        sample_temperature=args.sample_temperature,
        decision_top_k=args.decision_top_k,
        decision_alpha=args.decision_alpha,
        decision_beta=args.decision_beta,
        max_window=args.max_window,
    )
    runner_2 = build_match_runner("eve", with_ui=False, black_source=opponent_black, white_source=candidate_white)
    s2 = run_selfplay(
        games=args.games,
        runner=runner_2,
        initial_game_mode=args.game_mode,
        progress_callback=lambda done, total, p1, p2, d, steps: _render_progress(
            "eval-v3-r2", done, total, extra=f"w/d/l={p1}/{d}/{p2}"
        ),
    )
    sys.stdout.write("\n")

    if args.verbose:
        _print_round(f"model_v3(black) vs {args.opponent}(white)", s1)
        _print_round(f"{args.opponent}(black) vs model_v3(white)", s2)

    candidate_wins = s1.p1_wins + s2.p2_wins
    opponent_wins = s1.p2_wins + s2.p1_wins
    draws = s1.draws + s2.draws
    total_games = args.games * 2
    total_steps = s1.steps + s2.steps

    print("--- V3 Model Eval Summary ---")
    print(f"mode={args.game_mode} device={args.device} opponent={args.opponent}")
    print(
        f"sample_top_k={args.sample_top_k} sample_temperature={args.sample_temperature:.3f} "
        f"decision_top_k={args.decision_top_k} decision_alpha={args.decision_alpha:.3f} "
        f"decision_beta={args.decision_beta:.3f} max_window={args.max_window}"
    )
    print(
        " ".join(
            [
                f"model_black_fast={paths['black_fast'] or '(none)'}",
                f"model_black_slow={paths['black_slow'] or '(none)'}",
                f"model_white_fast={paths['white_fast'] or '(none)'}",
                f"model_white_slow={paths['white_slow'] or '(none)'}",
            ]
        )
    )
    print(f"games={total_games} model_wins={candidate_wins} opponent_wins={opponent_wins} draws={draws}")
    print(f"model_win_rate={candidate_wins / total_games:.3f} avg_steps={total_steps / total_games:.2f}")


if __name__ == "__main__":
    main()
