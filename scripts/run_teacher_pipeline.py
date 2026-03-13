"""Run the default slow teacher-data pipeline sequentially."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import os


def run_step(repo_root: Path, title: str, args: list[str]) -> None:
    print()
    print(f"=== {title} ===")
    cmd = [sys.executable, "-u", *args]
    print(" ".join(cmd))
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


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    trace_raw = "artifacts/datasets/teacher_trace_raw.jsonl"
    summary_raw = "artifacts/datasets/teacher_games_summary.jsonl"
    trace_balanced = "artifacts/datasets/teacher_trace_balanced.jsonl"
    summary_balanced = "artifacts/datasets/teacher_games_balanced.jsonl"
    model_slow = "artifacts/models/model_tiny_policy_slow.pt"

    run_step(
        repo_root,
        "1/5 Export Raw Teacher Data (slow)",
        [
            "scripts/export_teacher_data.py",
            "--games",
            "2000",
            "--workers",
            "8",
            "--trace-output",
            trace_raw,
            "--summary-output",
            summary_raw,
        ],
    )
    run_step(
        repo_root,
        "2/5 Curate Balanced Dataset",
        [
            "scripts/curate_teacher_dataset.py",
            "--trace-input",
            trace_raw,
            "--summary-input",
            summary_raw,
            "--trace-output",
            trace_balanced,
            "--summary-output",
            summary_balanced,
            "--split-step",
            "40",
        ],
    )
    run_step(
        repo_root,
        "3/5 Train Slow Model",
        [
            "scripts/train_policy_model.py",
            "--trace",
            trace_balanced,
            "--mode",
            "slow",
            "--output",
            model_slow,
            "--epochs",
            "12",
            "--batch-size",
            "256",
            "--device",
            "cuda",
        ],
    )
    run_step(
        repo_root,
        "4/5 Eval vs Weak (slow)",
        [
            "scripts/eval_model_ai.py",
            "--model",
            model_slow,
            "--opponent",
            "weak",
            "--games",
            "50",
            "--game-mode",
            "slow",
            "--device",
            "cuda",
        ],
    )
    run_step(
        repo_root,
        "5/5 Eval vs Baseline (slow)",
        [
            "scripts/eval_model_ai.py",
            "--model",
            model_slow,
            "--opponent",
            "baseline",
            "--games",
            "50",
            "--game-mode",
            "slow",
            "--device",
            "cuda",
        ],
    )

    print()
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
