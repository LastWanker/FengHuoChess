"""Train TinyPolicyCNN with policy+value objectives from trace JSONL."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, random_split
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "train_policy_model.py requires torch. Please install torch in this project's .venv first."
    ) from exc

from src.fenghuo_chess.ai.model_ai import TinyPolicyCNN
from src.fenghuo_chess.constants.gameplay import BOARD_SIZE


@dataclass
class Sample:
    x: np.ndarray
    y: int
    weight: float
    value: float
    is_counterexample: bool


@dataclass
class LoadTraceStats:
    total_rows: int = 0
    kept_rows: int = 0
    skipped_by_mode: int = 0
    skipped_by_player: int = 0
    counterexample_rows: int = 0


class TraceDataset(Dataset):
    def __init__(self, samples: list[Sample]) -> None:
        self._x = torch.tensor(np.stack([s.x for s in samples]), dtype=torch.float32)
        self._y = torch.tensor([s.y for s in samples], dtype=torch.long)
        self._w = torch.tensor([s.weight for s in samples], dtype=torch.float32)
        self._v = torch.tensor([s.value for s in samples], dtype=torch.float32)
        self._neg = torch.tensor([1 if s.is_counterexample else 0 for s in samples], dtype=torch.bool)

    def __len__(self) -> int:
        return int(self._y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._x[idx], self._y[idx], self._w[idx], self._v[idx], self._neg[idx]


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


def encode_state_dict(state: dict) -> np.ndarray:
    board = np.array(state["board"], dtype=np.int64)
    channels = np.zeros((9, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    channels[0] = (board == 1).astype(np.float32)
    channels[1] = (board == 2).astype(np.float32)
    channels[2].fill(1.0 if int(state.get("current_player", 1)) == 1 else 0.0)

    stage = int(state.get("stage", 1))
    stage_idx = max(1, min(4, stage)) - 1
    channels[3 + stage_idx].fill(1.0)

    game_mode = str(state.get("game_mode", "slow"))
    channels[7].fill(1.0 if game_mode == "fast" else 0.0)

    legal_mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for row, col in state.get("stage_positions", []):
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and board[row, col] == 0:
            legal_mask[row, col] = 1.0
    channels[8] = legal_mask
    return channels


def outcome_weight(outcome: int) -> float:
    if outcome > 0:
        return 1.0
    if outcome == 0:
        return 0.7
    return 0.4


def _parse_counterexample_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _policy_per_item_with_counterexamples(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    is_counterexample: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    pos_per_item = criterion(logits, labels)
    if not bool(torch.any(is_counterexample)):
        return pos_per_item
    probs = torch.softmax(logits, dim=1)
    target_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    neg_per_item = -torch.log(torch.clamp(1.0 - target_probs, min=1e-6))
    return torch.where(is_counterexample, neg_per_item, pos_per_item)


def load_trace_samples(
    path: Path,
    max_samples: int = 0,
    mode_filter: str = "auto",
    player_filter: str = "all",
) -> tuple[list[Sample], LoadTraceStats]:
    samples: list[Sample] = []
    stats = LoadTraceStats()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            stats.total_rows += 1
            row = json.loads(raw)
            state = row["state"]
            state_mode = str(state.get("game_mode", "slow"))
            if mode_filter != "auto" and state_mode != mode_filter:
                stats.skipped_by_mode += 1
                continue
            player = int(row.get("player", 0))
            if player_filter == "black" and player != 1:
                stats.skipped_by_player += 1
                continue
            if player_filter == "white" and player != 2:
                stats.skipped_by_player += 1
                continue
            action = row["action"]
            if len(action) != 2:
                continue
            r, c = int(action[0]), int(action[1])
            if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
                continue
            label = r * BOARD_SIZE + c
            x = encode_state_dict(state)
            is_counterexample = _parse_counterexample_flag(row.get("counterexample", False))
            outcome = int(row.get("outcome", 0))
            base_weight = outcome_weight(outcome)
            sample_weight = row.get("sample_weight", 1.0)
            try:
                sample_weight_value = float(sample_weight)
            except Exception:
                sample_weight_value = 1.0
            if sample_weight_value <= 0.0:
                sample_weight_value = 1.0
            w = base_weight * sample_weight_value
            v = float(max(-1, min(1, outcome)))
            samples.append(Sample(x=x, y=label, weight=w, value=v, is_counterexample=is_counterexample))
            stats.kept_rows += 1
            if is_counterexample:
                stats.counterexample_rows += 1
            if max_samples > 0 and len(samples) >= max_samples:
                break
    return samples, stats


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    policy_criterion: nn.Module,
    value_criterion: nn.Module,
    value_loss_weight: float,
) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_correct = 0
    total_items = 0
    total_pos = 0
    with torch.no_grad():
        for x, y, w, v, is_neg in loader:
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)
            v = v.to(device)
            is_neg = is_neg.to(device)

            policy_logits, value_pred = model.forward_with_value(x)
            logits = _mask_logits_with_legal(policy_logits.flatten(start_dim=1), x)
            policy_per_item = _policy_per_item_with_counterexamples(
                logits=logits,
                labels=y,
                is_counterexample=is_neg,
                criterion=policy_criterion,
            )
            value_per_item = value_criterion(value_pred, v)
            policy_loss = (policy_per_item * w).mean()
            value_loss = (value_per_item * w).mean()
            loss = policy_loss + float(value_loss_weight) * value_loss
            total_loss += float(loss.item()) * int(y.shape[0])
            total_policy_loss += float(policy_loss.item()) * int(y.shape[0])
            total_value_loss += float(value_loss.item()) * int(y.shape[0])
            total_items += int(y.shape[0])
            pred = torch.argmax(logits, dim=1)
            pos_mask = ~is_neg
            if bool(torch.any(pos_mask)):
                total_correct += int(((pred == y) & pos_mask).sum().item())
                total_pos += int(pos_mask.sum().item())
    if total_items == 0:
        return 0.0, 0.0, 0.0, 0.0
    acc = (total_correct / total_pos) if total_pos > 0 else 0.0
    return total_loss / total_items, total_policy_loss / total_items, total_value_loss / total_items, acc


def _load_init_weights(model: nn.Module, init_path: Path, device: torch.device) -> None:
    try:
        payload = torch.load(init_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(init_path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(payload)!r}")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported state_dict format: {type(state_dict)!r}")
    # Allow warm-start from legacy policy-only checkpoints (missing value_head).
    model.load_state_dict(state_dict, strict=False)


def _mask_logits_with_legal(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply legal-action masking from channel-8 before CE loss."""
    legal_mask = x[:, 8, :, :].flatten(start_dim=1) > 0.5
    if not bool(torch.any(legal_mask)):
        return logits
    masked = logits.masked_fill(~legal_mask, -1e9)
    return masked


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyPolicyCNN with policy+value objectives")
    parser.add_argument("--trace", type=str, default="artifacts/datasets/teacher_trace.jsonl", help="Input trace JSONL")
    parser.add_argument("--output", type=str, default="artifacts/models/model_tiny_policy.pt", help="Output model path")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--value-loss-weight", type=float, default=0.25, help="Weight for value MSE loss term")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means no limit")
    parser.add_argument("--init-model", type=str, default="", help="Optional init checkpoint for warm-start training")
    parser.add_argument("--mode", choices=["auto", "fast", "slow"], default="auto", help="Filter samples by game_mode")
    parser.add_argument("--player-filter", choices=["all", "black", "white"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.exists():
        raise FileNotFoundError(f"trace not found: {trace_path}")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.value_loss_weight < 0.0:
        raise ValueError("--value-loss-weight must be >= 0")
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0, 1)")
    init_model = args.init_model.strip()
    if init_model and not Path(init_model).exists():
        raise FileNotFoundError(f"init model not found: {init_model}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples, load_stats = load_trace_samples(
        trace_path,
        max_samples=max(0, args.max_samples),
        mode_filter=args.mode,
        player_filter=args.player_filter,
    )
    if not samples:
        raise RuntimeError("No training samples loaded from trace.")

    dataset = TraceDataset(samples)
    val_count = int(len(dataset) * args.val_ratio)
    train_count = len(dataset) - val_count
    generator = torch.Generator().manual_seed(args.seed)
    if val_count > 0:
        train_set, val_set = random_split(dataset, [train_count, val_count], generator=generator)
    else:
        train_set = dataset
        val_set = None

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False) if val_set else None

    requested = args.device.strip().lower()
    if requested == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested)

    model = TinyPolicyCNN().to(device)
    if init_model:
        _load_init_weights(model, Path(init_model), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    policy_criterion = nn.CrossEntropyLoss(reduction="none")
    value_criterion = nn.MSELoss(reduction="none")

    print("--- Train TinyPolicyCNN (policy+value) ---")
    print(
        f"mode_filter={args.mode} rows_total={load_stats.total_rows} "
        f"rows_used={load_stats.kept_rows} rows_skipped_by_mode={load_stats.skipped_by_mode} "
        f"rows_skipped_by_player={load_stats.skipped_by_player} player_filter={args.player_filter} "
        f"counterexample_rows={load_stats.counterexample_rows}"
    )
    print(
        f"samples={len(dataset)} train={train_count} val={val_count} device={device} "
        f"value_loss_weight={args.value_loss_weight:.3f}"
    )
    if init_model:
        print(f"init_model={init_model}")

    train_steps_per_epoch = max(1, len(train_loader))
    total_train_steps = args.epochs * train_steps_per_epoch
    global_step = 0
    last_train_total_loss = 0.0
    last_train_policy_loss = 0.0
    last_train_value_loss = 0.0
    last_train_acc = 0.0
    last_val_total_loss = 0.0
    last_val_policy_loss = 0.0
    last_val_value_loss = 0.0
    last_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_correct = 0
        total_items = 0
        total_pos = 0

        for batch_idx, (x, y, w, v, is_neg) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)
            v = v.to(device)
            is_neg = is_neg.to(device)

            optimizer.zero_grad()
            policy_logits, value_pred = model.forward_with_value(x)
            logits = _mask_logits_with_legal(policy_logits.flatten(start_dim=1), x)
            policy_per_item = _policy_per_item_with_counterexamples(
                logits=logits,
                labels=y,
                is_counterexample=is_neg,
                criterion=policy_criterion,
            )
            value_per_item = value_criterion(value_pred, v)
            policy_loss = (policy_per_item * w).mean()
            value_loss = (value_per_item * w).mean()
            loss = policy_loss + float(args.value_loss_weight) * value_loss
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * int(y.shape[0])
            total_policy_loss += float(policy_loss.item()) * int(y.shape[0])
            total_value_loss += float(value_loss.item()) * int(y.shape[0])
            total_items += int(y.shape[0])
            pred = torch.argmax(logits, dim=1)
            pos_mask = ~is_neg
            if bool(torch.any(pos_mask)):
                total_correct += int(((pred == y) & pos_mask).sum().item())
                total_pos += int(pos_mask.sum().item())
            global_step += 1
            _render_progress(
                "train",
                global_step,
                total_train_steps,
                extra=(
                    f"epoch={epoch}/{args.epochs} batch={batch_idx}/{train_steps_per_epoch} "
                    f"loss={total_loss / max(1, total_items):.4f} "
                    f"p={total_policy_loss / max(1, total_items):.4f} "
                    f"v={total_value_loss / max(1, total_items):.4f} "
                    f"acc={total_correct / max(1, total_pos):.4f}"
                ),
            )

        train_loss = total_loss / max(1, total_items)
        train_policy_loss = total_policy_loss / max(1, total_items)
        train_value_loss = total_value_loss / max(1, total_items)
        train_acc = total_correct / max(1, total_pos)
        last_train_total_loss = train_loss
        last_train_policy_loss = train_policy_loss
        last_train_value_loss = train_value_loss
        last_train_acc = train_acc
        if val_loader is not None:
            val_loss, val_policy_loss, val_value_loss, val_acc = evaluate(
                model,
                val_loader,
                device,
                policy_criterion,
                value_criterion,
                args.value_loss_weight,
            )
            last_val_total_loss = val_loss
            last_val_policy_loss = val_policy_loss
            last_val_value_loss = val_value_loss
            last_val_acc = val_acc
            _render_progress(
                "train",
                global_step,
                total_train_steps,
                extra=(
                    f"epoch={epoch}/{args.epochs} done "
                    f"train_loss={train_loss:.4f} (p={train_policy_loss:.4f}, v={train_value_loss:.4f}) "
                    f"train_acc={train_acc:.4f} "
                    f"val_loss={val_loss:.4f} (p={val_policy_loss:.4f}, v={val_value_loss:.4f}) "
                    f"val_acc={val_acc:.4f}"
                ),
            )
    sys.stdout.write("\n")
    if val_loader is not None:
        print(
            f"final train_loss={last_train_total_loss:.4f} "
            f"(p={last_train_policy_loss:.4f}, v={last_train_value_loss:.4f}) "
            f"train_acc={last_train_acc:.4f} "
            f"val_loss={last_val_total_loss:.4f} "
            f"(p={last_val_policy_loss:.4f}, v={last_val_value_loss:.4f}) "
            f"val_acc={last_val_acc:.4f}"
        )
    else:
        print(
            f"final train_loss={last_train_total_loss:.4f} "
            f"(p={last_train_policy_loss:.4f}, v={last_train_value_loss:.4f}) "
            f"train_acc={last_train_acc:.4f}"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "meta": {
                "model": "TinyPolicyCNN(policy+value)",
                "board_size": BOARD_SIZE,
                "channels": 9,
                "samples": len(dataset),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "value_loss_weight": args.value_loss_weight,
                "mode_filter": args.mode,
                "trace": str(trace_path),
            },
        },
        out_path,
    )
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
