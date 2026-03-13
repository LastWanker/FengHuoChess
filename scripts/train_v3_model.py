"""Train V3 temporal model (CNN + Transformer) with policy+value objectives."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "train_v3_model.py requires torch. Please install torch in this project's .venv first."
    ) from exc

from src.fenghuo_chess.ai.model_v3 import V3TemporalPolicyValueNet, encode_state_dict_v3
from src.fenghuo_chess.constants.gameplay import BOARD_SIZE


@dataclass
class TraceStep:
    game_id: int
    step_index: int
    x: np.ndarray
    y: int
    weight: float
    value: float
    outcome: int
    is_counterexample: bool


@dataclass
class StepRef:
    game_id: int
    step_index: int
    sampling_weight: float
    cross_9_10: bool
    cross_25_26: bool
    critical_collapse: bool


@dataclass
class LoadTraceStats:
    total_rows: int = 0
    kept_rows: int = 0
    skipped_by_mode: int = 0
    skipped_by_player: int = 0
    bad_action_rows: int = 0
    counterexample_rows: int = 0


class TemporalTraceDataset(Dataset):
    def __init__(
        self,
        games: dict[int, list[TraceStep]],
        refs: list[StepRef],
        window: int,
    ) -> None:
        self._games = games
        self._refs = refs
        self._window = int(window)

    def __len__(self) -> int:
        return len(self._refs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ref = self._refs[idx]
        steps = self._games[ref.game_id]
        end = int(ref.step_index)
        start = max(0, end - self._window + 1)
        seq = steps[start : end + 1]

        x = np.zeros((self._window, 9, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        offset = self._window - len(seq)
        for i, step in enumerate(seq):
            x[offset + i] = step.x

        target = steps[end]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(target.y, dtype=torch.long),
            torch.tensor(target.weight, dtype=torch.float32),
            torch.tensor(target.value, dtype=torch.float32),
            torch.tensor(bool(target.is_counterexample), dtype=torch.bool),
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


def _outcome_weight(outcome: int) -> float:
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


def _load_trace_games(
    path: Path,
    *,
    mode_filter: str,
    player_filter: str,
    max_samples: int,
) -> tuple[dict[int, list[TraceStep]], LoadTraceStats]:
    games: dict[int, list[TraceStep]] = defaultdict(list)
    stats = LoadTraceStats()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            stats.total_rows += 1
            row = json.loads(raw)

            state = row.get("state", {})
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

            action = row.get("action", [])
            if not isinstance(action, list) or len(action) != 2:
                stats.bad_action_rows += 1
                continue
            r, c = int(action[0]), int(action[1])
            if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
                stats.bad_action_rows += 1
                continue

            game_id = int(row.get("game_id", -1))
            step_index = int(row.get("step_index", 0))
            if game_id < 0:
                stats.bad_action_rows += 1
                continue

            x = encode_state_dict_v3(state)
            label = r * BOARD_SIZE + c
            outcome = int(row.get("outcome", 0))
            is_counterexample = _parse_counterexample_flag(row.get("counterexample", False))
            if is_counterexample:
                stats.counterexample_rows += 1

            sample_weight = row.get("sample_weight", 1.0)
            try:
                sample_weight_value = float(sample_weight)
            except Exception:
                sample_weight_value = 1.0
            if sample_weight_value <= 0.0:
                sample_weight_value = 1.0

            total_weight = _outcome_weight(outcome) * sample_weight_value
            games[game_id].append(
                TraceStep(
                    game_id=game_id,
                    step_index=step_index,
                    x=x,
                    y=label,
                    weight=float(total_weight),
                    value=float(max(-1, min(1, outcome))),
                    outcome=int(outcome),
                    is_counterexample=bool(is_counterexample),
                )
            )
            stats.kept_rows += 1
            if max_samples > 0 and stats.kept_rows >= max_samples:
                break

    cleaned: dict[int, list[TraceStep]] = {}
    for game_id, rows in games.items():
        sorted_rows = sorted(rows, key=lambda item: item.step_index)
        for new_idx, row in enumerate(sorted_rows):
            row.step_index = int(new_idx)
        cleaned[game_id] = sorted_rows
    return cleaned, stats


def _window_cross_boundary(step_index: int, window: int, boundary: int) -> bool:
    move_no = int(step_index) + 1
    start_move = max(1, move_no - int(window) + 1)
    return start_move <= boundary < move_no


def _is_critical_collapse(
    *,
    step: TraceStep,
    game_total_steps: int,
    boundary_step: int,
    horizon: int,
) -> bool:
    move_no = int(step.step_index) + 1
    if move_no > int(boundary_step):
        return False
    if int(step.outcome) >= 0:
        return False
    remaining = int(game_total_steps) - move_no
    return remaining <= int(horizon)


def _build_refs(
    games: dict[int, list[TraceStep]],
    *,
    window: int,
    w_cross_9_10: float,
    w_cross_25_26: float,
    w_critical: float,
    critical_boundary_step: int,
    critical_horizon: int,
) -> list[StepRef]:
    refs: list[StepRef] = []
    for game_id, rows in games.items():
        total_steps = len(rows)
        for step in rows:
            cross_9_10 = _window_cross_boundary(step.step_index, window, boundary=9)
            cross_25_26 = _window_cross_boundary(step.step_index, window, boundary=25)
            critical = _is_critical_collapse(
                step=step,
                game_total_steps=total_steps,
                boundary_step=critical_boundary_step,
                horizon=critical_horizon,
            )

            sample_w = 1.0
            if cross_9_10:
                sample_w *= float(w_cross_9_10)
            if cross_25_26:
                sample_w *= float(w_cross_25_26)
            if critical:
                sample_w *= float(w_critical)
            refs.append(
                StepRef(
                    game_id=int(game_id),
                    step_index=int(step.step_index),
                    sampling_weight=float(max(1e-6, sample_w)),
                    cross_9_10=bool(cross_9_10),
                    cross_25_26=bool(cross_25_26),
                    critical_collapse=bool(critical),
                )
            )
    return refs


def _mask_logits_with_legal(logits: torch.Tensor, x_seq: torch.Tensor) -> torch.Tensor:
    legal_mask = x_seq[:, -1, 8, :, :].flatten(start_dim=1) > 0.5
    if not bool(torch.any(legal_mask)):
        return logits
    return logits.masked_fill(~legal_mask, -1e9)


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


def _load_init_weights(model: nn.Module, init_path: Path, device: torch.device) -> tuple[int, int]:
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

    own = model.state_dict()
    merged = dict(own)
    loaded = 0
    skipped = 0
    for key, value in state_dict.items():
        if key not in own:
            skipped += 1
            continue
        if own[key].shape != value.shape:
            skipped += 1
            continue
        merged[key] = value
        loaded += 1
    model.load_state_dict(merged, strict=False)
    return loaded, skipped


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    value_loss_weight: float,
    policy_criterion: nn.Module,
    value_criterion: nn.Module,
) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_correct = 0
    total_items = 0
    total_pos = 0
    with torch.no_grad():
        for x_seq, y, w, v, is_neg in loader:
            x_seq = x_seq.to(device)
            y = y.to(device)
            w = w.to(device)
            v = v.to(device)
            is_neg = is_neg.to(device)

            policy_logits, value_pred = model.forward_with_value(x_seq)
            logits = _mask_logits_with_legal(policy_logits.flatten(start_dim=1), x_seq)
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

            batch = int(y.shape[0])
            total_items += batch
            total_loss += float(loss.item()) * batch
            total_policy_loss += float(policy_loss.item()) * batch
            total_value_loss += float(value_loss.item()) * batch
            pred = torch.argmax(logits, dim=1)
            pos_mask = ~is_neg
            if bool(torch.any(pos_mask)):
                total_correct += int(((pred == y) & pos_mask).sum().item())
                total_pos += int(pos_mask.sum().item())

    if total_items == 0:
        return 0.0, 0.0, 0.0, 0.0
    acc = (total_correct / total_pos) if total_pos > 0 else 0.0
    return (
        total_loss / total_items,
        total_policy_loss / total_items,
        total_value_loss / total_items,
        acc,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V3 temporal policy-value model")
    parser.add_argument(
        "--trace",
        type=str,
        default="artifacts/v3_transformer/datasets/model_pool_balanced_trace.jsonl",
        help="Input trace JSONL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/v3_transformer/models/model_v3_candidate.pt",
        help="Output model path",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means no limit")
    parser.add_argument("--init-model", type=str, default="", help="Optional init checkpoint")
    parser.add_argument("--mode", choices=["auto", "fast", "slow"], default="auto")
    parser.add_argument("--player-filter", choices=["all", "black", "white"], default="all")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--window", type=int, default=16, help="Temporal window length")
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
    args = parser.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.exists():
        raise FileNotFoundError(f"trace not found: {trace_path}")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.window < 2:
        raise ValueError("--window must be >= 2")
    if args.value_loss_weight < 0.0:
        raise ValueError("--value-loss-weight must be >= 0")
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0,1)")
    init_model = args.init_model.strip()
    if init_model and not Path(init_model).exists():
        raise FileNotFoundError(f"init model not found: {init_model}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    games, load_stats = _load_trace_games(
        trace_path,
        mode_filter=args.mode,
        player_filter=args.player_filter,
        max_samples=max(0, args.max_samples),
    )
    if not games:
        raise RuntimeError("No samples loaded from trace.")

    game_ids = sorted(games.keys())
    rng = random.Random(args.seed)
    rng.shuffle(game_ids)
    val_game_count = int(len(game_ids) * args.val_ratio)
    train_ids = game_ids[val_game_count:]
    val_ids = game_ids[:val_game_count]
    if not train_ids:
        raise RuntimeError("No training games after split. Reduce --val-ratio.")

    train_games = {gid: games[gid] for gid in train_ids}
    val_games = {gid: games[gid] for gid in val_ids} if val_ids else {}

    train_refs = _build_refs(
        train_games,
        window=args.window,
        w_cross_9_10=args.sampling_boundary_9_10_weight,
        w_cross_25_26=args.sampling_boundary_25_26_weight,
        w_critical=args.sampling_critical_weight,
        critical_boundary_step=args.critical_boundary_step,
        critical_horizon=args.critical_horizon,
    )
    if not train_refs:
        raise RuntimeError("No train refs produced.")
    val_refs = _build_refs(
        val_games,
        window=args.window,
        w_cross_9_10=args.sampling_boundary_9_10_weight,
        w_cross_25_26=args.sampling_boundary_25_26_weight,
        w_critical=args.sampling_critical_weight,
        critical_boundary_step=args.critical_boundary_step,
        critical_horizon=args.critical_horizon,
    ) if val_games else []

    train_dataset = TemporalTraceDataset(train_games, train_refs, window=args.window)
    val_dataset = TemporalTraceDataset(val_games, val_refs, window=args.window) if val_refs else None
    train_weights = torch.tensor([float(ref.sampling_weight) for ref in train_refs], dtype=torch.float32)
    train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_refs), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, drop_last=False)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        if val_dataset is not None
        else None
    )

    requested = args.device.strip().lower()
    if requested == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested)

    model = V3TemporalPolicyValueNet(
        hidden_channels=args.hidden_channels,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        max_window=args.window,
        dropout=args.dropout,
    ).to(device)
    loaded_keys = 0
    skipped_keys = 0
    if init_model:
        loaded_keys, skipped_keys = _load_init_weights(model, Path(init_model), device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    policy_criterion = nn.CrossEntropyLoss(reduction="none")
    value_criterion = nn.MSELoss(reduction="none")

    cross_9_10_count = sum(1 for ref in train_refs if ref.cross_9_10)
    cross_25_26_count = sum(1 for ref in train_refs if ref.cross_25_26)
    critical_count = sum(1 for ref in train_refs if ref.critical_collapse)
    print("--- Train V3 Temporal Model (policy+value) ---")
    print(
        f"rows_total={load_stats.total_rows} rows_used={load_stats.kept_rows} "
        f"rows_skipped_by_mode={load_stats.skipped_by_mode} rows_skipped_by_player={load_stats.skipped_by_player} "
        f"counterexample_rows={load_stats.counterexample_rows} bad_action_rows={load_stats.bad_action_rows}"
    )
    print(
        f"games_total={len(games)} train_games={len(train_ids)} val_games={len(val_ids)} "
        f"train_refs={len(train_refs)} val_refs={len(val_refs)} window={args.window} "
        f"cross_9_10={cross_9_10_count} cross_25_26={cross_25_26_count} critical={critical_count}"
    )
    print(
        f"sampling_weights=(9_10={args.sampling_boundary_9_10_weight:.2f}, "
        f"25_26={args.sampling_boundary_25_26_weight:.2f}, critical={args.sampling_critical_weight:.2f})"
    )
    print(
        f"device={device} value_loss_weight={args.value_loss_weight:.3f} "
        f"model_cfg=(hidden={args.hidden_channels}, d_model={args.d_model}, heads={args.n_heads}, "
        f"layers={args.n_layers}, ffn={args.ffn_dim}, dropout={args.dropout:.2f})"
    )
    if init_model:
        print(f"init_model={init_model} loaded_keys={loaded_keys} skipped_keys={skipped_keys}")

    train_steps_per_epoch = max(1, len(train_loader))
    total_steps = train_steps_per_epoch * args.epochs
    global_step = 0
    last_train = (0.0, 0.0, 0.0, 0.0)
    last_val = (0.0, 0.0, 0.0, 0.0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_items = 0
        total_correct = 0
        total_pos = 0

        for batch_idx, (x_seq, y, w, v, is_neg) in enumerate(train_loader, start=1):
            x_seq = x_seq.to(device)
            y = y.to(device)
            w = w.to(device)
            v = v.to(device)
            is_neg = is_neg.to(device)

            optimizer.zero_grad()
            policy_logits, value_pred = model.forward_with_value(x_seq)
            logits = _mask_logits_with_legal(policy_logits.flatten(start_dim=1), x_seq)
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

            batch = int(y.shape[0])
            total_items += batch
            total_loss += float(loss.item()) * batch
            total_policy += float(policy_loss.item()) * batch
            total_value += float(value_loss.item()) * batch
            pred = torch.argmax(logits, dim=1)
            pos_mask = ~is_neg
            if bool(torch.any(pos_mask)):
                total_correct += int(((pred == y) & pos_mask).sum().item())
                total_pos += int(pos_mask.sum().item())

            global_step += 1
            _render_progress(
                "train-v3",
                global_step,
                total_steps,
                extra=(
                    f"epoch={epoch}/{args.epochs} batch={batch_idx}/{train_steps_per_epoch} "
                    f"loss={total_loss / max(1, total_items):.4f} "
                    f"p={total_policy / max(1, total_items):.4f} "
                    f"v={total_value / max(1, total_items):.4f} "
                    f"acc={total_correct / max(1, total_pos):.4f}"
                ),
            )

        train_metrics = (
            total_loss / max(1, total_items),
            total_policy / max(1, total_items),
            total_value / max(1, total_items),
            total_correct / max(1, total_pos),
        )
        last_train = train_metrics
        if val_loader is not None:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                value_loss_weight=float(args.value_loss_weight),
                policy_criterion=policy_criterion,
                value_criterion=value_criterion,
            )
            last_val = val_metrics
            _render_progress(
                "train-v3",
                global_step,
                total_steps,
                extra=(
                    f"epoch={epoch}/{args.epochs} done train={train_metrics[0]:.4f} "
                    f"(p={train_metrics[1]:.4f},v={train_metrics[2]:.4f},acc={train_metrics[3]:.4f}) "
                    f"val={val_metrics[0]:.4f} (p={val_metrics[1]:.4f},v={val_metrics[2]:.4f},acc={val_metrics[3]:.4f})"
                ),
            )

    sys.stdout.write("\n")
    if val_loader is not None:
        print(
            f"final train_loss={last_train[0]:.4f} (p={last_train[1]:.4f}, v={last_train[2]:.4f}) "
            f"train_acc={last_train[3]:.4f} "
            f"val_loss={last_val[0]:.4f} (p={last_val[1]:.4f}, v={last_val[2]:.4f}) val_acc={last_val[3]:.4f}"
        )
    else:
        print(
            f"final train_loss={last_train[0]:.4f} (p={last_train[1]:.4f}, v={last_train[2]:.4f}) "
            f"train_acc={last_train[3]:.4f}"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "meta": {
                "version": "v3",
                "model": "V3TemporalPolicyValueNet",
                "arch": "cnn_temporal_transformer",
                "board_size": BOARD_SIZE,
                "channels": 9,
                "window": args.window,
                "model_cfg": dict(model.model_cfg),
                "samples": len(train_refs) + len(val_refs),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "value_loss_weight": args.value_loss_weight,
                "mode_filter": args.mode,
                "player_filter": args.player_filter,
                "trace": str(trace_path),
                "sampling": {
                    "w_cross_9_10": args.sampling_boundary_9_10_weight,
                    "w_cross_25_26": args.sampling_boundary_25_26_weight,
                    "w_critical": args.sampling_critical_weight,
                    "critical_boundary_step": args.critical_boundary_step,
                    "critical_horizon": args.critical_horizon,
                },
            },
        },
        out_path,
    )
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()

