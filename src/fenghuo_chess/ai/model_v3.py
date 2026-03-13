"""V3 model AI: CNN + temporal transformer with value-aware action selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import random
import warnings

import numpy as np

from src.fenghuo_chess.application.logic_api import Action, legal_actions, simulate_action
from src.fenghuo_chess.constants.gameplay import BOARD_SIZE
from src.fenghuo_chess.domain.models import GameState
from src.fenghuo_chess.services.exposure_service import ExposureService

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - optional runtime dependency
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


BOARD_CELLS = BOARD_SIZE * BOARD_SIZE

# Silence a noisy PyTorch performance warning from TransformerEncoder(norm_first=True).
# This does not affect model behavior; it only suppresses repeated console spam.
warnings.filterwarnings(
    "ignore",
    message=(
        "enable_nested_tensor is True, but self.use_nested_tensor is False "
        "because encoder_layer.norm_first was True"
    ),
    category=UserWarning,
    module=r"torch\.nn\.modules\.transformer",
)


def encode_state_dict_v3(state: dict) -> np.ndarray:
    board = np.array(state["board"], dtype=np.int64)
    channels = np.zeros((9, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    channels[0] = (board == 1).astype(np.float32)
    channels[1] = (board == 2).astype(np.float32)
    channels[2].fill(1.0 if int(state.get("current_player", 1)) == 1 else 0.0)

    stage = int(state.get("stage", 1))
    stage_idx = max(1, min(4, stage)) - 1
    channels[3 + stage_idx].fill(1.0)
    channels[7].fill(1.0 if str(state.get("game_mode", "slow")) == "fast" else 0.0)

    legal_mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for row, col in state.get("stage_positions", []):
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and board[row, col] == 0:
            legal_mask[row, col] = 1.0
    channels[8] = legal_mask
    return channels


def encode_game_state_v3(state: GameState) -> np.ndarray:
    channels = np.zeros((9, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    channels[0] = (state.board == 1).astype(np.float32)
    channels[1] = (state.board == 2).astype(np.float32)
    channels[2].fill(1.0 if int(state.current_player) == 1 else 0.0)
    stage_idx = max(1, min(4, int(state.stage))) - 1
    channels[3 + stage_idx].fill(1.0)
    channels[7].fill(1.0 if state.game_mode == "fast" else 0.0)

    legal_mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for row, col in state.stage_positions:
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and state.board[row, col] == 0:
            legal_mask[row, col] = 1.0
    channels[8] = legal_mask
    return channels


class V3TemporalPolicyValueNet(nn.Module):
    """CNN spatial encoder + causal temporal transformer."""

    def __init__(
        self,
        *,
        in_channels: int = 9,
        hidden_channels: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        ffn_dim: int = 256,
        max_window: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_window = int(max(2, max_window))
        self.model_cfg = {
            "in_channels": int(in_channels),
            "hidden_channels": int(hidden_channels),
            "d_model": int(d_model),
            "n_heads": int(n_heads),
            "n_layers": int(n_layers),
            "ffn_dim": int(ffn_dim),
            "max_window": int(max_window),
            "dropout": float(dropout),
        }

        self.step_body = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.step_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.step_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_window, d_model))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        self.policy_head = nn.Linear(d_model, BOARD_CELLS)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        policy, _ = self.forward_with_value(x_seq)
        return policy

    def forward_with_value(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x_seq.dim() == 4:
            x_seq = x_seq.unsqueeze(1)
        if x_seq.dim() != 5:
            raise ValueError(f"Expected input dims [B,T,C,H,W] or [B,C,H,W], got shape={tuple(x_seq.shape)}")

        batch, steps, channels, height, width = x_seq.shape
        if steps > self.max_window:
            raise ValueError(f"steps={steps} exceeds max_window={self.max_window}")
        if channels != 9 or height != BOARD_SIZE or width != BOARD_SIZE:
            raise ValueError(
                f"Expected [*,*,9,{BOARD_SIZE},{BOARD_SIZE}], got shape={tuple(x_seq.shape)}"
            )

        flat = x_seq.reshape(batch * steps, channels, height, width)
        feat = self.step_body(flat)
        token = self.step_proj(self.step_pool(feat)).reshape(batch, steps, -1)
        token = token + self.pos_embedding[:, :steps, :]

        causal_mask = torch.triu(
            torch.full((steps, steps), float("-inf"), device=x_seq.device, dtype=token.dtype),
            diagonal=1,
        )
        encoded = self.temporal_encoder(token, mask=causal_mask)
        root = encoded[:, -1, :]
        policy = self.policy_head(root).view(batch, 1, BOARD_SIZE, BOARD_SIZE)
        value = self.value_head(root).squeeze(-1)
        return policy, value


@dataclass
class _LoadedModel:
    model: V3TemporalPolicyValueNet
    loaded: bool


class ModelAISourceV3:
    """V3 model source with value-aware top-k re-ranking."""

    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        *,
        model_fast_path: str | None = None,
        model_slow_path: str | None = None,
        model_black_path: str | None = None,
        model_white_path: str | None = None,
        model_black_fast_path: str | None = None,
        model_black_slow_path: str | None = None,
        model_white_fast_path: str | None = None,
        model_white_slow_path: str | None = None,
        rng: random.Random | None = None,
        sample_top_k: int = 0,
        sample_temperature: float = 1.0,
        decision_top_k: int = 5,
        decision_alpha: float = 1.0,
        decision_beta: float = 0.5,
        max_window: int = 16,
    ) -> None:
        if torch is None:
            raise RuntimeError(
                "ModelAISourceV3 requires torch, but importing torch failed."
            ) from _TORCH_IMPORT_ERROR

        self.device = self._resolve_device(device)
        self._rng = rng or random.Random()
        self._sample_top_k = max(0, int(sample_top_k))
        self._sample_temperature = max(1e-6, float(sample_temperature))
        self._decision_top_k = max(1, int(decision_top_k))
        self._decision_alpha = float(decision_alpha)
        self._decision_beta = float(decision_beta)
        self._max_window = max(2, int(max_window))
        self._exposure_service = ExposureService()
        # Temporal context must be shared across both players within one game.
        # Using per-color histories would drop every opponent move at inference time.
        self._histories: dict[str, list[np.ndarray]] = {}
        self._last_stones: dict[str, int] = {}

        common_fast = (model_fast_path or model_path or "").strip()
        common_slow = (model_slow_path or model_path or "").strip()
        black_base = (model_black_path or model_path or "").strip()
        white_base = (model_white_path or model_path or "").strip()
        self._paths: dict[str, dict[str, str]] = {
            "black": {
                "fast": (model_black_fast_path or black_base or common_fast or common_slow).strip(),
                "slow": (model_black_slow_path or black_base or common_slow or common_fast).strip(),
            },
            "white": {
                "fast": (model_white_fast_path or white_base or common_fast or common_slow).strip(),
                "slow": (model_white_slow_path or white_base or common_slow or common_fast).strip(),
            },
        }

        loaded: dict[str, dict[str, _LoadedModel | None]] = {
            "black": {
                "fast": self._load_model(self._paths["black"]["fast"]),
                "slow": self._load_model(self._paths["black"]["slow"]),
            },
            "white": {
                "fast": self._load_model(self._paths["white"]["fast"]),
                "slow": self._load_model(self._paths["white"]["slow"]),
            },
        }

        self._models: dict[str, dict[str, V3TemporalPolicyValueNet]] = {"black": {}, "white": {}}
        for color in ("black", "white"):
            fast = loaded[color]["fast"]
            slow = loaded[color]["slow"]
            if fast is None and slow is None:
                fresh = self._build_default_model()
                self._models[color]["fast"] = fresh
                self._models[color]["slow"] = self._clone_model(fresh)
            elif fast is None and slow is not None:
                self._models[color]["slow"] = slow.model
                self._models[color]["fast"] = self._clone_model(slow.model)
            elif slow is None and fast is not None:
                self._models[color]["fast"] = fast.model
                self._models[color]["slow"] = self._clone_model(fast.model)
            else:
                assert fast is not None and slow is not None
                self._models[color]["fast"] = fast.model
                self._models[color]["slow"] = slow.model

        for mode in ("fast", "slow"):
            black_model = self._models["black"][mode]
            white_model = self._models["white"][mode]
            black_loaded = loaded["black"][mode] is not None and bool(loaded["black"][mode].loaded)  # type: ignore[index]
            white_loaded = loaded["white"][mode] is not None and bool(loaded["white"][mode].loaded)  # type: ignore[index]
            if black_loaded and not white_loaded:
                self._models["white"][mode] = self._clone_model(black_model)
            if white_loaded and not black_loaded:
                self._models["black"][mode] = self._clone_model(white_model)

        for color_models in self._models.values():
            for model in color_models.values():
                model.to(self.device)
                model.eval()

    def select_action(self, state: GameState, legal_actions_list: Sequence[Action]) -> Action | None:
        if not legal_actions_list:
            return None
        legal_actions_seq = list(legal_actions_list)
        key = self._history_key(state)
        self._reset_history_if_needed(key, state)

        current_encoded = encode_game_state_v3(state)
        history = self._histories.get(key, [])
        seq_current = self._build_sequence(history + [current_encoded])
        model = self._model_for_context(state.game_mode, int(state.current_player))
        logits = self._predict_logits_from_sequence(model, seq_current)

        ranked = sorted(
            (
                (float(logits[action.row, action.col]), action)
                for action in legal_actions_seq
                if np.isfinite(float(logits[action.row, action.col]))
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        if not ranked:
            choice = self._rng.choice(legal_actions_seq)
            self._remember_state(key, current_encoded, state)
            return choice

        chosen = self._choose_action_with_value(state, ranked, history, current_encoded)
        if chosen not in legal_actions_seq:
            chosen = self._rng.choice(legal_actions_seq)

        self._remember_state(key, current_encoded, state)
        return chosen

    def _choose_action_with_value(
        self,
        state: GameState,
        ranked: list[tuple[float, Action]],
        history: list[np.ndarray],
        current_encoded: np.ndarray,
    ) -> Action:
        if self._decision_beta <= 0.0:
            sampled = self._sample_from_topk(ranked)
            return sampled or ranked[0][1]

        k = min(len(ranked), self._decision_top_k)
        if k <= 1:
            return ranked[0][1]

        legal_logits = np.array([score for score, _ in ranked[:k]], dtype=np.float64)
        shifted = legal_logits - float(np.max(legal_logits))
        exp = np.exp(shifted)
        policy_probs = exp / max(1e-12, float(np.sum(exp)))
        policy_log_probs = np.log(np.clip(policy_probs, 1e-9, 1.0))

        current_player = int(state.current_player)
        scored: list[tuple[float, Action]] = []
        for idx in range(k):
            action = ranked[idx][1]
            step = simulate_action(state, action, exposure_service=self._exposure_service)
            if not step.ok:
                continue
            next_state = step.state
            if next_state.game_over:
                if int(next_state.winner) == 0:
                    root_value = 0.0
                else:
                    root_value = 1.0 if int(next_state.winner) == current_player else -1.0
            else:
                next_encoded = encode_game_state_v3(next_state)
                seq_next = self._build_sequence(history + [current_encoded, next_encoded])
                next_model = self._model_for_context(next_state.game_mode, int(next_state.current_player))
                _, value_next = self._predict_policy_and_value_from_sequence(next_model, seq_next)
                root_value = -float(value_next)

            combined = self._decision_alpha * float(policy_log_probs[idx]) + self._decision_beta * float(root_value)
            scored.append((combined, action))

        if not scored:
            sampled = self._sample_from_topk(ranked[:k])
            return sampled or ranked[0][1]
        scored.sort(key=lambda item: item[0], reverse=True)
        sampled = self._sample_from_topk(scored)
        return sampled or scored[0][1]

    def _sample_from_topk(self, ranked: list[tuple[float, Action]]) -> Action | None:
        if self._sample_top_k <= 0:
            return None
        k = min(len(ranked), self._sample_top_k)
        if k <= 1:
            return ranked[0][1]
        logits = np.array([float(ranked[idx][0]) for idx in range(k)], dtype=np.float64)
        if not np.all(np.isfinite(logits)):
            return None
        scaled = (logits - float(np.max(logits))) / self._sample_temperature
        weights = np.exp(scaled)
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            return None
        point = self._rng.random() * total
        accum = 0.0
        for idx in range(k):
            accum += float(weights[idx])
            if point <= accum:
                return ranked[idx][1]
        return ranked[k - 1][1]

    def _history_key(self, state: GameState) -> str:
        mode = "fast" if str(state.game_mode).strip().lower() == "fast" else "slow"
        return mode

    def _reset_history_if_needed(self, key: str, state: GameState) -> None:
        stones = int(np.count_nonzero(state.board))
        prev = self._last_stones.get(key)
        if stones <= 1 or (prev is not None and stones < prev):
            self._histories[key] = []
        self._last_stones[key] = stones

    def _remember_state(self, key: str, encoded: np.ndarray, state: GameState) -> None:
        history = self._histories.setdefault(key, [])
        history.append(encoded)
        if len(history) > self._max_window:
            del history[:-self._max_window]
        self._last_stones[key] = int(np.count_nonzero(state.board))

    def _predict_logits_from_sequence(self, model: V3TemporalPolicyValueNet, seq: np.ndarray) -> np.ndarray:
        policy, _ = self._predict_policy_and_value_from_sequence(model, seq)
        return policy

    def _predict_policy_and_value_from_sequence(
        self,
        model: V3TemporalPolicyValueNet,
        seq: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        with torch.no_grad():
            x = torch.from_numpy(seq).unsqueeze(0).to(self.device)
            policy_logits, value = model.forward_with_value(x)
            logits = policy_logits.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            legal_mask = seq[-1, 8, :, :] > 0.5
            masked = np.full((BOARD_SIZE, BOARD_SIZE), -np.inf, dtype=np.float32)
            masked[legal_mask] = logits[legal_mask]
            return masked, float(value.squeeze(0).item())

    def _build_sequence(self, encoded_steps: list[np.ndarray]) -> np.ndarray:
        trimmed = encoded_steps[-self._max_window :]
        seq = np.zeros((self._max_window, 9, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        start = self._max_window - len(trimmed)
        for idx, arr in enumerate(trimmed):
            seq[start + idx] = arr
        return seq

    def _load_model(self, model_path: str) -> _LoadedModel | None:
        path = Path(model_path)
        if not model_path or not path.exists():
            return None
        try:
            payload = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location=self.device)

        meta: dict = {}
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
            raw_meta = payload.get("meta", {})
            if isinstance(raw_meta, dict):
                meta = raw_meta
        elif isinstance(payload, dict):
            state_dict = payload
        else:
            raise ValueError(f"Unsupported checkpoint format: {type(payload)!r}")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported state_dict format: {type(state_dict)!r}")

        cfg = dict(self._build_default_model().model_cfg)
        model_cfg = meta.get("model_cfg", {})
        if isinstance(model_cfg, dict):
            for key, value in model_cfg.items():
                cfg[key] = value
        cfg["max_window"] = max(int(self._max_window), int(cfg.get("max_window", self._max_window)))
        model = V3TemporalPolicyValueNet(**cfg)
        own = model.state_dict()
        merged = dict(own)
        for key, value in state_dict.items():
            if key not in own:
                continue
            if own[key].shape != value.shape:
                continue
            merged[key] = value
        model.load_state_dict(merged, strict=False)
        return _LoadedModel(model=model, loaded=True)

    def _build_default_model(self) -> V3TemporalPolicyValueNet:
        return V3TemporalPolicyValueNet(max_window=self._max_window)

    @staticmethod
    def _clone_model(model: V3TemporalPolicyValueNet) -> V3TemporalPolicyValueNet:
        cloned = V3TemporalPolicyValueNet(**model.model_cfg)
        cloned.load_state_dict(model.state_dict(), strict=True)
        return cloned

    def _model_for_context(self, mode: str, player: int) -> V3TemporalPolicyValueNet:
        color = "black" if int(player) == 1 else "white"
        key = "fast" if str(mode).strip().lower() == "fast" else "slow"
        return self._models[color][key]

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        key = requested.strip().lower()
        if key == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        if key == "cpu":
            return torch.device("cpu")
        return torch.device(requested)
