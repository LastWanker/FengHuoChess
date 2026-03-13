"""Model-backed AI source with legal-action masking."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import random

import numpy as np

from src.fenghuo_chess.application.logic_api import Action
from src.fenghuo_chess.constants.gameplay import BOARD_SIZE
from src.fenghuo_chess.domain.models import GameState

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - import guarded for optional runtime dependency.
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


class TinyPolicyCNN(nn.Module):
    """Small policy-value network.

    Forward keeps backward compatibility by returning policy logits only.
    Use ``forward_with_value`` when value prediction is needed.
    """

    def __init__(self, in_channels: int = 9, hidden_channels: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(
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
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        policy, _ = self.forward_with_value(x)
        return policy

    def forward_with_value(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.body(x)
        policy = self.head(feat)
        value = self.value_head(feat).squeeze(-1)
        return policy, value


class ModelAISource:
    """Policy model source that picks argmax over legal actions."""

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
        explore_second_prob: float = 0.0,
        explore_gap_threshold: float = 0.25,
        sample_top_k: int = 0,
        sample_temperature: float = 1.0,
    ) -> None:
        if torch is None:
            raise RuntimeError(
                "ModelAISource requires torch, but importing torch failed."
            ) from _TORCH_IMPORT_ERROR

        self.model_path = model_path
        self.device = self._resolve_device(device)
        self._rng = rng or random.Random()
        self._explore_second_prob = max(0.0, min(1.0, float(explore_second_prob)))
        self._explore_gap_threshold = max(0.0, float(explore_gap_threshold))
        self._sample_top_k = max(0, int(sample_top_k))
        self._sample_temperature = max(1e-6, float(sample_temperature))
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
        self._models: dict[str, dict[str, TinyPolicyCNN]] = {
            "black": {"fast": TinyPolicyCNN(), "slow": TinyPolicyCNN()},
            "white": {"fast": TinyPolicyCNN(), "slow": TinyPolicyCNN()},
        }
        loaded: dict[str, dict[str, bool]] = {
            "black": {
                "fast": self._load_weights_if_present(self._models["black"]["fast"], self._paths["black"]["fast"]),
                "slow": self._load_weights_if_present(self._models["black"]["slow"], self._paths["black"]["slow"]),
            },
            "white": {
                "fast": self._load_weights_if_present(self._models["white"]["fast"], self._paths["white"]["fast"]),
                "slow": self._load_weights_if_present(self._models["white"]["slow"], self._paths["white"]["slow"]),
            },
        }
        self._mirror_missing_modes(loaded, "black")
        self._mirror_missing_modes(loaded, "white")
        self._mirror_missing_colors(loaded)

        for color_models in self._models.values():
            for model in color_models.values():
                model.to(self.device)
                model.eval()

    def encode_state(self, state: GameState) -> torch.Tensor:
        """Encode GameState into model input tensor: [1, C=9, 15, 15]."""
        board = state.board
        channels = np.zeros((9, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        channels[0] = (board == 1).astype(np.float32)
        channels[1] = (board == 2).astype(np.float32)
        channels[2].fill(1.0 if int(state.current_player) == 1 else 0.0)

        stage_idx = max(1, min(4, int(state.stage))) - 1
        channels[3 + stage_idx].fill(1.0)

        channels[7].fill(1.0 if state.game_mode == "fast" else 0.0)

        legal_mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for row, col in state.stage_positions:
            if board[row, col] == 0:
                legal_mask[row, col] = 1.0
        channels[8] = legal_mask

        tensor = torch.from_numpy(channels).unsqueeze(0).to(self.device)
        return tensor

    def predict_logits(self, state: GameState) -> np.ndarray:
        """Predict raw action logits as a 15x15 float array."""
        model = self._model_for_context(state.game_mode, int(state.current_player))
        with torch.no_grad():
            encoded = self.encode_state(state)
            logits = model(encoded).squeeze(0).squeeze(0)
        return logits.detach().cpu().numpy().astype(np.float32, copy=False)

    def select_action(self, state: GameState, legal_actions: Sequence[Action]) -> Action | None:
        if not legal_actions:
            return None

        logits = self.predict_logits(state)
        masked = np.full((BOARD_SIZE, BOARD_SIZE), -np.inf, dtype=np.float32)
        for action in legal_actions:
            masked[action.row, action.col] = logits[action.row, action.col]

        if not np.isfinite(masked).any():
            return self._rng.choice(list(legal_actions))

        ranked = sorted(
            ((float(masked[action.row, action.col]), action) for action in legal_actions),
            key=lambda item: item[0],
            reverse=True,
        )
        if not ranked:
            return self._rng.choice(list(legal_actions))

        chosen = self._sample_from_topk(ranked) or ranked[0][1]
        # Softmax top-k sampling is the primary stochastic path.
        # Legacy top2 exploration only applies when top-k sampling is disabled.
        if self._sample_top_k <= 0 and chosen == ranked[0][1] and self._should_pick_second(ranked):
            chosen = ranked[1][1]
        if chosen in legal_actions:
            return chosen

        return self._rng.choice(list(legal_actions))

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

    def _should_pick_second(self, ranked: list[tuple[float, Action]]) -> bool:
        if self._explore_second_prob <= 0.0:
            return False
        if len(ranked) < 2:
            return False
        gap = ranked[0][0] - ranked[1][0]
        if gap > self._explore_gap_threshold:
            return False
        return self._rng.random() < self._explore_second_prob

    def _load_weights_if_present(self, model: TinyPolicyCNN, model_path: str) -> bool:
        path = Path(model_path)
        if not model_path or not path.exists():
            return False

        try:
            payload = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location=self.device)
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        elif isinstance(payload, dict):
            state_dict = payload
        else:
            raise ValueError(f"Unsupported checkpoint format: {type(payload)!r}")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported state_dict format: {type(state_dict)!r}")
        # Allow loading legacy policy-only checkpoints that do not contain value_head.
        model.load_state_dict(state_dict, strict=False)
        return True

    def _mirror_missing_modes(self, loaded: dict[str, dict[str, bool]], color: str) -> None:
        fast_loaded = loaded[color]["fast"]
        slow_loaded = loaded[color]["slow"]
        if fast_loaded and not slow_loaded:
            self._models[color]["slow"].load_state_dict(self._models[color]["fast"].state_dict(), strict=True)
            loaded[color]["slow"] = True
        if slow_loaded and not fast_loaded:
            self._models[color]["fast"].load_state_dict(self._models[color]["slow"].state_dict(), strict=True)
            loaded[color]["fast"] = True

    def _mirror_missing_colors(self, loaded: dict[str, dict[str, bool]]) -> None:
        black_any = loaded["black"]["fast"] or loaded["black"]["slow"]
        white_any = loaded["white"]["fast"] or loaded["white"]["slow"]
        if black_any and not white_any:
            self._models["white"]["fast"].load_state_dict(self._models["black"]["fast"].state_dict(), strict=True)
            self._models["white"]["slow"].load_state_dict(self._models["black"]["slow"].state_dict(), strict=True)
        if white_any and not black_any:
            self._models["black"]["fast"].load_state_dict(self._models["white"]["fast"].state_dict(), strict=True)
            self._models["black"]["slow"].load_state_dict(self._models["white"]["slow"].state_dict(), strict=True)

    def _model_for_context(self, mode: str, player: int) -> TinyPolicyCNN:
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
