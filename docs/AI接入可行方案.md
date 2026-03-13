# AI 接入可行方案（最小侵入、可落地）

## 1. 方案目标

在不破坏现有 PVP 和 UI 体验的前提下，补齐 AI 所需最小能力：

1. 无 UI 的纯逻辑一步结算
2. 合法动作枚举
3. 可复现的状态序列化
4. 训练数据导出
5. 人类/AI 统一为“玩家输入源”
6. PVP/PVE/EVE 统一编排（运行层，不与快/慢 `game_mode` 冲突）

## 2. 设计原则

- 单一规则源：AI 与 UI 都调用同一套规则，不复制逻辑。
- 小步快跑：先补接口，不上来重写控制器。
- 向后兼容：`pygame_main + GameController` 入口不改或最小改。
- 可测试优先：新增能力必须可在无 pygame 下运行。

## 3. 推荐分阶段

## Phase 1：纯逻辑 API（1-2 天）

新增文件：

- `src/fenghuo_chess/application/logic_api.py`
- `src/fenghuo_chess/domain/events.py`

建议接口：

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Action:
    row: int
    col: int

@dataclass
class StepResult:
    ok: bool
    state: "GameState"
    events: list["LogicEvent"]
    reason: str | None = None

def legal_actions(state: GameState) -> list[Action]: ...
def apply_action(state: GameState, action: Action) -> StepResult: ...
def simulate_action(state: GameState, action: Action) -> StepResult: ...
```

实现要求：

- `apply_action` 复用当前 `GameController.make_move` 的结算顺序：
  - 落子 -> 立即胜负 -> 出锋连锁 -> 阶段推进 -> 终局计分 -> 切换玩家
- `simulate_action` 必须在深拷贝状态上运行，不污染原状态。

## Phase 2：状态序列化与训练样本（1-2 天）

新增文件：

- `src/fenghuo_chess/domain/serialization.py`
- `src/fenghuo_chess/services/trace_recorder.py`

建议结构：

```python
def state_to_dict(state: GameState, core_only: bool = True) -> dict: ...
def state_from_dict(data: dict) -> GameState: ...
```

`core_only=True` 时仅导出训练必要字段：

- `board`
- `stage`
- `current_player`
- `game_mode`
- `mode_locked`
- `game_over`
- `winner`
- `stage_positions`
- 出锋核心状态（用于一致复现）

训练样本格式（JSONL）：

```json
{
  "state": {...},
  "legal_actions": [[r, c], ...],
  "action": [r, c],
  "player": 1,
  "events": ["MOVE_APPLIED", "STAGE_CHANGED"],
  "outcome": null
}
```

对局结束后回填 `outcome`（1/-1/0）。

## Phase 3：AI 作为输入源（1 天）

新增文件：

- `src/fenghuo_chess/application/player_source.py`
- `src/fenghuo_chess/ai/baselines/random_ai.py`

抽象：

```python
class PlayerSource(Protocol):
    def select_action(self, state: GameState, legal_actions: list[Action]) -> Action | None: ...
```

类型：

- `HumanPlayerSource`：来自鼠标点击
- `RandomAISource`：随机合法动作
- 后续 `MinimaxSource` / `MCTSSource`

UI 仅负责在当前玩家是 Human 时处理点击；AI 回合调用 `select_action` 后再走 `apply_action`。

## Phase 4：无 UI 对局运行器（1 天）

新增文件：

- `src/fenghuo_chess/ai/selfplay_runner.py`

能力：

- headless 对局循环（不初始化 pygame）
- 指定双方策略（随机/规则/后续模型）
- 自动导出 trace

## 补充：PVP / PVE / EVE 对战编排（0.5-1 天）

新增文件：

- `src/fenghuo_chess/application/match_runner.py`

建议抽象：

```python
@dataclass
class MatchConfig:
    match_mode: Literal["pvp", "pve", "eve"]
    black_source: PlayerSource
    white_source: PlayerSource
    with_ui: bool = True
```

运行约束（与当前代码结构对齐）：

- `PVP`：`HumanPlayerSource` vs `HumanPlayerSource`。通常 `with_ui=True`；`with_ui=False` 也可运行，但没有额外输入源时不会有有效人类输入。
- `PVE`：`HumanPlayerSource` vs `AISource`。通常 `with_ui=True`；若要 headless，需要把人类侧替换为脚本输入源（如 `ScriptedPlayerSource`）。
- `EVE`：`AISource` vs `AISource`。默认 `with_ui=False` 跑自博弈；也可 `with_ui=True` 做可视化演示。
- `match_mode` 是“对战编排模式”，`game_mode(fast/slow)` 是“规则参数模式”，两者并行存在且互不替代。

## 4. 与现有代码的最小改造点

1. `GameController` 改为调用 `logic_api.apply_action`（内部替换，外部签名可不变）
2. `consume_broadcasts` 改为从 `events` 转译（先保留旧文案映射）
3. `ui/pygame_main.py` 不关心规则细节，只关心“当前玩家输入源”
4. `app.py/pygame_main.py` 只负责 `with_ui=True` 的事件循环；headless 由 `match_runner` 驱动

## 5. 验收标准（现实可执行）

1. 无 UI 构造初始状态并能获取合法动作列表
2. `simulate_action` 不修改原状态
3. `apply_action` 与现有 PVP 一致（含出锋连锁、阶段推进）
4. 可导出至少 100 局 JSONL 对局数据
5. PVP 入口仍可运行
6. PVP/PVE/EVE 三种编排都可跑通（其中 PVP/PVE 在无 UI 时需显式提供替代输入源）

## 6. 推荐先做的最小 Demo

先实现：

- `legal_actions`
- `simulate_action`
- `RandomAISource`

然后做一个“人类 vs 随机 AI”演示，验证接口正确性，再继续上训练导出与更强策略。

## 7. 不建议当前就做的事

- 暂不引入复杂神经网络训练框架（PyTorch/RLlib）到主仓库
- 暂不改动 UI 动画/主题相关代码
- 暂不追求并行自博弈性能优化（先保证规则正确）
