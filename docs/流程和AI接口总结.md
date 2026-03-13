# 流程和 AI 接口总结（当前阶段）

更新时间：2026-03-09

## 1. 当前状态总览

项目已完成从“纯 UI 对战”到“统一逻辑 + 多模式编排 + headless 自博弈”的改造，核心特征如下：

- 规则结算已抽离为纯逻辑接口（`logic_api`），UI 与 AI 共用同一规则源。
- 对战编排支持 `PVP / PVE / EVE` 包装，也支持直接按 `P1/P2` 选手类型组局。
- UI 启动介绍页可直接配置 `P1 黑` 与 `P2 白` 的输入源（人类/菜鸟AI/baseline/大师AI）。
- UI 模式下 AI 落子有最小 1 秒等待；headless 模式不加延迟。
- 终局后不会继续落子；悔棋与重开在终局状态下仍可用（权限高于终局遮罩）。

## 2. 主流程

### 2.1 UI 主流程（`ui/pygame_main.py`）

入口：`run(match_mode='pvp', human_player=1)`（UI 介绍页默认会覆盖为 `P1=human, P2=baseline`）

流程要点：

1. 介绍页阶段：
   - 选择 `intro_p1_source`（`human/weak/baseline/master`）
   - 选择 `intro_p2_source`（`human/weak/baseline/master`）
   - 点击开始后调用 `controller.configure_players(...)` 进入对局

2. 对局阶段：
   - 人类点击棋盘通过 `controller.make_move(...)` 触发一步结算
   - AI 回合由 UI 循环驱动 `controller.step_turn()`
   - 若双方均为 AI，则自动连续推进；若含人类，则按“人类触发后 AI 跟手”推进
   - 仅 UI 模式启用 AI 最小等待：`UI_AI_MIN_TURN_DELAY = 1.0`

3. 终局阶段：
   - `game_over=True` 后停止自动推进
   - 结束弹窗可关闭（仅隐藏，不改变 `game_over`）
   - `undo/restart` 始终可点击，且优先级高于终局遮罩

### 2.2 Headless 主流程（`ai/selfplay_runner.py`）

入口：`run_selfplay(games=..., output_path=...)`

流程要点：

- 每局 `create_initial_state()`，默认 `MatchRunner('eve', with_ui=False)`。
- `while not state.game_over` 循环：
  - `legal_actions(state)`
  - `runner.select_action(...)`
  - `apply_action(...)`
- 可选导出 JSONL trace。
- 不包含 UI 延迟逻辑，也不提供交互式悔棋入口。

## 3. AI 接口与编排接口

### 3.1 纯逻辑接口（`application/logic_api.py`）

- `Action(row, col)`
- `StepResult(ok, state, events, reason)`
- `legal_actions(state) -> list[Action]`
- `apply_action(state, action) -> StepResult`
- `simulate_action(state, action) -> StepResult`（深拷贝，不污染原状态）

`apply_action` 当前结算顺序：

1. 落子合法性校验
2. 立即成线胜负判定
3. 出锋链处理
4. 截锋后再次胜负判定（已补）
5. 阶段推进 / 终局计分
6. 切换玩家（若未结束）

### 3.2 输入源接口（`application/player_source.py`）

- `PlayerSource.select_action(state, legal_actions) -> Action | None`
- `HumanPlayerSource`：通过 `submit_action(...)` 注入点击动作
- `RandomAISource`：随机合法动作基线（`ai/baselines/random_ai.py`）
- `HeuristicAISource`：规则老师基线（`ai/baselines/heuristic_ai.py`）
- `ModelAISource`：模型推理对手（`ai/model_ai.py`，支持快/慢双权重；通过 `source_kind=master` 接入）

### 3.3 对战编排（`application/match_runner.py`）

- `MatchConfig(match_mode, black_source, white_source, human_player, p1_source_kind, p2_source_kind, with_ui)`
- `build_match_runner(...)`
- `MatchRunner.source_for_player(...) / select_action(...) / submit_human_action(...)`

编排规则：

- 包装模式：`pvp/pve/eve`（兼容旧入口）
- 直接模式：`p1_source_kind + p2_source_kind`（UI 当前主用）

## 4. Controller 分工（`application/game_controller.py`）

`GameController` 目前职责：

- 持有 `GameState`、`HistoryService`、`ExposureService`、`MatchRunner`
- 把一步推进委托给 `logic_api.apply_action`
- 维护历史快照（`undo`）
- 管理广播事件缓存（供 UI 消费）
- 管理模式切换、重开确认等交互状态

## 5. 终局与遮罩逻辑（当前约定）

- `game_over` 是“对局是否结束”的唯一硬状态。
- `end_overlay_closed` 只控制“结束弹窗是否显示”。
- 关闭结束弹窗不会恢复对局。
- 悔棋/重开可在终局后执行，并可改变 `game_over`。

## 6. 序列化与训练数据

### 6.1 状态序列化（`domain/serialization.py`）

- `state_to_dict(state, core_only=True)`
- `state_from_dict(data)`

`core_only=True` 面向训练最小字段；`core_only=False` 保留 UI 相关状态。

### 6.2 Trace 导出（`services/trace_recorder.py`）

每步样本字段：

- `state`
- `legal_actions`
- `action`
- `player`
- `events`
- `outcome`

支持局终后回填 outcome，并写入 JSONL。

## 7. CLI 入口现状

`app.py`：

- UI：`--match-mode pvp|pve|eve`，`--human-player 1|2`
- Headless：当前仅允许 `--match-mode eve`

示例：

```powershell
python -m src.fenghuo_chess.app --match-mode pve --human-player 2
python -m src.fenghuo_chess.app --headless --match-mode eve --games 100 --output ./artifacts/selfplay.jsonl
```

## 8. 当前边界与后续建议

- 当前 AI 仅有 `RandomAISource` 基线；可后续扩展 `Minimax/MCTS/模型推理`。
- headless CLI 暂只开放 EVE；若未来要支持 headless PVE/PVP，需要引入脚本输入源。
- 终局弹窗已与终局状态解耦，但终局后 UI 仍允许悔棋/重开（按当前产品需求）。
