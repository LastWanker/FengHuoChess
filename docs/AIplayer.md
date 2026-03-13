# AI Player 现状分析与修缮建议（基于当前代码）

## 1. 文档目的

这份文档不再作为“开发指令提示词”，而是作为当前项目接入 AI 的真实现状评估与修缮清单。

适用代码基线：

- `src/fenghuo_chess/application/game_controller.py`
- `src/fenghuo_chess/domain/*`
- `src/fenghuo_chess/services/*`
- `src/fenghuo_chess/ui/*`

## 2. 目标对照（现实状态）

### 目标 A：规则逻辑是唯一真实来源

- 现状：`基本达成`
- 依据：
  - 胜负、阶段、合法性、计分在 `domain.rules/domain.scoring`
  - 出锋/截锋在 `services.exposure_service`
  - UI 不直接改棋盘规则，仅通过 `GameController.make_move`
- 备注：规则层已不依赖 pygame，这是正向基础。

### 目标 B：可深拷贝、可序列化 GameState

- 现状：`部分达成`
- 已有：
  - `GameState` 已集中在 `domain/models.py`
  - `HistoryService` 使用 `copy.deepcopy` 快照
- 缺口：
  - 还没有 `to_dict/from_dict` 或稳定序列化协议
  - `GameState` 混有 UI 状态字段（`show_intro`、重开按钮状态），不利于训练样本纯净化

### 目标 C：纯逻辑接口 `apply_action` / `legal_actions`

- 现状：`未达成（核心缺口）`
- 当前只有：
  - `GameController.make_move(row, col) -> bool`（就地修改内部状态）
  - `domain.rules.is_valid_move`（单点合法性）
- 缺少：
  - `legal_actions(state)` 批量合法动作枚举
  - `apply_action(state, action) -> new_state, events` 的无副作用接口

### 目标 D：UI/播报与逻辑解耦

- 现状：`部分达成`
- 已有：
  - 领域规则和服务不依赖 pygame
- 缺口：
  - `ExposureService` 仍通过 broadcast callback 推送文本事件
  - `GameController` 中混有偏 UI 的播报文本和颜色管理

### 目标 E：AI 作为“输入源”而非直接改棋盘

- 现状：`可行但未封装`
- 说明：
  - 任何调用方都可通过 `GameController.make_move` 落子，理论可接 AI
  - 但还没有 `PlayerSource` 抽象，也没有“人机/机机”回合驱动层

### 目标 F：训练数据导出

- 现状：`未达成`
- 缺少：
  - 对局轨迹记录器
  - `state/legal_actions/action/outcome` 标准化导出格式（JSONL/Parquet）

### 目标 G：保持现有 PVP 可运行

- 现状：`已达成`
- 当前 UI 流程稳定通过 `pygame_main -> GameController` 运行。

## 3. 当前架构中最适合承载 AI 接口的位置

- `domain/`：放“纯函数规则工具”（`legal_actions`, state 编码）
- `application/`：放“回合执行接口”（`apply_action`, `simulate_action`）和事件汇总
- `services/`：放“轨迹记录器/数据导出器”
- `ui/`：仅作为一种输入源实现（Human），后续可并列 AI 输入源

## 4. 现实修缮清单（最小侵入）

1. 新增逻辑 API 层（不改 UI 主流程）
- 新建 `application/logic_api.py`
- 提供：
  - `legal_actions(state) -> list[Action]`
  - `apply_action(state, action) -> StepResult`（纯逻辑）
  - `simulate_action(state, action)`（内部深拷贝）

2. 新增标准事件模型（替代散落文案触发）
- 新建 `domain/events.py`
- 事件类型示例：`MOVE_APPLIED`, `WIN`, `EXPOSURE`, `STAGE_CHANGED`, `GAME_OVER`
- UI 层把事件翻译为中文播报和动画，不让规则层直接持有 UI 文案

3. 规范 GameState 序列化
- 在 `domain/models.py` 增加：
  - `state_to_dict(state)`
  - `state_from_dict(data)`
- `numpy board` 统一转 list 或压缩字符串

4. 新增训练导出器
- 新建 `services/trace_recorder.py`
- 每步记录：
  - `state`
  - `legal_actions`
  - `action`
  - `player`
  - `events`
- 对局结束补 `outcome`

5. 保留现有 UI 路径
- `GameController.make_move` 可先内部调用新 `logic_api.apply_action`，对 UI 无感迁移。

## 5. 风险与现实约束

- 出锋/截锋有连锁反应，`apply_action` 必须一次性结算完整回合，否则 AI 评估会失真。
- 当前 `GameState` 含 UI 字段，若直接导出会污染训练数据；需要分离“核心状态视图”。
- 现有回放系统缺失，建议导出 trace 时顺带保存 `seed/version` 用于复现。

## 6. 结论

当前项目已经具备 AI 接入的关键地基（规则集中、控制层统一、UI 与规则基本分层），但离“可训练、可模拟”的标准还差三件关键能力：

- `legal_actions`
- 纯逻辑 `apply_action/simulate_action`
- 结构化对局导出

这三项可以在不破坏现有 PVP 的前提下增量完成。

