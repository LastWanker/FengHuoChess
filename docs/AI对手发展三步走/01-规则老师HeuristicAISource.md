# 第一步：规则老师 AI（HeuristicAISource）

## 本步定位

- 目标：最快得到“明显不蠢、稳定可用”的 AI 对手。
- 交付：仅新增 1 个代码文件。
- 新增文件：`src/fenghuo_chess/ai/baselines/heuristic_ai.py`

## 为什么这一步可以一次性完成

当前项目已经具备完整依赖链：

- 规则试走：`simulate_action(state, action)`
- 合法动作：`legal_actions(state)`
- 统一接口：`PlayerSource.select_action(state, legal_actions)`
- EVE 对局循环：`run_selfplay(...)`

所以本步不需要改 UI、不需要改主流程，只要补一个 `PlayerSource` 实现即可接入。

## 与现有接口对齐方案

在 `heuristic_ai.py` 中定义：

- `class HeuristicAISource:`
- `def select_action(self, state, legal_actions) -> Action | None`
- `def score_state(self, state, me: int) -> float`

动作选择流程：

1. 遍历 `legal_actions`
2. 对每个动作调用 `simulate_action`
3. 用 `score_state` 给结果局面打分
4. 选最大分动作（并加少量随机打破平分）

## 评分函数（第一版硬规则）

只做短程、可解释规则：

- 直接赢（`state.game_over and winner == me`）：`+1e6`
- 直接输：`-1e6`
- 阶段 2/3 触发出锋且己方受罚风险：大幅扣分
- 控制中心/当前阶段可用区占位：中等加分
- 阶段 4 三连库存（`count_threes` 差值）：中等加分
- 其余局面：轻量子力与连线潜力分

## 验收标准（强执行）

- 能直接作为 `PlayerSource` 在 `MatchRunner` 中运行。
- 在 headless EVE 下能稳定跑完 200 局无异常。
- 对 `RandomAISource` 胜率目标：`>= 70%`。

## 一次性实现建议（本周直接做）

- 仅提交一个文件：`heuristic_ai.py`
- 用现有脚本临时评估（可先用内联 Python，不必先加 `scripts/eval_ai.py`）
- 达标后再进入第二步
