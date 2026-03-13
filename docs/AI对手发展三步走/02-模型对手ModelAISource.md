# 第二步：模型对手（ModelAISource）

## 本步定位

- 目标：把“老师经验”压缩成更快的推理器。
- 交付：仅新增 1 个代码文件。
- 新增文件：`src/fenghuo_chess/ai/model_ai.py`

## 关键约束（基于当前工程）

- 不改 `PlayerSource` 接口。
- 不改 `logic_api` / `match_runner` 主流程。
- 模型只负责：给合法动作打分并返回一个 `Action`。

## 文件内需要提供的能力

`model_ai.py` 一文件内包含：

- `class ModelAISource:`
- `__init__(model_path: str, device: str = "cpu")`
- `encode_state(state) -> tensor/ndarray`
- `predict_logits(state) -> 15x15 logits`
- `select_action(state, legal_actions) -> Action | None`

合法动作掩码规则：

- 对非法坐标 logits 置为 `-inf`
- 只在合法动作集合中 `argmax`

## 模型形态（来自 raw2，按当前阶段收敛）

第一版固定为 `TinyPolicyCNN`：

- 输入：多通道 15x15（黑白子、当前手、阶段 one-hot、模式、合法 mask）
- 输出：`1 x 15 x 15` 动作分数图
- 推理：合法掩码 + `argmax`

> 说明：训练脚本可暂在仓外或 notebook 先跑，仓内先落推理接入，确保主流程快速上线。

## 验收标准

- `ModelAISource` 可直接替换 `RandomAISource` 跑 EVE。
- 全程无非法落子。
- 推理速度显著快于逐手 `simulate_action` 的老师 AI（同机对比）。

## 一次性实现建议

- 先只交 `model_ai.py`，权重可先用占位或外部导入。
- 确认接口跑通后，再补训练脚本（不阻塞主线）。
