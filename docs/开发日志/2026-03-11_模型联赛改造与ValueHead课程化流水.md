# 开发流水日志（对话全程整理）

日期：2026-03-11  
项目：FengHuoChess  
主题：`run_model_league_cycle.py` 主线升级、value head 引入、随机机制统一、混战池与课程化改造

---

## 0. 起点与问题背景

这轮对话一开始，核心问题并不是“脚本跑不起来”，而是“训练一直能跑但增长接近停滞”，并且伴随几个实际痛点：

1. baseline 对战计算慢，拖慢全流程。
2. 指标波动大，同一套权重多次测得胜率差异明显。
3. 有时出现“二极管”现象（不是全胜就是全输），怀疑随机策略和默认参数不合理。
4. 脚本之间逻辑不统一，尤其是旧脚本 `eval_model_ai.py` 与主线脚本的偏差。
5. 希望主线聚焦 `scripts/run_model_league_cycle.py` 与 `scripts/run_model_league_cycle.py`（模型自举版本）。
6. 希望默认参数“开箱即用”，不依赖每次手调。

这不是单点 bug，而是训练系统层的“策略-数据-评估”协同问题。

---

## 1. 早期对齐：为什么结果经常不一致

对齐结论是：随机性本来就存在，而且存在于多个层面，不是单一来源。

1. 对局采样有随机性（softmax 采样、top-k 采样、对手采样）。
2. 多进程并行会放大样本组合差异。
3. 训练过程本身也有随机性（数据分割、batch 顺序、初始化）。
4. 评估样本数不足时，统计波动会非常明显。

你指出“best 对 baseline 有时能 50%+，但经常不一样”，这个观察是准确的：不是单一 bug，就算代码没错也会出现波动。后续改造目标变成“让随机变得可控、一致、可解释”，而不是追求虚假的“每次数字都一样”。

---

## 2. 关于探索参数与 softmax 的关键分歧与落地

中段讨论最关键的一段是：

1. 是否继续保留 `explore_second_prob / gap_threshold`。
2. 是否只保留 softmax 随机并统一随机入口。
3. 低探索为何会导致平均步数更高（你给出的 `53.20` vs `20.27` 这个案例非常有价值）。

最终方向是“softmax 主导 + legacy 探索弱化/归零 + 用温度控制随机强度”，避免双机制互相掐架，减少“看似随机但实际锁死”的情况。

---

## 3. 混战池主线切换与比例定稿

你明确指定主线脚本后，混战池比例经历了两次确认，最终以你最后版本为准：

1. `20% random`
2. `10% teacherAI`
3. `30% 规则 baseline`
4. `40% oldbest`

并且要求实现方法参考 `scripts/run_league_cycle.py`。这一步的目标不是“追求某个固定比率最优”，而是确保训练样本源不会塌缩到单一对手。

---

## 4. 数据积累机制修正（你特别强调的一点）

这里有一次关键纠偏：

1. 初始理解是“按最大桶对齐并补齐”，你明确否定。
2. 你的真实要求是“仍按最小桶，只是最小桶也吃到跨轮积累”。
3. 举例：第一轮桶计数 `2,3,4,5,6,7` 按 2；下一轮变成 `4,5,6,7,8,...` 按 4。

最终改成：

1. 每次脚本运行时可重置一次数据。
2. 同一运行内 round 与 round 之间不重置原始对战数据。
3. curation 继续按最小桶（`min bucket`）做无重复平衡，不做复制填充。

这套逻辑保证 round 增加是“有效样本积累”，而不是伪循环。

---

## 5. 训练轮次随 round 累进

按你的要求实现了 `2N+1` 风格的累进：

1. 初始 `epochs=5`
2. 第 3 轮开始变 6
3. 第 5 轮变 7
4. 以此类推（每 2 轮 +1）

这解决了“多轮脚本有没有价值”的核心质疑：即使 candidate 每轮从 best warm-start，也可以通过“更多数据 + 更长训练”获取增量。

---

## 6. value head 改造（policy -> policy+value）

这是一次结构级改造，不只是参数微调。

### 6.1 模型改造

在 `src/fenghuo_chess/ai/model_ai.py`：

1. `TinyPolicyCNN` 增加 `value_head`。
2. `forward()` 仍返回 policy logits，保持推理兼容。
3. 新增 `forward_with_value()` 返回 `(policy_logits, value)`。
4. checkpoint 加载 `strict=False`，兼容旧 policy-only 权重。

### 6.2 训练改造

在 `scripts/train_policy_model.py`：

1. 训练目标变为 `policy CE + value_weight * MSE`。
2. 样本中加入 `value` 目标（来自 outcome 的 `-1/0/1`）。
3. 新增 `--value-loss-weight`，默认 `0.25`。
4. warm-start 同样兼容旧权重（`strict=False`）。

这一步的目的是缓解“纯策略模仿只能稳、难上限”的问题，补长程信用分配。

---

## 7. 权重路径迁移到 value 专用目录

为了让你“直接运行主脚本即可”，模型路径主线迁移到：

`artifacts/value_plus_models`

相关脚本默认值都切到新路径，并加了“缺失自动从 legacy 拷贝”的 bootstrap 逻辑，避免第一次运行直接挂掉。

涉及脚本：

1. `scripts/run_model_league_cycle.py`
2. `scripts/export_model_selfplay_data.py`

---

## 8. candidate 是否跨失败轮次持续训练的讨论

你提出两种策略：

1. 每轮从 best 重新 warm-start（当前实现）。
2. candidate 失败后继续在 candidate 基础上训练，直到上位。

当前仍采用第 1 种。理由是更稳，避免 candidate 偏离过大导致“看似进步、实则过拟合特定窗口”的风险。配合数据积累与 epochs 累进，round 仍有实质价值。

---

## 9. 本次（最后阶段）三项“课程化”落地

你说“可以做这三个课程化”，随后要求继续施工并最终写日志。本次实际落地如下。

### 9.1 对手池课程化（round 进度驱动）

在 `scripts/run_model_league_cycle.py` 新增：

1. `--pool-opponent-curriculum {on,off}`（默认 `on`）
2. `--pool-opponent-pool-start`（默认 `random:0.2,teacherAI:0.1,baseline:0.3,oldbest:0.4`）
3. `--pool-opponent-pool-end`（默认 `random:0.1,teacherAI:0.05,baseline:0.3,oldbest:0.55`）

每轮按 `progress` 线性插值并归一化，下发给 export。

### 9.2 温度课程化（round 进度驱动）

新增：

1. `--pool-temperature-curriculum {on,off}`（默认 `on`）
2. `--pool-sample-temperature-start`（默认 `1.2`）
3. `--pool-sample-temperature-end`（默认 `0.9`）
4. `--pool-sample-temperature`（作为课程关闭时的固定值）

这让早期更发散、后期更收敛。

### 9.3 hard-case 资产池与加权训练

这部分分三段打通：

1. `scripts/export_model_selfplay_data.py` 的 summary 增加：
   - `opponent_kind`
   - `best_player / best_color`
   - `best_outcome`
   - `pool_mode`
2. `scripts/curate_model_dataset.py` 增加 hard-case 机制：
   - `--hard-case-weight`
   - `--hard-case-opponents`（默认 `baseline,teacher`）
   - 桶内采样从纯随机升级为“hard-case 加权无放回抽样”
   - 对命中 hard-case 且为 best 侧的 trace 行注入 `sample_weight`
3. `scripts/train_policy_model.py` 支持读取 `sample_weight`：
   - 最终权重 = `outcome_weight * sample_weight`

再由 `run_model_league_cycle.py` 每轮下发 hard-case 权重：

1. `--hard-case-curriculum {on,off}`（默认 `on`）
2. `--hard-case-weight-start`（默认 `1.2`）
3. `--hard-case-weight-end`（默认 `2.0`）
4. `--hard-case-weight`（课程关闭时固定值）
5. `--hard-case-opponents`

---

## 10. 本次联调验证（小规模 smoke）

执行了一轮小规模验证，核心命令是 `run_model_league_cycle.py` 一轮低配参数。

观察到的关键结果：

1. 脚本打印了课程化参数：`progress=0.00`、`pool_temp=1.200`、`hard_case_weight=1.200`。
2. export summary 已包含 `opponent_kind`、`best_outcome` 等新字段。
3. curation 输出显示：
   - `hard_case_games_selected=20`
   - `hard_case_rows_weighted=464`
4. 训练阶段正常跑通（黑白 candidate 均完成）。
5. 对战晋升流程也跑通，且该次 smoke 出现 `promote_pair=PASS`。

说明三条课程化链路已经接通，不是只改了参数入口。

---

## 11. 这次对话里的“反复确认点”记录

这部分单独记下，防止后续再次误解：

1. “按最小桶平衡”是硬约束，不要复制填充追最大桶。
2. round 间数据要累积，同一次脚本运行内不重置。
3. 主线脚本是 `scripts/run_model_league_cycle.py`。
4. 默认值要偏实战，不依赖每次手调。
5. 旧脚本可修但不应主导结论。
6. 随机机制要统一，避免多套探索逻辑互相冲突。

---

## 12. 当前状态总结

到本日志落笔时，系统状态可以概括为：

1. 已从“固定池 + 固定温度 + 均匀采样”升级为“round 课程化 + hard-case 强化”。
2. 已从 policy-only 升级为 policy+value 联训。
3. 已把主线默认模型目录切到 `artifacts/value_plus_models`，并保留 legacy 兼容。
4. 已验证一轮端到端可运行。

后续工作重点就不再是“修脚本能不能跑”，而是：

1. 观察多轮课程化是否稳定抬升晋升通过率。
2. 观察 hard-case 加权是否带来真实泛化提升，而非短期噪声收益。
3. 根据真实日志再细调课程曲线（池比例、温度曲线、hard-case 权重区间）。

---

## 13. 额外备注（给后续排障）

1. 如果后续出现“二极管回归”，优先查：
   - 对手池是否塌缩到单一来源。
   - 温度是否过低。
   - hard-case 权重是否过大导致分布偏移。
2. 如果出现“指标看起来升了但晋升不过”：
   - 查 duel 总量与置信度。
   - 分离看黑白两侧是否一强一弱互相抵消。
3. 如果训练明显变慢：
   - 优先看 baseline 比例和 arena 游戏数，而不是先怀疑模型结构。

---

（本文件为本次完整对话的工程流水整理版，后续如有新轮次改造，建议按日期追加新日志文件，而不是覆盖本文件。）
