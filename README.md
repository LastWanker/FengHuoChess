# FengHuoChess

锋火战略棋项目（Python + pygame + numpy）。
独⽴设计并实现原创战略棋类 FengHuoChess（Python / Pygame / PyTorch），从规则引
擎、UI 交互到 AI 训练系统均⾃主设计。
棋局采⽤四阶段动态战场（3×3→15×15）与“出锋-截锋”机制，形成具有跨阶段⻓期
布局特性的策略环境。
基于该环境构建完整训练闭环（⾃博弈数据⽣成、分桶均衡采样、模型训练、联赛评估与
晋升），实现 AI 从规则⽼师 → CNN policy/value → 时序 Transformer 的演进。
项⽬累计运⾏多轮⼤规模实验（单轮 2000+ 局，数万步样本），并通过随机性治理与评估公
平性设计提升训练稳定性与可复现性。

## 当前状态

- 规则引擎、PVP/PVE/EVE、headless 自博弈已稳定可用。
- V2 模型链路（导出 -> 筛样 -> 训练 -> 擂台晋升）可运行。
- V3（CNN + Temporal Transformer + Value 决策）已落地，包含独立数据、训练、评估与联赛脚本。

## 快速开始

1. 安装依赖（建议 Python 3.11+）：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

2. 启动 UI：

```powershell
.\.venv\Scripts\python.exe -m src.fenghuo_chess.app
```

3. 启动 headless（EVE）：

```powershell
.\.venv\Scripts\python.exe -m src.fenghuo_chess.app --headless --match-mode eve --games 20 --output artifacts/selfplay.jsonl
```

## V3 相关命令

1. V3 对战评估（示例）：

```powershell
.\.venv\Scripts\python.exe scripts/eval_v3_model_ai.py --opponent baseline --games 30 --game-mode slow --device cuda
```

2. V3 联赛循环（示例）：

```powershell
.\.venv\Scripts\python.exe scripts/run_v3_league_cycle.py --rounds 3
```

> 说明：V3 脚本默认读写 `artifacts/v3_transformer/` 目录，尽量与 V2 产物隔离。

## 文档保留策略

- 保留：本 `README.md`
- 保留：`docs/玩法规则.md`
- 其它文档默认不入库（避免上传隐私内容）
