# FengHuoChess

锋火战略棋项目（Python + pygame + numpy）。

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
