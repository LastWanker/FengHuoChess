# FengHuoChess

锋火战略棋项目（pygame + numpy）。

## 项目结构

```text
src/fenghuo_chess/
├─ constants/      # 视觉、尺寸、规则常量
├─ domain/         # 领域模型与纯规则
├─ services/       # 出锋/悔棋/模式服务
├─ application/    # GameController 流程编排
└─ ui/             # pygame 循环、渲染、输入映射、播报动画
```

## 快速运行

1. 安装依赖并以可编辑模式安装项目

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

2. 启动

```powershell
.\.venv\Scripts\python.exe -m src.fenghuo_chess.app
```

或使用脚本入口：

```powershell
.\.venv\Scripts\fenghuo-chess.exe
```

3. 启动不同对战编排模式

```powershell
.\.venv\Scripts\python.exe -m src.fenghuo_chess.app --match-mode pvp
.\.venv\Scripts\python.exe -m src.fenghuo_chess.app --match-mode pve
.\.venv\Scripts\python.exe -m src.fenghuo_chess.app --match-mode eve
.\.venv\Scripts\python.exe -m src.fenghuo_chess.app --match-mode pve --human-player 2
```

说明：
- UI 介绍页可直接给 `P1(黑)/P2(白)` 分别选择：`人类/菜鸟AI/baseline/大师AI`。
- 默认组合为：`P1=人类`，`P2=baseline`。
- `大师AI` 使用 `ModelAISource`（快/慢双权重）；可通过环境变量指定：
  - `FENGHUO_MASTER_MODEL_FAST_PATH`（快速模式权重）
  - `FENGHUO_MASTER_MODEL_SLOW_PATH`（慢速模式权重）
  - `FENGHUO_MASTER_MODEL_PATH`（单权重兜底，未分别设置时会复用）
  - `FENGHUO_MASTER_DEVICE`（默认 `cuda`）
- 未设置环境变量时，`大师AI` 默认优先读取 `artifacts/models/model_tiny_policy_slow_best.pt`。
- 只要开启 UI，AI 回合都会有至少 `1s` 的最小等待时间；headless 模式不受该限制。

4. 无 UI 自博弈（EVE）

```powershell
.\.venv\Scripts\python.exe -m src.fenghuo_chess.app --headless --match-mode eve --games 10 --output ./artifacts/selfplay.jsonl
```

5. 模型数据流水线（slow 默认）

```powershell
.\scripts\run_teacher_pipeline.ps1
.\.venv\Scripts\python.exe scripts/run_teacher_pipeline.py

.\.venv\Scripts\python.exe scripts/export_teacher_data.py --games 2000 --workers 8 --trace-output artifacts/datasets/teacher_trace_raw.jsonl --summary-output artifacts/datasets/teacher_games_summary.jsonl
.\.venv\Scripts\python.exe scripts/curate_teacher_dataset.py --trace-input artifacts/datasets/teacher_trace_raw.jsonl --summary-input artifacts/datasets/teacher_games_summary.jsonl --trace-output artifacts/datasets/teacher_trace_balanced.jsonl --summary-output artifacts/datasets/teacher_games_balanced.jsonl
.\.venv\Scripts\python.exe scripts/train_policy_model.py --trace artifacts/datasets/teacher_trace_balanced.jsonl --mode slow --output artifacts/models/model_tiny_policy_slow.pt --epochs 12 --batch-size 256 --device cuda
.\.venv\Scripts\python.exe scripts/eval_model_ai.py --model-fast artifacts/models/model_tiny_policy_fast.pt --model-slow artifacts/models/model_tiny_policy_slow.pt --opponent weak --games 50 --game-mode slow --device cuda
.\.venv\Scripts\python.exe scripts/eval_model_ai.py --model-fast artifacts/models/model_tiny_policy_fast.pt --model-slow artifacts/models/model_tiny_policy_slow.pt --opponent baseline --games 50 --game-mode slow --device cuda
```

6. 自博弈擂台升级（单轮）

```powershell
.\.venv\Scripts\python.exe -m src.fenghuo_chess.ai.selfplay_league --best-model artifacts/models/model_tiny_policy_slow_best.pt --candidate-model artifacts/models/model_tiny_policy_slow.pt --arena-games 200 --selfplay-games 400 --game-mode slow --device cuda
```

7. 一键跑一轮升级循环（导出→筛样→训练→晋升）

```powershell
.\.venv\Scripts\python.exe scripts/run_league_cycle.py
```

## 文档

- 架构说明：`docs/架构说明.md`
- 一次性改造计划：`docs/一次性改造计划.md`
- 玩法规则：`docs/玩法规则.md`
- AI 现状分析：`docs/AIplayer.md`
- AI 接入可行方案：`docs/AI接入可行方案.md`
