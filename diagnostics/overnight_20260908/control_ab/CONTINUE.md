# Phase 2：FlashSAC 手指滤波短 A/B

状态：只准备配置与队列；本文生成时未启动此阶段。必须等待 Phase 1 队列结束，由主任务核查后启动。

## 要回答的问题

官方 checkpoint 在更改动作滤波后可能失效，但这不能证明使用该控制设置从零训练的 FlashSAC 一定学不会。本实验直接测试：保持手臂滤波 1.0，只把手指滤波从 1.0 改为 0.1，是否改善当前固定抓取搬运任务的学习。

- A：arm_moving_average=1.0，hand_moving_average=1.0。
- B：arm_moving_average=1.0，hand_moving_average=0.1。
- 三对训练 seed：0、1、2，按 seed0 A/B、seed1 A/B、seed2 A/B 排队。
- 每个运行从零训练 10,000,384 transitions，1,024 环境，9,766 次并行交互。
- 原生 FlashSAC Actor/Critic、奖励、replay 10M、batch 2048、更新比率、学习率日程、目标、重置与 162 维输入不改。
- 不是原论文复现或完整收敛预算，也不是 Dex4D 3-stage/student 流程。

配置位于 `configs/`；六份是 Hydra 完全解析后的 trainer 配置。与第一个运行相比，允许不同的字段只有 seed 及其 agent/env 引用、手指滤波、输出标签/保存路径。CPU 断言结果和共享配置 SHA 在 `config_assertions.json`。共同的 Isaac 注册任务 YAML 仍由原环境入口在启动时应用。

## 每个运行之后做什么

1. 保存准确的最终 `step9766` checkpoint；保存目录不带 TIMESTAMP。
2. 在相同手臂/手指控制设置下，各评测 128 次 deterministic 和 128 次 native stochastic rollout。
3. 评测使用 32 环境，seed 与该 A/B 对相同；固定初态任务，不把 128 回合当作泛化覆盖。
4. 在 CPU 上对 stochastic rollout 数据和该运行 checkpoint 的 reward normalizer 做 Critic 诊断。

主要比较：目标成功率、物理抬升过程、持续离桌且靠近掌部代理指标、掉落、目标误差、reward 各项与 Q/Actor 诊断。代理指标不等于经过接触验证的稳定抓取。三 seed 的方向比单次高 reward 更可信；没有改善也不能据此宣布 LSTM 必需或 FlashSAC 不可用。

## 文件位置

根目录：`/home/abao/flashsac-robotics/diagnostics/overnight_20260908/control_ab`

- 队列：`manifest.json`，共 24 个串行 job（6 train + 12 eval + 6 CPU critic）。
- 模型：`models/arm1_hand1/seed0/step9766/` 等六个目录。
- TensorBoard：`runs/control_ab/arm1_hand1/` 和 `runs/control_ab/arm1_hand01/`，logger 自身的运行名仍会附带时间以避免覆盖；模型路径没有时间后缀。
- 评测：`evaluations/<condition>_seed<n>/<deterministic|stochastic>/summary.json`、`episodes.jsonl`、`transitions.npz`。
- Critic：`critics/<condition>_seed<n>.json`。
- 运行日志：`logs/`；重试写 attempt 后缀，不覆盖旧日志。
- 队列状态：启动时指定此目录的 `queue_status.json`。

## 主任务核查后的启动命令

```bash
/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python /home/abao/flashsac-robotics/scripts/run_str_diagnostic_queue.py /home/abao/flashsac-robotics/diagnostics/overnight_20260908/control_ab/manifest.json --status /home/abao/flashsac-robotics/diagnostics/overnight_20260908/control_ab/queue_status.json --after-status /home/abao/flashsac-robotics/diagnostics/overnight_20260908/queue_phase1_status.json
```

截止时间：2026-09-09 01:00 UTC，即北京时间 09:00。可以提前结束，不能由该 runner 延长。依赖队列仍运行时本队列显示 waiting_dependency；依赖队列 completed/finished_with_failures 后才允许启动；依赖队列 deadline/interrupted 则停止并返回非零。

训练前要求至少 36 GiB 空闲显存，评测前至少 12 GiB；10M replay 存储拼接的 324 维 observation 和 next observation，另需仿真与优化器开销。队列记录其他 GPU 进程，但不杀它们。超时/截止只清理自己启动的进程组。不会无限续训、自动扩大预算或改参数。

runner 依据最终完成文件跳过完成项；manifest 的 depends_on 是审计记录，单 job 上游失败不会触发替代训练，后续缺少模型/数据会报错并保留日志。若中断留下不完整模型或评测目录，先核查并保留这些产物，不要盲目覆盖或把部分文件认作完成。训练的完整 checkpoint 应同时包含 actor、critic、target critic、temperature 和 reward normalizer 等文件。

## 查看本阶段 TensorBoard

```bash
/home/abao/play2perfect/.venv_isaacsim/bin/tensorboard --logdir=/home/abao/flashsac-robotics/diagnostics/overnight_20260908/control_ab/runs/control_ab --host=127.0.0.1 --port=6010 --load_fast=false
```

## 核查训练期间代码是否变化

`source_hashes.json` 保存 41 个实际源码/配置/manifest 的绝对路径与 SHA-256，包括当前未提交的文件内容。捕获时间为北京时间 2026-09-08 21:25:27；队列记录首项训练随后于 21:25:45 启动，哈希采集本身没有启动任务。文件还保存了 24 个任务的执行路径、输出父目录和启动脚本可执行性检查结果。

明天若怀疑源码被改过，运行以下只读检查。全部显示 OK 才说明这些文件仍与捕获时一致；FAILED 需要结合任务启动时间解释，不能把后续改动后的运行混入同一 A/B 结论。哈希不覆盖完整 Isaac/CUDA 依赖或动态生成资产。

```bash
/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python - <<'PY'
import json, subprocess
path = "/home/abao/flashsac-robotics/diagnostics/overnight_20260908/control_ab/source_hashes.json"
with open(path) as stream:
    files = json.load(stream)["files"]
checks = "".join(item["sha256"] + "  " + item["resolved_path"] + "\n" for item in files)
subprocess.run(["sha256sum", "--check"], input=checks, text=True, check=True)
PY
```
