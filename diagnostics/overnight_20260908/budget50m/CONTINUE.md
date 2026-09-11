# 50M 预算对照：已准备，未启动

状态：本目录只生成和 CPU 验证配置/manifest。**没有启动训练、评测、GPU 或后台队列。主任务核查后才能启动，并必须等待 `warmstart/retention_status.json` 对应队列结束。**

## 为什么做

此前每个 seed 10M 内没有达到目标，不能据此宣布 FlashSAC 永远学不会。这里在同一固定 STR 单物体/初态/目标任务上，延长连续采样预算，观察 10/20/30/40/50M 的变化。仍非论文复现或收敛保证，只做 seed0 的预算诊断。

- A：arm moving-average=1.0，hand=1.0。
- B：arm moving-average=1.0，hand=0.1。
- 每项从零连续训练 **50,001,920 transitions = 48,830 并行交互 = 5 × 10,000,384**。
- 1,024 环境、10M replay ring buffer、batch 2048、UTD、原生 Flash Actor/Critic、162维 state、reward、控制器、reset、goal、终止逻辑均沿用同一 Phase2 配方。
- 不加载模型/优化器/replay，不把 10M 分段重启五次；50M 内 replay 仍是容量10M的连续环形 buffer。

## 防止预算变化同时改变前10M的学习率

已核对 Phase2 完全解析配置：

```yaml
agent.learning_rate_warmup_step: 0
agent.learning_rate_decay_step: 19532
```

两项在本轮明确覆盖为同样值，**不让 50M 总预算重新计算这两个字段**。init/peak/end 也不变。各优化器仍沿用自身更新计数，达到该日程终点后学习率保持 end 值；不按50M重新拉长 cosine。这里验证的是前10M训练配方一致，不承诺 GPU 仿真随机数/数值结果逐位重现。

只额外关闭无实际 episode 的视频记录 cadence：新旧均 `num_record_episodes=0`、`num_record_envs=null`；不影响训练数据和优化器。训练内在线评测仍为0 episodes。每50步写TB、每250步额外统计，与Phase2相同。

`configs/*.json` 是两份 Hydra 完全解析配置；`config_assertions.json` 记录与各自 Phase2 baseline 的逐字段差异，允许的只有预算衍生字段、输出路径/标签和上述零episode记录cadence。两份新配置之间只有 hand filter 和输出名不同。

## 保存与评测

每9,766次交互保存一次：

| checkpoint | transitions |
| --- | ---: |
| step9766 | 10,000,384 |
| step19532 | 20,000,768 |
| step29298 | 30,001,152 |
| step39064 | 40,001,536 |
| step48830 | 50,001,920 |

每个已保存 checkpoint 后续使用32环境独立评测：deterministic 64回合 + 原生 stochastic 128回合，匹配训练的 arm/hand filter。当前 runner 是串行队列：先完整训练一个50M，再逐个评测其保存模型，避免训练中断/replay重载；之后另一个条件。

最终50M两个条件都做CPU Critic诊断；hand=0.1额外做10M、30M Critic诊断。共26 jobs：2连续训练、20 rollout评测、4 CPU Critic。

目标成功从任务的 `all_goals_hit` 判断；抬升阈值、持续接近掌部等代理与真正目标成功分开。固定初始设置，不把重复回合当作多初态泛化。最终无成功仍不构成无限预算不可学习的证明。

## 由主任务核查后执行（此命令尚未执行）

```bash
/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python /home/abao/flashsac-robotics/scripts/run_str_diagnostic_queue.py /home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/manifest.json --status /home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/queue_status.json --after-status /home/abao/flashsac-robotics/diagnostics/overnight_20260908/warmstart/retention_status.json
```

**不要省略 `--after-status`。** manifest 中的 `required_prior_status` 本身只是说明字段，现有 runner 不会自动读取它。主任务需先核查 retention 阶段结果，再决定是否排队；本目录没有自行启动任何任务。

保留现有 runner 北京时间09:00硬截止（2026-09-09 01:00 UTC），不能延长。每项训练单job超时7200秒；评测/CPU Critic各1200秒。两项训练加评测可能超过一小时，GPU等待和运行速度变化也会影响耗时；不承诺具体完成时间。截止前未完成的项目保留状态，不扩预算、不自动改参数。

训练要求至少36GiB空闲显存，评测至少12GiB。现有runner不杀他人任务，只能清理自己启动的进程组。`depends_on` 记录单job来源，但runner仅保证串行顺序，不执行逐job成功依赖；上游失败后下游CLI会因缺文件失败，不会自动重训。

训练完成标记选 `step48830/agent_state.pt`（现有保存函数最后写入），但若中断仍应核对同目录6个checkpoint文件完整性。不要将中间actor文件视为整个50M完成，也不要覆盖已有部分结果。

## 文件

- `manifest.json`：26项队列。
- `configs/`：两份解析配置。
- `config_assertions.json`：CPU断言。
- `source_hashes.json`：启动前源码/配置/manifest SHA256；修改后需重新核查。
- `models/<condition>/seed0/step*/`：未来模型输出。
- `runs/budget50m/<condition>/`：未来TB。
- `evaluations/<condition>_seed0/step*/<sampling>/`：未来逐回合评测。
- `critics/`、`logs/`：未来诊断与运行日志。
- `prepare_manifest.py`：只生成/验证清单，不运行队列；发现已有queue_status即拒绝覆盖。

```bash
/home/abao/play2perfect/.venv_isaacsim/bin/tensorboard --logdir=/home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/runs/budget50m --host=127.0.0.1 --port=6010 --load_fast=false
```
