# 50M train/frozen差异：8项后续对照

**8/8任务完成全部预算，6784个记录回合均0 goal。** 同一final50M hand=.1 actor、arm1/hand.1、原生随机采样（noise multiplier1、global zeta repeat）、同一固定eraser/目标；没有新训练。6784是记录总数，不能当成6784个独立训练seed或泛化任务。

## 协议矩阵

| 已完成任务 | env | 启动处理 | 实际TF32/matmul | seed | 完成/请求 | goal | 曾抬升 | 抬升后低于初始高度 | raw return | 平均步数 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [env32_startup_seed0](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/protocol_check/env32_startup_seed0/summary.json) | 32 | 初始随机progress+1随机action | False/highest | 0 | 128/128 | 0/128 | 106/128 | 104/128 | 295.503 | 529.828 |
| [env1024_clean_seed0](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/protocol_check/env1024_clean_seed0/summary.json) | 1024 | clean | False/highest | 0 | 2048/2048 | 0/2048 | 2048/2048 | 1989/2048 | 317.524 | 599.922 |
| [env1024_startup_seed0](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/protocol_check/env1024_startup_seed0/summary.json) | 1024 | 初始随机progress+1随机action | False/highest | 0 | 2048/2048 | 0/2048 | 1183/2048 | 1125/2048 | 279.815 | 448.795 |
| [env32_clean_seed1](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/protocol_check/env32_clean_seed1/summary.json) | 32 | clean | False/highest | 1 | 128/128 | 0/128 | 128/128 | 126/128 | 317.846 | 600.000 |
| [env32_clean_tf32_seed0](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/protocol_check/env32_clean_tf32_seed0/summary.json) | 32 | clean | True/high | 0 | 128/128 | 0/128 | 128/128 | 126/128 | 316.852 | 599.430 |
| [env1024_clean_tf32_seed0](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/protocol_check/env1024_clean_tf32_seed0/summary.json) | 1024 | clean | True/high | 0 | 2048/2048 | 0/2048 | 2048/2048 | 1981/2048 | 317.613 | 599.865 |

已有clean32/seed0基准复用，没有记成新实验：goal0/128、lift128/128、低于初始高度127/128、rawreturn317.209。

启动对照必须分组：随机progress缩短初始回合可用时间，不能只拿总lift下降推断控制变差。

| 启动任务 | cohort | 回合数 | goal | 曾抬升 | 后续低于初始高度 | raw return | 步数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| env32_startup_seed0 | initial_batch | 32 | 0/32 | 10/32 | 8/32 | 234.195 | 319.312 |
| env32_startup_seed0 | subsequent_episodes | 96 | 0/96 | 96/96 | 96/96 | 315.938 | 600.000 |
| env1024_startup_seed0 | initial_batch | 1024 | 0/1024 | 159/1024 | 131/1024 | 241.994 | 297.943 |
| env1024_startup_seed0 | subsequent_episodes | 1024 | 0/1024 | 1024/1024 | 994/1024 | 317.636 | 599.647 |

## 只改变首帧上一回合reward

两组32env、seed0、native stochastic、TF32=False/highest；只在第一次显式reset前写入raw.reward_buf，随后真实reward计算和auto-reset完全保留。成功值取已记录BC成功终止reward中位数100.14024353，实际actor首帧字段=1.001402378（FP32），0组为0。intervention sidecar验证只有一次explicit reset；summary另外验证各128/128回合完成。

| 首帧reward原始值 | cohort | 完成回合 | goal | 曾抬升 | 后续低于初始高度 | raw return | 步数 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00000000 | initial_batch | 32 | 0/32 | 32/32 | 32/32 | 316.491 | 600.000 |
| 0.00000000 | subsequent_episodes | 96 | 0/96 | 96/96 | 95/96 | 317.449 | 600.000 |
| 100.14024353 | initial_batch | 32 | 0/32 | 32/32 | 31/32 | 319.813 | 600.000 |
| 100.14024353 | subsequent_episodes | 96 | 0/96 | 96/96 | 93/96 | 317.122 | 600.000 |

initial_batch就是IDs0–31，直接受干预；subsequent_episodes是IDs32–127，是后续轨迹，不能写成另外96个独立首帧干预。两组总计分别0/128。注入0与之前clean基准的完整计数和平均return完全相同，支持该包装在基准条件下未改变结果。成功reward注入改变动作但未产生goal，不能把少量drop/return差别过度解读。

## 排除了什么，尚未排除什么

- 在这一个checkpoint和这些预算内，单独增至1024env、启动bundle、TF32/high-matmul bundle、另一个rollout seed或初始成功reward，均不足以恢复训练窗口中的高goal率；不是“32env一定看不到成功”或“首帧reward=0是唯一原因”的证据。
- TF32的实际前后状态已记录且一致，但compile仍关闭；没有测试TF32×startup联合，也没有重建50M原训练进程的隐藏仿真/RNG状态。
- 在线窗口仍含策略/BN更新中的多个版本、完成回合时间选择效应；没有逐回合成功终态dump和每个版本actor，因此无法仅靠最后actor.pt验证那30个vector-step的完整事实。
- 抬升/近掌部代理不证明稳定抓握；低于初始高度也是明确几何指标，不是接触传感器确认。

## 最有区分度的两个廉价下一步（仅建议，未执行）

1. **同state同noise的编译/加载前向对照，先不跑物理。** 同一actor.pt、同一批已记录reset/抬升/近goal观测，分别走原生训练配置的compiled actor和当前eager冻结actor；固定同一cachednoise，比较μ、σ、action、BN buffers。若不一致，先定位保存/加载/编译路径；若一致，降低该路径嫌疑，再决定是否值得短rollout。TF32对照本身不能代替compile检查。
2. **在原生训练再次出现成功窗口时，原进程冻结，再clean reset。** 保存每次done的pre-reset物体/goal/keypoint误差和actor标识；冻结所有优化器及BN更新，先不重建/不额外reset跑两个horizon，再用同一冻结actor执行clean reset跑两个horizon。继续成功但clean后失败指向状态/重置分布；冻结后立刻失效指向移动策略/时间窗口；独立keypoint重建不支持日志则回到成功接口。当前原训练进程已结束，不能假装这个live-state实验已经做过，必要时在下一次受控短续训中捕获。

这些是鉴别实验，不是加reward、换网络或继续盲调超参数。

## 可追溯性

所有8组actor前后digest相同、task overrides相同、episode IDs唯一且达到请求预算。JSON列出每个summary/metadata/episodes.jsonl、两个manifest/status与intervention sidecar的SHA256，以及actor.pt SHA256。`prepared_only`保留历史准备含义，完成事实采用queue_status和实际summary。

重跑CPU汇总：`/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python /home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/summarize_protocol_followup.py`。无GPU、无core/manifest/main report修改。
