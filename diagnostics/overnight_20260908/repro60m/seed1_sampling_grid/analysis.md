# Seed1 60M：冻结推理采样对累计达标的影响

6个新增评测全部完成；另引用并重新逐帧核验原确定性与native1.0基线。没有训练、参数更新、控制器变化；变的是推理采样标准差倍率/时间相关性/rollout seed。

| 条件 | 成功/全部 | 相同起始32回合成功 | 成功中连续10帧 | 成功中离开后重入 | 全部回合原始reward均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| deterministic | 1/64 | 1/32 | 0/1 | 1/1 | 1147.62 |
| std025 | 33/128 | 5/32 | 1/33 | 32/33 | 1203.35 |
| std05 | 52/128 | 5/32 | 2/52 | 50/52 | 1230.54 |
| std075 | 69/128 | 10/32 | 16/69 | 53/69 | 1262.64 |
| native1_seed0 | 88/128 | 19/32 | 16/88 | 72/88 | 1291.24 |
| std15 | 103/128 | 25/32 | 39/103 | 64/103 | 1308.75 |
| native_per_step | 82/128 | 21/32 | 12/82 | 70/82 | 1292.04 |
| native_seed1 | 88/128 | 23/32 | 23/88 | 65/88 | 1290.30 |

std025/std05/std075/std15是原生时间相关采样，pre-tanh标准差分别乘.25/.5/.75/1.5；native1_seed0是已有1.0基线。native_per_step保持1.0但每步独立抽噪声；native_seed1保持原生1.0，仅更换评测随机种子。确定性是tanh(mean)，不是新训练policy。

倍率越大的这些已测点成功率越高，但这只是当前范围、同一训练权重/固定任务的冻结采样结果；不能把1.5称为最优，也不能当成训练时加大探索能提高学习效果的因果证明。独立每步82/128与原生88/128不构成统计显著性结论；换一个rollout seed也88/128不等于轨迹相同或训练复现。

## 独立核验

所有回合逐帧重建reward所用固定尺寸角点最大距离：≤3cm记一次near，累计第10次恰好成功terminated；所有未成功回合600步truncated。goal bonus严格等于100×near帧数；sticky抬升与300分lift bonus匹配；逐帧reward总和与回合return一致。回合内部next_obs[t]==obs[t+1]完全相同，末帧使用pre-reset final_obs，未跨reset差分。

8组是同一个actor checkpoint，评测前后模型digest相同；实际actor与数据文件SHA256保留。完整resolved config哈希记录在JSON；只归一化eval seed（顶层/agent/env）与save_path后8组配置哈希相同，其他配置无漂移。

最初32回合step0观测对native seed0基线的最大绝对差：0。后续reset时刻随轨迹变化；全128不是逐条状态相同的配对轨迹。32回合只是完整预算子集，不能重复计入样本量。

## 失败与成功不能只靠reward判断

| 条件 | 失败near次数:回合数 | 成功时间中位数(s) | 成功终点FD速度均值(m/s) | 成功终点FD角速度均值(deg/s) |
| --- | --- | ---: | ---: | ---: |
| deterministic | 7:5, 8:55, 9:3 | 0.700 | 0.120 | 69.6 |
| std025 | 7:18, 8:55, 9:22 | 0.700 | 0.154 | 74.2 |
| std05 | 6:3, 7:15, 8:35, 9:23 | 0.683 | 0.155 | 64.6 |
| std075 | 6:1, 7:8, 8:33, 9:17 | 0.683 | 0.151 | 64.5 |
| native1_seed0 | 4:1, 6:3, 7:6, 8:12, 9:18 | 0.700 | 0.125 | 68.4 |
| std15 | 0:1, 4:1, 7:4, 8:9, 9:10 | 0.750 | 0.136 | 77.2 |
| native_per_step | 7:5, 8:22, 9:19 | 0.692 | 0.165 | 70.1 |
| native_seed1 | 7:13, 8:13, 9:14 | 0.700 | 0.128 | 71.6 |

失败仍可以取得7–9次near带来的700–900分奖励；成功也只是累计10帧，并不要求连续停稳。速度是16.667ms区间平均差分。数据成功即终止，不能判断后续保持、接触力或稳定抓握，更不能据此说可以部署真机。

![采样倍率与真实成功率](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m/seed1_sampling_grid/success_vs_sampling.png)

- All share same trained checkpoint and fixed task, not independent training seeds.
- Multiplier scales pre-tanh sampling std externally, not entropy temperature alpha and not a SAC training intervention.
- Native baseline1.0/global-zeta/seed0 is reused, not relabeled a new trial.
- noise-repeat1 is independent per-step noise rather than native global-zeta repetition.
- Altered RNG draws and trajectories mean evaluations are not bitwise paired state/action trajectories.
- Improved cumulative-goal rate alone does not show stable grasp, post-goal hold, or deployment readiness.
- All full128 vs initial32 results are separate denominators; initial32 is a subset, not another independent trial.
- Points summarize one fixed trained policy/task. No statistical significance or optimal multiplier claim.
- Motion is per-transition backward finite difference; no episode boundary crossing, no post-success holding frames.
- Seeds here are rollout RNG seeds, not new training seeds. Frozen inference sampling does not identify the cause of training exploration gains.

CPU复核：`/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python /home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m/seed1_sampling_grid/analyze.py`。无GPU、无核心/manifest/main report修改。
