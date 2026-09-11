# Reward 曲线来源与口径

本目录只读取现有TensorBoard数据，不启动训练、仿真或评测。

## 本地0–100M图

- 失败参考：`full_nodr_20260909_1004/runs/str_full_nodr/full_distribution_seed0`，完整1200资产分布、原140D前馈actor、arm=.1、hand=.1，100M。
- 简单任务：`overnight_20260908/repro60m/runs/repro60m/arm1_hand01_seed{0,1,2}`，固定eraser/起点/goal、162D前馈actor、arm=1、hand=.1，各60M。三个seed全部展示，未挑选最佳。已有最终确定性评测62/64、1/64、64/64；随机策略评测127/128、88/128、128/128，不能隐去seed1确定性不稳定。
- 若已产生真实event，蓝色虚线加入 `control_ab_20260909_arm1/runs/str_control_ab/full_arm1_seed0`：完整任务只把arm=.1改为1的新试训，停在图例注明的实际观测步数，不补齐到100M。它不是完成后的最终结果。
- 横轴直接采用上述FlashSAC事件记录的环境transitions，不乘env数或PPO rollout长度。每1M transitions做TB窗口等权平均，再跨三个简单任务seed取中位数与min/max。阴影不是置信区间，没有向缺失训练预算外推。
- `episode/return`是未经learner reward normalization的环境累计reward；`episode/cumulative/*`是在每个episode累计各reward项，episode结束才上报；`episode/final/successes`是每episode完成goal数。代码依据：`flash_rl/envs/isaaclab.py:278–299`。TB窗口均值不保留全体episode计数，因此此图没有重建episode加权的全程精确总体均值。
- 一次性抬升奖300分：曾越过任务高度阈值，不证明稳定抓取，弹起也可能触发。连续抬升项在曾触发阈值后关闭，因此它下降不一定代表退步。
- 简单任务成功一次就结束，完整任务最多50个goal；goal数不是同任务成功率比较。部分奖励分量没有列入六panel（goal bonus、关节速度惩罚），不能拿图中可见分量相加期望恰好等于总reward。

## 原STR历史参考

最终绘图的KUKA主来源是**xug6已验证合并历史0→66.453504B**：`/home/lixiyuan/simtoolreal/tb/s2r_seed0_full/merged`。只读提取保存了 `remote_sources/xug6_original/scalars_early_raw_100m.json`（每项255个原始点，0→99.876864M）与 `scalars_binned100m.json`（每项665个100M bin，原始每项169001点）；metadata保留event路径/大小/时间及merge manifest、verify report的SHA256。没有下载约958MB原event。早期图按1M重新聚合原始点，全历史直接使用已聚合100M均值，不重复平滑。

已核对并保存两条 xug52 参考事件及其原始 Hydra 配置；精确路径、设置、末值和 tag 映射见 `remote_sources/xug52_sources.json`：

- `xug52_original_kuka_full_dr`：原 STR 的 KUKA iiwa14 + Sharpa、完整 DR、SAPG，保留原始 resume 横轴 `33.974B–66.384B`。
- `xug52_wuji_no_dr`：M6 + Wuji 左手、关闭 DR/noise/delay 的 SAPG 续训，保留原始横轴 `38.507B–62.145B`。这是不同 embodiment；而且 action moving average 仍为 arm `0.1`、hand `0.07`，不能标成作者建议的两者 `1.0`。

图上KUKA只用xug6；xug52 KUKA副本保留作来源检查，不与xug6强行拼接或作为另一次独立实验重复画。xug6合并历史末尾100M-bin均值与xug52续训最后单点不相同；也不能因一个合并历史尾点回落就宣称训练崩溃。

原STR的全训练预算独立用B（十亿transitions）作横轴；每个tag都需声明步数转换和纵轴统计口径。不把PPO的iteration当transitions，也不把每步reward乘episode长度伪造累计reward。未记录的分量标“未记录”，不是0。

两条 SAPG 均采用 `mixed_expl`、24576 env、4096 block。远端实际版本在 `a2c_common.py` 中令 `ignore_env_boundary=24576-4096=20480`：`rewards/*`、`episode_cumulative/*` 和 `episode_final/*` 只统计最后4096个环境完成的episode，不能描述为全24576环境均值。`successes_per_block/block_0..5` 才是各探索block的显式诊断。`rewards/step` 与 `rewards/iter` 数值相同，并都以事件中的全局frame（环境transitions）作 `Event.step`。

原SAPG最初100M的平均goal/episode也只有约0.000763。数十B后的成功不能直接拿来证明FlashSAC在100M时已不可能学会。`episode_cumulative/*`与本地所画分量同属episode累计量纲，但原observer使用最近已完成episode队列，FlashSAC使用日志窗口，采样和权重不同。

与简单任务、完整FlashSAC之间同时存在机器人、观测、控制、DR、任务和预算差异时，只能作学习阶段与历史参考，不能当严格算法优劣对照。

## 复现

```bash
/home/abao/play2perfect/.venv_isaacsim/bin/python /home/abao/flashsac-robotics/diagnostics/reward_comparison_20260909/plot_reward_comparison.py --self-test
/home/abao/play2perfect/.venv_isaacsim/bin/python /home/abao/flashsac-robotics/diagnostics/reward_comparison_20260909/plot_reward_comparison.py
```

输出 `local_reward_components.png`、数据齐全时的 `original_str_reference.png`，以及包含精确event目录、tag和原始scalar序列的 `curves.json`。同一run目录多event文件只通过EventAccumulator目录读取一次；不同run不静默合并。
