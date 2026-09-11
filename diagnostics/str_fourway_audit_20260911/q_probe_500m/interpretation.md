# 500M 正式 Q 动作探针：结论不确定，不能据此归因 Critic

来源：本目录 `summary.json`；32 个初始状态，prefix 0 / 60 / 120，单次动作干预，后续冻结 Gaussian 策略，最长 600 步。没有训练。

## 配对是否成立

- prefix 0：7 个分支各 32/32 通过可见状态和 RNG 检查；最大初始误差 `5.96e-8`。
- prefix 60 与 120：**所有分支都是 0/32 可用配对**。这些阶段的 Q 方向结论应记为 `null / unavailable`，不是成功率 0 或 Critic 错误。
- prefix 60 已出现 episode 计数差 52–59 步，prefix 120 差 113–115 步；Torch CPU/CUDA 随机状态不同。不同终止／goal 事件已导致随机流分叉，但目前没有逐步记录，无法确定最先发生的是哪一处物理或数值差异。

## 即使 prefix 0 配对，长时间重复噪声仍太大

mean 与 mean_repeat 不改变任何动作干预，却产生有限 soft return 绝对差：均值 `0.165474`，中位数 `0.054416`，最大 `0.743578`；9/32 对的是否真正终止都不同。最高物体高度差最大 `0.333574 m`，goal 数差最大 6。

18 对 mean／repeat 都真正终止：此子集没有未知尾部，回报绝对差均值仍为 `0.0987285`、最大 `0.678885`。因此，**问题不只是截断尾部，单次同 seed 长轨迹比较的方差本身就很大**。

| 单步干预 | 有限回报差超过本次 repeat 最大差的例数 | 同向例数 | 双方都真正终止 | 真终止子集中超过 repeat 最大差 |
|---|---:|---:|---:|---:|
| zero | 3/32 | 2/3 | 19 | 0 |
| arm Q+ | 0/32 | 不适用 | 17 | 0 |
| arm Q− | 0/32 | 不适用 | 17 | 0 |
| hand Q+ | 1/32 | 1/1 | 14 | 0 |
| hand Q− | 0/32 | 不适用 | 15 | 0 |

本次 repeat 最大值只是经验参照，不是置信界。真终止子集也有选择偏差：动作会改变谁终止，不能把这个子集当总体无偏效果。表中稀少极端值不能用来宣称 Q 正确或错误。

未真正终止的轨迹，未知尾折扣均为 `0.99^600=0.00240501`。这个数是折扣权重，不是剩余回报的数值上界；单步 Q+/- 变化约 `0.0003–0.0009`，仍可能受未知尾部影响。

prefix 0 的 32 个资产：brush 9、eraser 7、hammer 6、screwdriver 4、spatula 4、marker 2；不是全 1,200 资产覆盖。prefix 60/120 没有有效样本，故没有可报告的中途搬运 Q 证据。

## 对重置原因的源码核对

当前 `env.reset()` 调用 `_reset_idx`、`scene.write_data_to_sim()` 和 `sim.forward()`，**不会停止并重建整个 PhysX 仿真**：`/home/abao/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py:273`。

`raw.sim.reset(soft=False)` 则通过 IsaacSim 的 stop/play 重置仿真视图；资产在 STOP 失效、PLAY 重建物理句柄：

- `/home/abao/IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py:486`
- `/home/abao/IsaacLab/_isaac_sim/exts/isaacsim.core.api/isaacsim/core/api/simulation_context/simulation_context.py:624`
- `/home/abao/IsaacLab/source/isaaclab/isaaclab/assets/asset_base.py:304`

因此硬重置是**合理的诊断候选，不是已经证明有效的修复**。它可能改变初始化后通过 PhysX tensor views 写入的摩擦等参数；STR 的 `apply_physx_material_properties` 只在环境初始化时调用。若测试硬重置，必须保持初始化与每个对照一致、恢复并验证物理参数，然后再 seed + task reset；先只验证重复 baseline 的 120 步重放，不直接重复整个 Q 实验。

也不能直接把当前差异归咎于“PhysX 缓存”：初始可见状态与 RNG 一致并不包含内部接触／求解器缓存，GPU 接触运算本身也可能不确定，当前输出没有逐步首次分叉证据。

**本轮合理结论：Q 排序实验未取得可信的搬运阶段因果证据。保持严格 `1e-5` 配对门限，不修改训练、不提高门限硬凑结果，也不把诊断的不可重复性称为训练失败根因。**

后续有界检查已完成：见 `../q_hardreset_baseline/interpretation.md`。使用硬重置并保持原物理参数后，32 个环境的 120 步参考／重放逐步全部通过状态与 RNG 配对；这解决了该短前缀诊断的可重复性条件，但没有重跑本目录的 7 分支 600 步 Q 对照，本目录原 Q 结论仍为不确定。
