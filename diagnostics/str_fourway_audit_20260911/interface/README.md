# STR 环境接口与实际控制链：已完成的证据

## 结论

**当前 STR Lab 原生入口与 FlashSAC wrapper，回放同一段真实策略动作时，59 个记录字段逐元素一致，最大差值 0。** 没有发现 wrapper 在本次测试中改变 observation、动作、控制目标、reward、物理演化、goal 切换或终止/超时。

这是当前配置和所覆盖轨迹的接口验证，不是“整个项目永远无 bug”，也不是官方 legacy Gym 端到端等价，更不能保证长期训练收敛。

## 测试是什么

- 两个独立 Python 进程。Native 使用 `AppLauncher → SimToolRealEnvCfg → 注册 YAML → 明确 arm1/noDR 覆盖 → gym.make`，没有调用适配器构造或覆盖函数；wrapper 使用当前生产适配入口。
- 冻结 100M Actor，Native 用其确定性 `tanh(mean)` 产生动作磁带，wrapper 完整回放，不学习、不修改 checkpoint、不填 replay。
- 32 个环境 × 600 个 policy step，共 19,200 个环境步。配置仍是完整随机 STR 任务池；本次实际选择 brush 9、hammer 6、screwdriver 4、spatula 4、eraser 7、marker 2，共六类32个实例，不冒充遍历1200个资产。
- arm EMA=1.0，hand EMA=0.1，DR 关闭；容差固定 `0.02905653603374958`，不让课程改变比较标准。
- 两边初始观测、机器人/物体/目标状态、配置、资产文件名、控制参数、Actor 文件 hash、动作磁带 hash，以及 Python/NumPy/Torch CPU/CUDA 初始 RNG hashes 全部一致。
- 脚本输出 `status.json`，在启动、场景准备、reset 和每100步记录状态。此次没有运行 native-self，因为跨入口对照本身已经逐元素一致；未声称“测过 native-self”。

## 真正覆盖了哪些分支

| 事件 | 次数 | 含义 |
|---|---:|---|
| 真终止 | 30 | `terminated=True` 的环境步 |
| 时间截断 | 11 | `truncated=True` 的环境步 |
| goal 命中 | 33 | `_pending_goal_reset` 置位的环境步，不是33个完整任务 |
| 有效 final_obs | 41 | 以上终止或截断对应的 reset 前观测 |

轨迹包含自动 reset 后的后续回合，因此这些不是“32条首回合”的成功率统计。没有完整50-goal任务成功的宣称。

## 控制链：命令确实送到了哪里

共有556,800个标量动作，全部有限且位于 `[-1,1]`；实际收到的动作与裁剪后的磁带差值为0。

独立 NumPy 公式检查了 canonical→Lab 排序、机械臂速度增量、手部绝对位置缩放、EMA 平滑和关节限位：

- 公式算出的 target 与任务产生的 target：最大差值 **0 rad**。
- 任务产生的 target 与 PhysX 收到的关节 target：最大差值 **0 rad**。
- target 超出关节限位的标量数：**0**（检查容差 `1e-6 rad`）。
- 每个 policy step 实际调用 `_apply_action` **2次**；physics dt=`0.00833333333 s`，policy dt=`0.01666666666 s`。
- 机械臂每个 policy step 的最大目标变化 `0.024999976 rad`，符合 `1.5 rad/s × 1/60 s` 的增量上限。

这支持“动作接口和目标生成没有接错”。Actor 本身的动作已合法，所以这段 GPU rollout 没有覆盖故意超过 `[-1,1]` 的输入；CPU自测仅验证了控制 oracle 的越界裁剪公式，不能将其说成越界wrapper实测。

## target 与实际关节位置不同，并不等于接错

误差取 **physics step 结束、自动 reset 之前** 的 `abs(target - actual_joint_position)`，避免把新回合 reset 位置与旧回合 target 混在一起。

| 全轨迹位置误差 | 平均 | 中位数 | P95 |
|---|---:|---:|---:|
| 机械臂 | 0.03026 rad | 0.01542 rad | 0.10215 rad |
| 手 | 0.14764 rad | 0.03277 rad | 0.74248 rad |

**target 是指令，不是下一时刻必然达到的位置。** 动力学滞后、接触阻挡都会产生误差。以上说明存在跟踪误差，但没有力矩/接触力记录，不能仅据此断言 Kp/Kd 错、力矩饱和或抓取稳定。另附去除各回合 reset 后10步瞬态的统计，结论未被 reset 瞬态主导。

## final_obs 不是误拿新回合观测，也不是比较无效旧缓存

- 有效 mask 与 `terminated | truncated` 完全一致。
- **41/41** 有效 final_obs 都不同于自动 reset 后返回的 observation；其中 **11/11** 时间截断也是如此。
- 从单独记录的 reset 前物理状态，重新计算关节归一化观测、关节速度、目标、lift latch、goal计数、progress和当前reward，与 final_obs 对照；最大误差 `2.384185791015625e-7`。
- 所有 done 行的 reset 后 episode/goal计数及lift latch均清零；所有非done行的 reset 前后关节位置差值为0。
- 无效 final_obs 行被显式置零，不用未初始化或历史缓存充数。

## 应如何使用这个结论

可以暂时降低“当前 wrapper 把这批动作/观测/终止信号传错”作为训练失败原因的优先级，继续检查数据覆盖、Q排序和策略行为。

尚未证明：原 legacy Gym 与当前 Lab 的相同物理结果；全1200资产/全部随机状态；开启DR的路径；训练初始化特有的随机 episode counter（本次 `random_start_init=False`）；接触/执行力矩与真实硬件一致；策略为何不能稳定搬运。

## 文件

- [跨进程逐字段结果](native_vs_wrapper.json)
- [控制链与 final_obs 独立分析](control_analysis.json)
- [Native metadata](native/metadata.json)、[Wrapper metadata](wrapper/metadata.json)
- Native/Wrapper 各自保存 `trace.npz`、`actions.npy` 和实际任务配置。
- 实验脚本：`/home/abao/flashsac-robotics/scripts/check_str_interface_rollout.py`。

本次只新增诊断代码与结果，没有修改生产任务、算法、奖励或 checkpoint。

