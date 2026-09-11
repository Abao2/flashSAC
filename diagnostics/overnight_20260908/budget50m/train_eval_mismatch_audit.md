# 50M训练成功与冻结评测失败：独立CPU核对

结论：配置/代码检查没有发现goal换了、reward换了、观测顺序错了或成功率平均方式错了。但**训练末尾80.21%的短窗口，不能替代最终checkpoint的独立评测**；hand=.1最终det0/64、stoch0/128。它能短暂抬升，并非完全没学到动作；多数回合随后掉落，尚不能称为可靠搬运策略。

本核对只读取既有文件和CPU执行日志/网络函数。未启动Isaac、GPU、训练，未修改环境、配置或checkpoint。

## 1. 配置哪些相同，哪些不同

训练保存的JSON与TensorBoard的`config/text_summary`逐项相同。训练和最终冻结评测的`env.task_cfg_overrides`逐项相同：

| 项目 | 训练与冻结评测 |
| --- | --- |
| 机器人/物体 | 同一个STR任务入口；IIWA+SHARPA，单个固定生成eraser |
| observation | actor162维，critic162维；字段、顺序相同，wrapper拼成324维，actor只取前162 |
| action | 29维，wrapper clamp[-1,1]×1；arm filter1、hand filter.1 |
| 控制 | arm为增量目标：1.5×dt×action；hand为关节limit内绝对目标后滤波 |
| start pose | (0,0,.5559664621)，单位四元数 |
| goal pose | (.05,0,.7059664621)，单位四元数；1个goal |
| 判定 | tolerance.02×keypoint_scale1.5=.03m；10个near-goal帧，可不连续 |
| DR/noise | 所列object/obs/action delay关闭，状态噪声0、力/力矩0、friction multiplier1 |
| 时间 | 注册配置physics dt≈1/120、decimation2，评测metadata实际policy dt≈1/60；600步/10秒 |
| tolerance实际日志 | 976个训练窗口均.02，没有隐含变松 |

关键不同：训练1024 env、在线更新；冻结32 env、不更新。训练初次reset默认随机化episode进度，评测默认不随机化；训练从scratch还先收集约98个vector-step的随机动作满足100k replay门槛，而冻结直接用checkpoint。训练的初始随机进度不在每次auto-reset重加，不能单独解释50M末尾。

冻结还把buffer/batch调至1、关闭optimizer/normalizer加载、关闭compile；配置中的学习率日程/预算/cadence数值随32 env改变，但没有优化器更新，不应据这些静态数值声称实际学了不同策略。是否存在数值执行精度影响，需另外匹配实测。

源证据：[wrapper构造与同一配置入口](/home/abao/flashsac-robotics/flash_rl/envs/isaaclab.py:180)、[相同action clamp](/home/abao/flashsac-robotics/flash_rl/envs/isaaclab.py:361)、[实际控制](/home/abao/simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/action_utils.py:50)、[native取actor字段](/home/abao/flashsac-robotics/flash_rl/agents/flashSAC/agent.py:470)、[native采样的training=False BN](/home/abao/flashsac-robotics/flash_rl/agents/flashSAC/agent.py:234)、[训练初次reset/随机动作](/home/abao/flashsac-robotics/train.py:127)、[随机progress](/home/abao/flashsac-robotics/flash_rl/envs/isaaclab.py:327)。

### 几何信息的来源边界

评测实际`object_scales × .04`给出物体边长 **(.14605714,.05659970,.04993292)m**，与官方teacher dataset一致。`.04`是normalization base size，不是实际物体大小；reward的固定keypoint几何 **(.141,.03025,.0271)m** 也不是实际box大小，两者用途不同。

table顶面metadata=.53m，依据table URDF碰撞box与table reset_z计算，不是测到的实际碰撞接触面；训练未保存逐步几何dump，不能把配置相同写成已逐帧验证了两个进程的物理状态一致。单物体生成器使用固定seed42，与env数无关；无证据表明这一物体尺寸本身随32/1024改变。见[物体生成调用](/home/abao/simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py:1744)、[scale映射](/home/abao/simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py:1700)、[评测几何计算](/home/abao/flashsac-robotics/scripts/diagnose_str_rollouts.py:53)。

## 2. 80.21%日志有没有算错？

已追踪完整路径并执行CPU fixture：

1. `_get_dones`先把本步成功加到当前`_successes`；`all_goals_hit = _successes >= 1`。
2. `_get_rewards`在auto-reset之前发布`episode_final`；bool转float产生独立张量，不是`_prev_episode_successes`。
3. Isaac随后执行reset；wrapper只选本步done环境，返回`(均值, done_count)`。
4. TB logger用`mean × count / total_count`加权，不是按每个vector-step等权平均。

CPU实际函数测试：第一批1/2成功、第二批1/1成功，结果2/3，不是3/4；故意把`_prev_episode_successes`置成9再将当前成功计数清零，已发布的成功仍是正确0/1快照。976个TB窗口`all_goals_hit`与`done_max_successes`相同；这是同源交叉一致性，不是独立物理成功证据。

源证据：[成功定义](/home/abao/simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/logging_utils.py:8)、[发布在reset前](/home/abao/simtoolreal/isaacsimenvs/tasks/simtoolreal/simtoolreal_env.py:80)、[DirectRLEnv先reward再reset](/home/abao/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py:371)、[wrapper done_count加权](/home/abao/flashsac-robotics/flash_rl/envs/isaaclab.py:289)、[logger接受权重](/home/abao/flashsac-robotics/flash_rl/common/logger.py:65)、[平均器实现](/home/abao/flashsac-robotics/flash_rl/common/logger.py:125)。

最后窗口是vector-step48800→48830，即最后30步的完成回合；policy和BN仍在更新，且有完成时间选择效应。TB没有保存窗口episode总数，所以不能唯一反推80.20833%对应多少分子/分母，更不能声称它是50M全程成功率或最终冻结模型成功率。

## 3. 冻结hand=.1具体失败在哪

64个det回合：全部曾抬过10cm，**0个进入真实reward keypoint的3cm范围，55个随后低于初始物体高度**；128个stoch回合127个随后低于初始高度。`reset_when_dropped=false`，所以落回桌面后仍能继续到600步timeout。

det每回合最佳keypoint误差范围3.768–5.098cm，中位4.600cm；即使物体中心偶尔进3cm，方向/角点仍不合格。并非已到goal但success logger漏记。

代表回合固定选episode0，未按结果挑选：step21首次跨过10cm；step25达到最佳keypoint误差4.700cm；最高抬升13.215cm；step469首次低于初始高度；step600 timeout，最终keypoint误差25.255cm。曾满足“近掌部、低相对位移”的代理不代表接触确认或始终抓稳。

![实际冻结轨迹](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/frozen50m_episode0_trace.png)

成功误差按reward专用keypoint重建，未直接把观测keypoint长度或中心距离当成功；逐回合最终重建值与已存summary匹配。见[reward角点与成功累计](/home/abao/simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/obs_utils.py:182)、[评测采用pre-reset final_obs](/home/abao/flashsac-robotics/scripts/diagnose_str_rollouts.py:525)、[pose重建](/home/abao/flashsac-robotics/scripts/eval_str_state_teacher.py:12)。

## 4. 高价值候选：reset观测保留上一回合reward

当前162维包含`reward`。fresh启动为0；auto-reset不清`reward_buf`，所以成功终止后新回合第一帧可能含约1的reward channel（raw约100×.01）。这不是本次修改制造的行为。

直接证据：[只有初始化清零reward_buf](/home/abao/simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/reset_utils.py:210)、[任务reset没有清reward](/home/abao/simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/reset_utils.py:455)、[基类reset仅清progress等](/home/abao/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py:590)、[reward编码](/home/abao/simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/obs_utils.py:362)。现有成功BC数据实际新回合初始reward也约1，不是只凭代码推测。

只替换同一批32个真实fresh-reset观测的reward字段：0→实测成功BC终止reward中位数100.14024353×.01=1.0014024353；其他161维不变。冻结网络CPU结果：

| 模型 | mean绝对动作变化 | max动作变化 | arm均值 | hand均值 | 估计条件动作std：前→后 |
| --- | ---: | ---: | ---: | ---: | ---: |
| native hand.1 50M | .019589 | .136699 | .006334 | .023807 | .198230→.195761 |
| 已成功BC control | .002683 | .007746 | .004142 | .002219 | .001984→.002153 |

std用同一组8次独立高斯噪声/state估计tanh后条件波动，不是时序重复噪声或controller滤波后的实际关节std。native的pre-tanh sigma绝对变化均值.034795；BC为.002087。native CPU原始动作与记录的GPU冻结动作最大差2.09e−6，支持这一小批样本的读取/前向对应正确。

**这只证明reward字段会影响动作，不能证明它造成失败，也不是修改reset的授权或修复结论。** 下一步可做第一帧reward-only seed的冻结rollout，并与默认0严格配对；必须标为diagnostic，保留真实reward与后续auto-reset不变。

## 复现本CPU核对

在`/home/abao/flashsac-robotics`运行：

```bash
/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python diagnostics/overnight_20260908/budget50m/audit_train_eval_mismatch.py
/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python diagnostics/overnight_20260908/budget50m/audit_reset_reward_signal.py
```

JSON保存配置diff、全部64回合事件、CPU加权断言、checkpoint/dataset/source SHA256与敏感性结果。本报告不解决1024/32 env、仿真精度、更新中策略对冻结策略差异；这些仍需下一组控制实验。
