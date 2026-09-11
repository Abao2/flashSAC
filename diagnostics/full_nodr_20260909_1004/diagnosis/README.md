# 完整 STR 100M 未学会：修复与诊断

2026-09-09。本次按用户要求：修已知错误，不启动 eval，不重训、不改原奖励或训练配置。

## 结论先说

**已定位学习停滞的阶段，尚未证明唯一原因。** 三个 seed 都完成100,003,840 transitions；训练主要增加了指尖靠近奖励，偶发越过抬升阈值的比例几乎不变，目标进步奖励不增长。因此不能把问题仅解释成“最后姿态太难”，抬升阶段已经停滞。

评测启动失败是独立的代码错误，发生在训练保存之后，不是这次训练学不起来的原因。

## 1. 已修复的代码错误

`termination.eval_success_tolerance` 的实际默认值是 `None`。本地 Isaac Lab 配置更新器按当前值类型校验，因此拒绝 float 覆盖，六次评测均在 rollout 前退出。

- 修改 [eval_str_full_nodr.py](/home/abao/flashsac-robotics/scripts/eval_str_full_nodr.py:20)：将起始容差和目标容差同时设为请求值；清除 `eval_success_tolerance` 覆盖为 `None`。上下限相等，课程不能再改变容差，不必修改公共导入器。
- 回归 [test_str_full_nodr_eval.py](/home/abao/flashsac-robotics/tests/test_str_full_nodr_eval.py:48)：使用安装版本的真实类型检查函数及任务默认值；验证 .075/.01、已有覆盖清理，并调用真实课程函数检查冻结。
- 新回归在旧代码上6项失败，复现同一错误；修复后，与 full配置/queue/adapter/timeout/n-step 测试一起 **34 passed**。
- 本次没有启动 Isaac 或独立评测，故这里不声称 GPU eval 已通过。

复测命令，在 `/home/abao/flashsac-robotics` 执行：

```bash
/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python -m pytest -q tests/test_str_full_nodr_eval.py tests/test_str_full_nodr.py tests/test_str_full_nodr_queue.py tests/test_external_isaaclab_adapter.py tests/test_n_step_discount.py
```

20份训练前源文件快照重新核对：只有上面的评测脚本发生预期变化；算法、环境和训练配置仍与该轮运行前一致。未覆盖任何 checkpoint 或原实验日志。

## 2. 卡在抬升阶段，而不是完全没有学习

下表来自 [full TB](/home/abao/flashsac-robotics/diagnostics/full_nodr_20260909_1004/runs)。阶段内等权平均日志窗口；窗口内按结束 episode 数加权。**阶段数不是按所有 episode 重新加权的精确总体率。**

| seed | transitions阶段 | reward | 指尖靠近奖励 | 曾触发抬升奖 | 目标进步奖励 | 每回合完成goal数 |
|---|---|---:|---:|---:|---:|---:|
|0|0–10M|24.17|11.27|10.48%|1.023|0.00042|
|0|80–100M|81.21|56.76|10.91%|1.104|0.00056|
|1|0–10M|21.47|8.39|10.38%|1.007|0.00036|
|1|80–100M|80.13|56.34|10.80%|1.066|0.00090|
|2|0–10M|23.83|13.66|10.00%|1.014|0.00017|
|2|80–100M|78.77|56.03|10.76%|1.062|0.00070|

“曾触发抬升奖”=窗口平均 `lift_bonus_rew / 300`。每episode首次超过生成时高度10cm记一次300分，跨goal不重复；**它不证明稳定抓取，物体弹起也可能触发**。所以这里不能宣称已有约11%的稳定抓握成功率。

成功简单版的同一指标明显不同：前10M抬升奖励约3，10–30M约206–211，末10M约294–299，对应事件比例约1%→69–70%→98–100%。其三个seed最终确定性成功62/64、1/64、64/64，仍有seed1不稳定；不得只挑成功seed当一般结论。[旧审计](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m/audit.md)

容差 `.075` 没降符合代码：平均完成goal数达到3才收紧。课程不下降是目标几乎没完成的结果，不能倒过来认定课程损坏。

## 3. 已确认的差异与可检验机制

### A. 机械臂目标增量确实缩小10倍

在未触发关节限位时，真实 [action pipeline](/home/abao/simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/action_utils.py:50) 化简为：

`q_target_next = q_target_prev + arm_moving_average * 1.5 * (1/60) * action`

- 成功简单版 arm=1：满幅单步目标增量0.025rad，目标速率上限1.5rad/s。
- full版 arm=.1：满幅单步目标增量0.0025rad，目标速率上限.15rad/s。
- 两版 hand均为.1，手是绝对目标EMA，不应套用同一个速度解释。

已在CPU直接调用真实函数核验，10倍关系通过断言。这是**目标命令**，不是测得的实际机械臂速度或宣称任务必然慢10倍。

这会改变动作探索对物理状态的影响及有用动作的时间跨度，值得单变量对照。两版FlashSAC及STR PPO/SAPG都是gamma=.99，不能说原STR用了更长折扣。仅作时间尺度说明：`.99^60=.547`，`.99^300=.049`；不是已测的真实回报或失败归因。

### B. 奖励公式相同，但抬升参考高度变了

[reset](/home/abao/simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/reset_utils.py:294) 把物体**生成时**高度记作 `_object_init_z`，不是落桌后的高度。full随机生成物体高于桌面；其下落后，连续项 `20*clamp(.05 + z-z_init,0,.5)` 可能先被截到0。简单版直接在桌面上方1mm开始，不存在同样的下落距离。

CPU用同一扁平橡皮擦举例（不是full实际rollout）：落桌中心z=.554966，full例子的生成z=.63。相对落桌位置升0/1/3cm时，简单版连续奖励为 .98/1.18/1.58，full例子为0/0/.0993；抬升一次性奖励要求从落桌位置上升约17.5cm，而简单版约10.1cm。

因此，“只是扩大物体种类，奖励完全一样容易学”不成立。但这是**原任务本来就有的设置**，不是已确认移植bug；本次不修改奖励或reset。

### C. 162D有状态信息的actor换成140D无记忆actor

去掉了palm/object速度、最近距离tracker、是否曾抬起、progress、successes及reward等22维；critic仍有162维。没有增加LSTM。

CPU真实reward函数验证：相同当前关键点距离.2m，历史最近距离分别.25m/.2m时，关键点进步奖励可为10/0。说明任务有历史依赖；不是“同一张当前画面就一定能知道阶段”。但这并不证明最优动作一定不同，**也不证明LSTM必需**。简单版FF成功不能直接排除full任务的信息问题。

### D. 多个因素一次恢复，因果没有被隔离

物体1→1200、固定→随机初态/位置与旋转目标、单goal→50goal链、actor输入和机械臂控制同时变化。这轮是完整分布试训，不是单因素A/B。100M比60M多，也不能证明复杂任务样本预算足够；不能机械地要求1200倍预算，网络可以跨物体共享学习。

上述CPU检查可复现：

```bash
/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python /home/abao/flashsac-robotics/diagnostics/full_nodr_20260909_1004/diagnosis/control_reward_probe.py
```

## 4. 已排查的数值/更新问题

[18个checkpoint的CPU审计](/home/abao/flashsac-robotics/diagnostics/full_nodr_20260909_1004/diagnosis/checkpoint_state_audit.json)：三个full种子的5个checkpoint，加三个简单版最终checkpoint。

- 每个full最终Q更新调用195,128次，Actor/温度实际更新97,564次；Critic实际更新195,007/195,002/195,007次。AMP只跳过约.06%，不是优化器停更；权重持续变化。
- 权重、优化器等检查均finite；未发现BN负方差或数值爆炸。正常数值不等于Q预测准确。
- 最终alpha约1.30e-6/8.17e-7/7.74e-7；训练熵接近配置目标−13.867。采样动作使用 `tanh(mean+std*noise)`，alpha不直接乘在采样噪声上，**alpha小≠不探索**。
- 全局更新比例没有减半/加倍：1024样本2次Q更新与2048样本4次Q更新相同；LR decay都固定19532，未随100M拉长；重复噪声仍按仿真step刷新。
- full归一化分母365/460/550，简单版约253/251/266，增大1.44/1.83/2.07倍；由历史最大回报分支主导。一次+300缩为.821/.653/.545。这是学习尺度的候选影响，不是数值无限爆炸，也没证明它是根因。

没有保存full的真实replay或代表性状态批，故本次不能测Q的动作排序正确性、TD目标溢出比例、分阶段动作std或actor梯度质量；没有用假观测冒充这些证据。

## 5. 下一步实验建议（本次没有启动）

优先从已成功的简单配方做单因素退回，每组保持任务、seed、预算、hand=.1和其他配置不变：

1. 只把arm从1变为.1，测试慢控制能否仍学出抬升。
2. 另开独立对照，只把actor162变为原140输入，测试信息缺失是否单独阻断抬升。

先看分阶段抬升事件比例/奖励是否随训练增长，不跑大规模独立eval。一个seed或短预算失败不能排除该配方；最终因果判断需要复现，且原native随机warmup动作空间未显式seed，不能保证仅凭同seed取得逐步一致的反事实轨迹。

本次只修真正报错，没有把“候选机制”当成已证bug，也没有为了出成功率改容易任务后冒称完整STR学会。
