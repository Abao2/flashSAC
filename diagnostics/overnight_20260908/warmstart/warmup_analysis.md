# Critic warmup2000：保护住已有技能了吗？

审计时间：2026-09-08T16:05:05.887048+00:00。CPU 只读；状态来自已完成的本地运行，不启动 GPU、不改核心代码或运行队列。

## 结论

**这一次没有保护住“解冻后的”技能。冻结时始终成功，解冻后仍失效；但不能据此得出“先训练 Q 没用”。本次 Q 预热期间，原生低熵 soft-return 目标大量落到 categorical support 下界 −5 之外，训出来的 Q 已接近下界饱和，不是一个经良好校准的成功策略 Q。**

| 组别/检查点 | 已发生 Actor 优化 | 确定性成功 | Native stochastic 成功 |
| --- | ---: | ---: | ---: |
| immediate / update2000 | 1000 | 0/64 | 0/128 |
| warmup2000 / update2000 | 0 | 64/64 | 128/128 |
| warmup2000 / update4000 | 已解冻约1000次机会 | 0/64 | 0/128 |
| immediate / final | 9668次实际优化 | 0/64 | 0/128 |
| warmup2000 / final | 8661次实际优化 | 0/64 | 0/128 |

两组总 critic 调用均为 19,338。Actor 尝试次数分别 9,669 和 8,669，实际 optimizer step 因 AMP skip 略少，因此表中 final 使用真实 optimizer state。两组仍然不是“相同 Actor 更新次数”的实验。

## 1. 冻结确实生效

warmup_audit.status=complete，冻结边界为第 2,000 次 critic 调用。独立加载 initial 与 update2000 的 actor.pt/temperature.pt：

- 每一个参数和 running BN tensor 完全相等；
- Actor、温度的 optimizer/scheduler audit 完全不变；
- update10、100、1000、2000 都是 deterministic64/64 和 stochastic128/128；
- 相同 512 个专家状态上，update2000 动作 RMSE=0，H 与初始相同，为 −138.29 nats。

所以，冻结期保持成功并不是“代码偷偷更新了 Actor”、也不是评测漏跑。

## 2. 相同状态的 critic 对照

使用 [early_update_audit.json](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/warmstart/early_update_audit.json) 中完全相同的 512 个专家状态、同一动作采样 RNG。目标使用共同 128-row chunks，target critic 克隆的 training BN；Actor/Q 推理使用各自 running BN。下面的 clipping 是 **501 条非终止 transition 上被截到 support 外的概率质量均值**，不是“失败率”，也不是训练 replay 的统计。

| 模型 | 实际 Actor steps | α | reward 分母 | 真 tanh H | Q(初始BC平均动作) | 下界截断概率质量 | 上界截断概率质量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| immediate / update2000 | 1000 | 0.007573 | 238.629 | 8.81 | 1.5074 | 0.00% | 4.81% |
| immediate / final | 9668 | 0.001120 | 238.629 | 10.22 | 4.3177 | 0.00% | 1.58% |
| warmup2000 / update2000 | 0 | 0.010000 | 281.766 | -138.29 | -4.9939 | 83.35% | 0.00% |
| warmup2000 / final | 8661 | 0.001326 | 268.955 | -1.48 | 3.8235 | 0.00% | 1.66% |

主要现象：warmup2000 的 Q(BC动作)≈−4.994，已挤到 −5 下限；同一非终止专家样本 **83.35%** 的目标概率质量还要继续落到 −5 以下。reset 子集的下界截断质量是 **99.47%**，Q≈−4.99895。

同一 reset 状态，Q 对初始 BC 平均动作相对 zero action 的“优势”约 **−.00991**，32 个均匀动作的 Q 标准差约 **.00200**。这是当前 critic 的估值，不是真实收益排序；它没有在这些状态上清楚表达“这套成功动作值得保持”。

### 自己实际采到的成功 rollout 也有同样问题

不是只在官方专家/OOD 状态上看到：

- warmup update2000 自己的 native128/128 成功轨迹；
- reset/far/near/off-table 等常见阶段的下界目标截断质量 **99.66%–99.96%**；
- far/near/off-table 的 Q 大多约 −4.999。

这些既有 critic 报告采用 phase-balanced 抽样，和上表均匀固定512不是同一加权口径；不能混用百分比。lifted_proxy 仍只是高度代理标签，不等于成功或任务正式 lift 标志。

## 3. 为什么成功了，soft Q 却可能很负？

原生 critic 目标是：

y = r_normalized + γ × (1−terminated) × [Q_next − α logπ_next]。

连续动作的微分熵允许为负。当前成功 BC 很集中：

- α=.01；
- 固定专家 H≈−138；常见 next-state 熵项 αH≈−1.2 至 −1.5/步；
- warmup reward 分母≈281.77，普通 reset 的 raw reward≈.938，即 r_normalized≈.00333；
- critic 只能表达 [−5,+5]。

用当前 reset 的均值做一项说明：若下一步 Q 已在 −5，熵项约 −1.4008，那么目标约为

.00333 + .99 × (−5−1.4008) = **−6.3335**，

必然被截回 −5。**“真实任务成功”不等于“这个熵权重下的 soft objective 高”，更不等于 support 足以容纳该 soft return。** 这不是 NaN；是有限但大量被投影的目标。本审计没有给出真实无限期 Q 或 MC 校准证明。

### 成功终止与 lift bonus 不能混为一谈

固定512里有 11 条成功终止 transition：

- terminated=True；目标只有当前 normalized reward；
- raw reward 平均≈100.084，warmup normalized≈.35520；
- bootstrap 和 next-state entropy 全部被 mask；
- 这11条上下界 clipping 均为0。

另有9条 **非终止** reward>200 transition，raw reward平均≈301.15，不能称为“终止成功奖金”。这些仍 bootstrap，warmup 下界 clipping≈84.25%。

JSON 的 next_entropy_contribution 字段保留未 mask 的诊断数值；**terminated 行即便显示负数，也没有进入目标**。目标实现乘了 (1−terminated)。

## 4. 预热后的 Q 梯度，反而更弱了

复用原审计的完全相同512 expert obs+next_obs、3次采样重复，测 Actor 克隆的原生 training-BN 损失梯度：

| 模型 | ∥∇(−Q)∥ | ∥∇(α logπ)∥ | Q/熵范数比 | 余弦 |
| --- | ---: | ---: | ---: | ---: |
| immediate / update2000 | 1.04874 | 0.89389 | 1.17335 | -0.190 |
| immediate / final | 0.85414 | 0.08000 | 10.67910 | 0.060 |
| warmup2000 / update2000 | 0.01114 | 3.56463 | 0.00313 | 0.061 |
| warmup2000 / final | 1.36096 | 0.20459 | 6.65086 | -0.414 |

初始随机 Q 的比值约 .104；warmup2000 后降为 **.00313**，即熵项范数约为 Q 项 **320倍**。这与 Q 下界饱和相符，解释了为什么这轮预热没有形成有力的行为保留信号。

这里只是固定专家状态、原始参数梯度，不含真实训练 replay 批次、Adam、AMP、权重归一化的完整历史。因此不能把这个比例直接当作实际参数位移贡献，也不能说“全部是熵一项造成”。

final 时两组在这批专家状态上的 Q/熵比已大于1，但策略仍失败；该比值不是“训练健康阈值”。

## 5. 最小下一步：降低熵强度，而不是再盲目延长预热

只在已有 warmup2000 checkpoint 上把目标公式里的 α 缩小、不改变 target 概率，做了一个 **公式级反事实**：

| α倍率 | 对应α | 固定501非终止专家样本的下界截断质量 |
| --- | ---: | ---: |
| 1 | .01 | 83.35% |
| .1 | .001 | 76.69% |
| .01 | .0001 | 0% |

这不是重新训练后的结果，但说明 .0001 是有量化依据的下一档：可先验证“同样的好策略数据是否终于能拟合出不贴底的 Q”。

最小合法 A/B：**同一个初始 BC/随机 Q，只改 α初值 .01 vs .0001，其余原生训练流程不变**，仍使用短预算和早期解冻快照。主流程准备扩展为 .01/.0001/.000001 × immediate/warmup2000 六臂，便于看与预热的交互；本分析没有改或启动该队列。

注意两个真实代码语义：

1. load_optimizer=false 仍会加载 temperature.pt。只传 agent.temp_initial_value 的训练 override **不会覆盖 ckpt 里的α**。必须单独克隆初始 checkpoint，只改原生 log_temp，并校验其余所有组件一致。
2. agent 会从 temp_target_sigma 重算 target entropy。只改 temp_target_entropy 字段无效；只改 target sigma，也不能降低第一步已经由α=.01加权的 Actor 熵梯度。

α变化同时作用于 Actor loss、critic soft target、温度自适应轨迹。这个 A/B **不单独分离**三条影响路径。

α=1e−6 的数值检查：原生 FP32 log_temp/exp 有限；在初始专家熵下，温度梯度≈−1.244e−4，是 Adam eps=1e−8 的12,442倍；CPU单步更新后α≈1.00030024e−6。不能承诺不同组后续自适应完全同速，接近熵目标时 epsilon/历史动量仍可能影响相对步长。

完整数据：[warmup_analysis.json](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/warmstart/warmup_analysis.json)。本报告描述的是低熵 BC warmstart 的有限对照，不能直接推广成“FlashSAC 从零训练所有 STR 任务的唯一根因”。
