# 技能保留失败：熵项、reward尺度与categorical critic的证据

本报告没有新训练。它区分已测现象、固定数据上的诊断与下一步假设。**下面的轨迹soft sum不是已校准SAC Q，也不是策略优劣/最优性的证明。**

## 1. 更新后发生了什么

立即更新分支：5次actor实际优化后，det目标成功64/64，stochastic95/128；50次actor优化后，两者均0成功。先只训critic分支：network calls10/100/1000/2000期间actor实际更新均为0，det64/64、stochastic128/128一直保留；恢复更新后，到999次实际actor优化的首个后续采样点，成功率已为0。

warmup分支没有在恢复后的5或50次actor更新处评测，因此**不能据图声称warmup延迟了遗忘到999次**：其丢失时刻只定位在0～999次之间。零actor更新的四个点重叠，不应画成四个递增actor步数。

![Actual actor optimizer steps versus task success](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/warmstart/retention_actor_steps.png)

## 2. 成功轨迹的reward和熵项：逐步折扣，不是63×平均熵

另取selected BC自己的128条成功rollout，平均62.375步。冻结actor、normalizer与alpha，逐个状态重新计算条件微分熵：解析Gaussian熵 + 每状态8次MC的tanh log-Jacobian。

原生一步target为 `r/D + γ(1−terminated)[Z_next − α logπ(a_next|s_next)]`。因此我们定义固定已记录轨迹的诊断量：

`S0 = Σ(t=0..T−1) γ^t r_t/D + α Σ(k=1..T−1) γ^k Hπ(a|s_k)`，γ=.99。

不加起点s0的熵，不在真实goal terminal后补尾项；所有被选回合均以真实终止结束，没有把timeout强行截成terminal。熵随每个状态变化，且每个回合长度分别计算。

平均折扣raw task reward为826.422266；平均折扣future-entropy sum为-5661.204086 nats。下面每一列使用一个固定checkpoint的reward除数，不混用训练中的动态除数。

| 固定alpha | S0：D=immediate update0 | S0：D=warmup update2000 |
| ---: | ---: | ---: |
| 0 | 3.463211 | 2.933013 |
| 1e-06 | 3.457550 | 2.927351 |
| 0.0001 | 2.897091 | 2.366892 |
| 0.001 | -2.197993 | -2.728191 |
| 0.01 | -53.148829 | -53.679028 |

两个D分别为238.628882、281.765666。alpha=0行只是相同轨迹的normalized task reward，不是另一条已训练策略。

重要边界：行为rollout使用缓存/重复噪声，而这里Hπ是网络给定状态的单步边际条件熵，不是给定完整噪声历史的真实行为条件密度。固定记录的state occupancy也不等于重新采样整条独立动作策略。这是**贴近原生actor边际定义的冻结轨迹surrogate**，不把它称为精确SAC Q/on-policy MC校准。

这个尺度检查提示：在该冻结高密度BC策略下，alpha=.01的surrogate远低于critic支持下限−5；alpha减小可以把它移回当前支持范围。但这不证明存在相应的最终SAC最优策略，也不保证低alpha学习成功。其他策略可以改变均值/方差、成功率、时长和状态分布。

## 3. 实际critic报告：并非只靠上面的surrogate猜测

warmup update2000时actor还没改，仍能成功，alpha=.01。下面是CPU从该checkpoint重新计算的诊断；不是历史训练replay的clipping频率。

| 查询数据/phase | normalized即时reward均值 | Q(recorded action) | 最低5atoms质量 | 未投影target均值 | clamp后target均值 | 下界外概率质量 | αH_next估计 | Q/熵梯度范数比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| expert/reset | 0.003324 | -4.990388 | 99.683% | -6.380522 | -4.988181 | 99.190% | -1.476395 | 0.0001338 |
| expert/far_from_object | 0.006255 | -4.987484 | 99.612% | -6.322569 | -4.977120 | 98.755% | -1.438549 | 0.0010338 |
| native/reset | 0.003336 | -4.998959 | 99.959% | -6.326910 | -4.999461 | 99.961% | -1.395490 | 0.0001859 |
| native/far_from_object | 0.006233 | -4.999647 | 99.985% | -6.111825 | -4.998918 | 99.941% | -1.181772 | 0.0125775 |

例如native/reset：即时reward/D约.0033，而下一状态熵项约−1.3955；γ再乘下一状态的Q与熵项。当前Q已靠近−5，target均值约−6.327，被截回约−4.999。这个报告直接测到概率质量挤在下界，不能用“成功reward很高”推断soft Q也应高。

梯度是在phase-balanced公共current/next batch上、训练actor BN+推理critic BN、FP32无AMP/优化器/投影下测的。范数比极小显示该诊断batch的actor梯度主要来自熵项；不代表每个真实训练batch都同比例，也不等于Adam后的参数位移比例。

专家数据可能有状态分布差异；native数据来自仍成功的冻结BC，与warmup rollout更接近。两者reset/far方向一致，增加了这个机制的可信度，但都不是训练replay抽样。

## 4. 目前最有根据的假设与如何反证

- 已确认：前馈actor能完成这个固定任务；仅冻结actor/temperature时技能保留；原生更新后技能丢失。不能再把这例失败简单归成网络无法表达或必须LSTM。
- 有证据的机制假设：成功但高密度/低微分熵的BC初始化，与当前熵权重、normalized reward尺度和[-5,5] categorical支持范围组合后，使soft target大量落在下界以下；Q对动作的区分梯度很小，actor更新受熵梯度强烈影响。
- 下一项配对诊断应从同一BC初始化开始，仅改初始alpha，保留native自适应温度、任务和网络；记录实际alpha轨迹、早期actor更新次数、det/stoch成功、Q边界质量及梯度比。降低初始alpha不是“永远固定低熵权重”，因为温度仍会自适应。
- 如果低alpha保住技能并减少下界饱和，支持该组合机制，但仍需区分actor直接熵梯度与critic soft target的作用；若低alpha仍同样遗忘，就需继续查Q学习、BN模式、optimizer/projection及采样分布，而不能追加未经验证的故事。
- 不据此宣称SAC不能解STR；不据此把entropy/reward尺度选择说成唯一原因；不据此要求修改环境reward。

JSON保留完整source SHA256、逐回合长度与折扣项、MC标准误、critic来源和图中每一个点的真实optimizer计数。
