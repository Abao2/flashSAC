# 本轮 Q 动作诊断：可支持的结论与边界

目标：冻结500M策略附近，**Q偏好的一次动作改变，是否带来更好的实际后续表现**。本轮不重新训练，不把有限rollout当成历史训练Q的精确MC。

正式设置：32环境、prefix=0/60/120、continuation seeds=30000/30060/30120、七个分支、最多600步；每次hard reset并核对物理参数、状态和RNG。容差固定 `0.02905653603374958`。正式动作结论以 `action_contrasts/summary.json` 为准，本文不预先填入未完成结果。

## 已有证据确实缩小了范围

| 诊断 | 能支持／降低优先级的解释 | 不能排除 |
|---|---|---|
| 原生Lab vs wrapper | 同动作磁带59字段逐元素相同，覆盖终止、超时和goal命中；这批轨迹不是wrapper传错动作/观测/reward/reset | 共享底层任务的问题、legacy Gym差异、未测DR和状态 |
| 最后10M replay | 4,183,638行已有≥1个goal，3,092,718行≥2，462,674行≥10；不是完全没有后续目标经验 | 行占比不是独立回合成功率；不能证明抓稳或每种物体都充分覆盖 |
| 实际控制链 | 独立控制公式、任务target、PhysX target差值0，限位与控制时间正确 | target不等于即时位置；无接触/力矩证据，不能判断PD、力矩饱和或抓取质量 |
| Actor局部梯度 | 已测上下文中熵项未整体压过Q项，tanh虽削弱部分梯度但并非全部消失 | 不是全部replay或实际Adam更新；有梯度不保证Q正确 |
| 真实replay BN | 三份2048-transition batch中，训练/执行BN均值动作平均绝对差约0.036–0.041，arm约0.063–0.065 | 尚不能认定BN是根因，也不能因平均差小排除关键状态影响 |

证据：[接口](../str_fourway_audit_20260911/interface/README.md)、[replay](../str_fourway_audit_20260911/replay/500M_coverage.json)、[梯度](actor_gradient/README.md)、[真实replay BN](actor_gradient/replay_batchnorm_notes.md)。

## 配对和三seed的统计层级

- prefix0/60/120参考存活环境数为32/28/21；后两个prefix针对存活到该阶段的子集。状态/RNG/物理参数检查失败的行不可解释，不放宽阈值让它们入组。
- 同seed的 `mean_repeat` 只检查模拟重复性，不是独立噪声重复。已完成600步长基线：prefix60/120的记录指标重复差为0；prefix0 raw折扣回报配对均差约−1.84、单环境最大绝对差58.78、goal最大差2。hard reset不是永久逐位确定性保证。[长基线](long_baseline/summary.json)
- 每个prefix先计算同env/同seed的branch−mean；每个env再跨三个noise seeds平均，报告环境间分布及每seed方向稳定性。对照噪声用同prefix/seed的mean-repeat，不拿不匹配组替代。
- 同env的不同prefix和seed是重复测量，不是九个独立初态。若给区间，至少按env成组保留重复测量；全局RNG仍可能产生跨env依赖，区间只能作探索性描述。三noise seeds不是三training seeds，不能支持跨训练种子稳定性。

## 相同Gaussian不保证后续随机目标完全相同

策略采样使用独立Generator，因此各分支可共享标准化Gaussian噪声；目标采样/reset使用环境全局RNG。动作改变goal命中或reset时间后，随机数消耗次序会分叉；并行环境还可能互相影响随机数的分配。

这不否定起点相同、动作不同的干预证据，但长时回报差也包含后续环境随机分叉。不能声称“之后每个目标完全一样，只差第一步机械运动”。跨seed稳定、超过对应重复性误差、并伴随可解释任务行为变化的差异，仍能支持**局部动作偏好与实际表现一致或失配**。

## n=3、Gaussian与Zeta：不能混称精确Q

实际replay仅累计原始reward；在bootstrap位置才减 `alpha*log_pi`。投影前target结构为：

`sum(gamma**i * reward_i / D) + gamma**k * (1-terminated) * (minQ_target - alpha*log_pi)`。

代码：[replay](/home/abao/flashsac-robotics/flash_rl/buffers/torch_buffer.py:126)、[target](/home/abao/flashsac-robotics/flash_rl/agents/flashSAC/update.py:202)。

- `discounted_raw_return`：真实环境reward的有限折扣和，回答任务收益。
- `finite_soft_return`：每个后续时刻均扣熵，是dense参考，不等于n=3 target。
- `finite_nstep_soft_return`：仅t=3,6,…扣熵，更贴近n-step结构，仍不是历史Q的精确MC。
- 两种soft量均不减被固定第一步动作的熵，符合本轮Q比较口径；不要为了模拟Actor loss随意改其含义。

历史replay中间动作来自随训练变化、带Zeta重复噪声的行为策略；本轮是冻结策略、每步独立Gaussian。categorical投影、BN上下文与冻结reward尺度也保留差异。**这是测量口径/原算法的区别，不是适配bug。** 本轮不声称量化off-policy n3偏差，也不因此追加大实验。

实际500M：`alpha=3.859370917780325e-5`，`D=1464.18916015625`。Q差1e-4约对应0.1464原始折扣reward尺度，熵需单列；不能直接与几千分原始回报相减。

## 终止与未知尾项

`terminated`按任务定义结束，后续价值为0，失败与完成50-goal应分开；`truncated`只是时间上限，训练需要bootstrap；`horizon_censored`是在探针预算耗尽时仍存活，不是失败或成功证明。

`gamma**600=0.002405`，仅网络±5表示范围给出的尾项尺度就约0.012，大于某些单动作Q差；这不是实际回报/熵的严格误差上界。超时更早时尾项更重要。保留所有配对行及截断类别；仅双方真终止子集可以辅助观察，但不能成为唯一主分析，因为动作会影响终止，筛选会引入选择偏差。

## 动作含义与最终判读

`zero`不是不动：arm零动作保持累积target，hand零动作指向关节范围中点。Q±是均值动作附近的一次有限扰动，clip后实际扰动可小于epsilon，要看实际动作与实际Q差；epsilon=.2的arm扰动仅约0.005rad目标增量量级。

Critic排名的 `training=False` 与实际Actor更新查询Critic的模式一致。但训练Actor使用大batch的 `training=True` 随机动作并优化Q与熵组合，因此Q±不是实际Actor参数更新的正负方向。[更新模式](/home/abao/flashsac-robotics/flash_rl/agents/flashSAC/update.py:106)

- **Q偏好动作跨seed反复改善收益/goal**：支持该状态区域的局部Q方向有用，降低“Q在所有相关状态都完全不辨动作”的优先级。
- **Q偏好动作跨seed反复更差，且不由重复性、尾项或仅熵口径解释**：支持局部价值偏好与任务效果失配，是可追踪候选瓶颈，不外推整个replay或全部训练阶段。
- **方向随seed改变或与对照噪声同量级**：该局部单步探针证据不足，不宣布Q好或坏。
- **称为训练根因的条件**：进一步把局部问题与训练中频繁出现的状态、Actor实际学习方向及行为损失连接起来。这是归因条件，不是本轮自动新增实验清单。

本文仅新增解释文档；未修改生产代码、参数、环境或checkpoint，未启动GPU工作。

