# 50M hand.1：CPU 加载等价性检查

时间：2026-09-08T17:23:47.321915+00:00。

**结果：CPU 上原生全量加载与 FrozenPolicy 的 Actor-only 加载等价。未发现漏载 BN、train/eval 模式、α参与采样或 process_transition 原地修改输入导致动作不一致。此结论不等于已验证 CUDA Inductor 编译执行。**

## 比较了什么

同一份最终 checkpoint：

[step48830](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/models/arm1_hand01/seed0/step48830)

- 原生端：实际调用 agent.load，完整加载 actor、critic、target、temperature、normalizer、agent_state；连 optimizer、scheduler、GradScaler 都恢复，agent计数为97,466。
- 冻结端：实际调用现有 FrozenPolicy 的 flash 分支，只读取 actor.pt，去除 _orig_mod. 前缀，严格 load_state_dict，然后 .eval()/requires_grad_(False)。
- 1024条固定离线状态：先前的512专家状态＋512旧10M hand.1失败状态；另取32条测试。
- 原生端使用 torch.compile **backend=eager** 包装，保留真实 compiled checkpoint 的 _orig_mod 命名空间，并包装 get_mean_and_std；没有运行 CUDA/Inductor 优化内核。

创建网络时的标准归一化在加载前发生；加载后的参数和 BN buffer 与冻结端逐 tensor **完全相等**，本次推理没有再次做权重归一化。

## 检查结果

| 检查 | 结果 |
| --- | --- |
| 原生 vs Frozen，同一strided输入的 mean/std | 32和1024条均逐元素完全相同 |
| 原生 vs Frozen，同一contiguous输入的 mean/std | 32和1024条均逐元素完全相同 |
| 原生 vs Frozen，确定性actions | 完全相同，最大差0 |
| 原生模块.train() vs.eval()，但显式training=False | 完全相同，最大差0 |
| 原生随机采样 vs Frozen native随机采样 | 相同RNG及噪声缓存下连续12次actions完全相同 |
| 噪声缓存、重复计数、重复长度 | 两端每次都完全相同 |
| 所有Actor参数、BN buffer | 检查前后完全不变 |

这12次随机比较有一个额外意义：原生完整加载的α≈2.065e−5，而 Frozen 未加载的闲置温度网络仍是默认.01，**动作却完全相同**。熵系数α并不设置采集器噪声倍数；原生 sample_actions 的training=True固定传1，deterministic传0，Frozen native stochastic这里也传noise_multiplier=1。

## 观测布局确实有很小的浮点差异，但不是两个加载器之间的差异

完整非对称观测是324维；取前162维后，row stride为324。复制成contiguous162维后，row stride为162。

| 批量 | strided vs contiguous：mean最大差 | std最大差 | tanh(mean)动作最大差 |
| --- | ---: | ---: | ---: |
| 32 | 4.268e−5 | 3.099e−5 | 3.598e−5 |
| 1024 | 5.627e−5 | 3.123e−5 | 5.194e−5 |

两个加载器在相同布局下仍完全相同。**不能把小差异直接宣布为“不可能影响接触仿真”**，长轨迹可能放大数值扰动；但这次没有发现“Actor-only加载带来额外误差”。

## 为什么模块.train()在这里不改变采集动作？

[UnitBatchNorm](/home/abao/flashsac-robotics/flash_rl/agents/flashSAC/layer.py:43) 使用显式 F.batch_norm(..., training=training)，不是根据模块的self.training决定。Actor逐层传递显式flag；[采样函数](/home/abao/flashsac-robotics/flash_rl/agents/flashSAC/agent.py:235)始终指定training=False。

所检查Actor网络中没有dropout或依赖self.training的额外分支。源码检查与实际切换train/eval的CPU结果一致。

## process_transition 有没有把在线obs/reward弄坏？

在完整加载的、一次性CPU agent上实际调用process_transition，使用1024条独立测试数组，包含terminated与truncated样本：

- observation、next_observation、action、reward、terminated、truncated：调用前后都完全不变。
- 随后故意修改原始next_observation测试数组，已写入replay的值仍保持原样，证实数据有独立副本。
- normalize_rewards返回新tensor，采样出的原始reward没有被原地除掉。
- 只有这个一次性agent自己的reward统计发生预期更新；未调用任何网络optimizer。
- Actor参数和BN再次核对，完全不变。

代码也一致：[训练主循环](/home/abao/flashsac-robotics/train.py:146)先copy next_obs给replay、替换终止处final_obs，process_transition后再把下一次采样使用的next_obs换回reset后的观测；[TorchUniformBuffer](/home/abao/flashsac-robotics/flash_rl/buffers/torch_buffer.py:98)会复制传入数据；reward normalization使用新结果而不是原地操作。

## 这次排除不了什么

1. 原生CUDA Inductor、TF32、不同批量大小、运行时图缓存的数值/状态行为；本次backend=eager不是这些内核。
2. 在线训练与冻结评测的reset/startup/环境数量，以及实际输入状态分布是否相同。
3. 训练中episode内不断变化的policy与固定最终policy的区别。
4. CPU测得的细小布局误差是否会在长接触轨迹中被放大。

所以目前应把“加载器漏权重/漏BN”降为低优先级，继续做主流程准备的GPU同状态与启动协议对照；不要把CPU通过误写成“训练和评测全流程已证明一致”。

完整数字、输入来源、checkpoint哈希及限制：[loader_parity.json](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/loader_parity.json)。未更改checkpoint、核心代码或运行队列；未启动GPU仿真。
