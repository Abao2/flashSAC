# 真实 replay：优化时与执行时的 Actor 是否不同？

**相同观测下，两种 BN 模式的均值动作存在可测差异；本测试尚不能认定这是训练停滞的原因。**

## 实测方法

- 冻结 500M checkpoint，从其最后保存的 10M 条 replay 中固定种子抽取 **3 份真实 2,048 条 transition**。
- 每份将 current / 已存 n-step next 的 Actor 观测拼成 **4,096 行**，与实际 `update_actor` 的输入组织一致；没有加入局部参考 query，也没有伪造后继。
- 原 Actor 使用 `training=False` 的 running statistics；复制 Actor 使用 `training=True` 的 batch statistics。比较两者的 `tanh(mean)`，不采样动作。
- 用同一个冻结 Critic 的推理模式分别评价两套动作。以下只报告 current 的 2,048 行；next 辅助结果在 JSON。

## 三份批次结果

| 固定种子 | 全动作平均绝对差 | 机械臂平均绝对差 | 动作分量差 >0.1 的比例 | 同 Critic 的平均 Q 差（训练 BN − 执行 BN） |
| --- | ---: | ---: | ---: | ---: |
| 20260911 | 0.03645 | 0.06453 | 8.21% | +0.000271 |
| 20260912 | 0.04105 | 0.06295 | 8.78% | +0.000146 |
| 20260913 | 0.03971 | 0.06334 | 9.27% | +0.000119 |

动作范围为 `[-1,1]`。超过 0.5 的差异约占 **0.79%–0.84% 的动作分量**，最大差 **1.45–1.49**；两种模式的 `abs(action)>=0.98` 饱和率都约 **15%–16%**。所以均值差异不大并不代表每个关键状态都一致。

Q 差值平均较小，但单状态尾部约为 **−0.034 到 +0.049**。这里的 Q 是归一化 reward 下学出的价值，不是原始 STR 分数，更不是实测后续回报。没有执行两套动作的物理对照，不能据此判定哪套更好。

## 保护与局限

- 使用现有覆盖审计相同的 `torch.load(weights_only=True, mmap=True, map_location='cpu')`，只索引选中行；没有整份复制或扫描约 24GB replay。峰值 RSS 约 **1.35GiB**。
- 原 actor / critic / target / 温度 / reward normalizer 的状态前后一致，5 个 checkpoint 文件 SHA256 不变，replay 文件大小、修改时间和 inode 不变。
- CPU float32，无 GPU、无 optimizer step、无重新训练。Actor 的 BN 更新仅发生在随后丢弃的副本上。
- 数据确实来自最终保存的 replay，但仅为三份批次，不代表所有历史训练阶段。比较的是分布均值动作，不是完整随机 SAC 目标。
- **它没有证明 BN 是根因，也没有证明 BN 无关。** 不应仅凭平均 Q 差小排除少数关键状态的影响。

复现：

```bash
/home/abao/play2perfect/.venv_isaacsim/bin/python /home/abao/flashsac-robotics/diagnostics/str_q_followup_20260911/actor_gradient/check_replay_batchnorm.py
```

[完整结果](/home/abao/flashsac-robotics/diagnostics/str_q_followup_20260911/actor_gradient/replay_batchnorm_result.json)
