# 冻结 500M policy：Actor 是否被熵项或动作边界抵消？

**本轮局部探针不支持“熵项把 Q 梯度整体抵消”。tanh 确实削弱部分动作梯度，但不是所有梯度都消失。** 这不是训练停滞的根因证明，也不代表 Q 的判断正确。

## 方法

- CPU float32；冻结 500M actor、critic、target、温度与 reward normalizer，无 optimizer step，无 GPU。
- 使用 `q_hardreset_baseline/reference_trace.npz` 中 32 个环境的第 0 / 60 / 120 步，共 96 个状态。Actor 读取前 140 维，Critic 读取后 162 维。
- 实际 `update_actor` 的目标是 `mean(alpha * log_pi - min(Q1,Q2))`，当前 BC=0。训练时 **Actor 用 batch statistics，Critic 用 running statistics**；探针按此模式计算，不把两者都设成推理模式冒充训练。
- 保留原模型不动，每次复制 Actor 后计算：Q 项参数梯度、熵项参数梯度及两者的和。固定基础种子 20260911，重复 8 次采样。按全参数、mean head、std head、backbone 分组，细节见 JSON。
- 当前参考轨迹只有 s0..s120，没有 s120 的后继。**没有伪造后继，也没有把这些数据称为真实 replay batch。** Actor 的 4,096 行 BN 上下文由 96 个查询状态加 4,000 个固定抽样参考状态构成，是局部轨迹上下文；只对查询状态的损失求导。
- 另用 Actor 推理模式的确定性动作，检查动作幅度、`dQ/da` 和 `dQ/dz=(1-a²)dQ/da`；这里 z 是 tanh 前动作。

## 结果

checkpoint 中 `alpha = 0.0000385937091778`。以下为 8 次采样均值，全 Actor 参数：

| 参考步骤 | 熵梯度范数 / Q 梯度范数 | 总梯度与 Q 项梯度的余弦 | 沿总负梯度的局部 Q 改善方向保留比例 |
| --- | ---: | ---: | ---: |
| 0 | 6.42% | 0.99835 | 96.90% |
| 60 | 1.98% | 0.99985 | 99.46% |
| 120 | 1.52% | 0.99992 | 99.83% |

最后一列是 `dot(gQ, gQ+gH) / ||gQ||²`，是当前参数空间的局部导数比值，**不是实测训练改善率或成功率**。熵项有部分相反方向，但本组数据中不足以压过 Q 项。

推理模式的确定性动作：

| 参考步骤 | 动作维度中 `abs(a)>=0.98` 的比例 | 经过 tanh 后 Q 梯度范数的平均保留比例 |
| --- | ---: | ---: |
| 0 | 20.6% | 35.5% |
| 60 | 28.9% | 50.0% |
| 120 | 27.4% | 59.5% |

说明部分维度接近边界且梯度受到衰减，但不能据此证明动作无效或学习被堵死；分母是各状态的 `||dQ/da||`，并非环境回报。

## 保护与边界

- 原 actor / critic / target / reward-normalizer 的内存 digest 前后一致；5 份源 checkpoint 文件 SHA256 也一致。一次性 Actor 副本的 BN 会按训练模式更新，随后丢弃；没有改原模型。
- 另检查了 768 个训练模式查询样本：双 Q 没有相等项，探针 `min` 与实际 `torch.minimum` 的动作梯度最大差值为 0。
- 这不是实际 replay 分布，也没有模拟 Adam 预条件、AMP、学习率和更新后的权重归一化。不能把这里的梯度大小直接解释为真实训练更新大小。
- 方向只针对**学出来的 Q**，不是实际后续回报。若 Q 排序本身有偏，Actor 有梯度并沿其优化仍可能没有好行为。

复现（不需要 Isaac / GPU）：

```bash
/home/abao/play2perfect/.venv_isaacsim/bin/python /home/abao/flashsac-robotics/diagnostics/str_q_followup_20260911/actor_gradient/check_actor_gradient.py
```

[完整结果](/home/abao/flashsac-robotics/diagnostics/str_q_followup_20260911/actor_gradient/result.json)
