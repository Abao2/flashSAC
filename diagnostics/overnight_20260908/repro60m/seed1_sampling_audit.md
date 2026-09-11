# Seed1 60M：确定性与随机采样的差别

使用同一冻结checkpoint、同一任务配置的完整已有回合；不是重新训练。独立几何核验成功，未把抬高/靠近palm代理值当成功。

| 采样 | 真实成功/请求回合 | 成功中连续10帧near | 失败near次数分布 |
| --- | ---: | ---: | --- |
| deterministic | 1/64 | 0/1 | 7帧:5回合, 8帧:55回合, 9帧:3回合 |
| stochastic | 88/128 | 16/88 | 4帧:1回合, 6帧:3回合, 7帧:6回合, 8帧:12回合, 9帧:18回合 |

## 失败究竟离目标多远

以下只看失败回合；误差为固定reward尺寸的4个对应角点最大距离，阈值3cm，而不是只看物体中心。所有失败均600步时间截断；没有被错误计成成功。

| 采样 | 最佳误差 min/median/max(cm) | 最佳时刻 median(s) | 终点误差 min/median/max(cm) | 最后1秒误差均值(cm) |
| --- | --- | ---: | --- | ---: |
| deterministic | 1.316/1.346/1.425 | 0.467 | 6.300/7.910/13.705 | 8.131 |
| stochastic | 1.084/1.382/2.373 | 0.467 | 5.806/8.123/11.791 | 8.522 |

**不是‘还差一点才碰到阈值’：63个确定性失败全部进入了3cm范围，且最佳误差约1.3–1.4cm；问题是没有累计够10帧。** 全部在0.450s首次进入，0.467s达到最佳；最后near帧中位时刻0.567s，最晚0.900s，之后没有再达到阈值，最终10s超时。多数是连续8帧后离开。

这些失败仍获得平均796.8分goal bonus（每near帧100分）与300分lift bonus，所以高reward不等于任务完成。末1秒平均线速度0.0134m/s、角速度12.40deg/s，停留在偏离目标的位置，而不是仍在高速寻找目标。这个描述只来自运动记录，不证明其内部优化原因。

随机成功88回合中，72个在离开near区后再次进入，才累计到10帧；其余连续10帧完成。首次near中位时刻仍0.450s，成功中位时刻0.700s。随机采样改变后续闭环轨迹，能补足缺的near帧；不能仅凭结果断言具体是哪个关节噪声、均值策略多峰或Q函数原因。

![同一起始回合的前1.5秒](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m/seed1_sampling_trace.png)

图按最低初始episode ID选一个确定性失败/随机成功的实例，不是总体成功率或典型性证明。随机曲线在真实成功处结束，不画虚构的后续保持。

## 成功与失败动作分别统计

| 采样/结果 | 回合数 | 持续时间 median(s) | 终点中心误差均值(cm) | 终点旋转误差均值(deg) | 终点FD线速度均值(m/s) | 终点FD角速度均值(deg/s) | 终点高度均值(cm) | 终点物体–palm距离均值(cm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deterministic/successful | 1 | 0.700 | 0.838 | 25.48 | 0.120 | 69.6 | 15.02 | 9.05 |
| deterministic/failed | 63 | 10.000 | 6.011 | 17.89 | 0.014 | 11.3 | 10.60 | 9.16 |
| stochastic/successful | 88 | 0.700 | 1.168 | 22.39 | 0.125 | 68.4 | 14.86 | 8.53 |
| stochastic/failed | 40 | 10.000 | 6.857 | 19.52 | 0.075 | 38.9 | 11.27 | 8.34 |

抬升/近掌部并非力闭合或稳定抓握的证据。JSON包含失败与成功的完整near帧序号、最佳误差/时刻、末秒窗口、连续near、运动和reward分量。

## 可得出与不能得出的结论

随机策略与tanh(mean)策略在这个checkpoint上的任务表现确实不同。SAC优化的是带采样的策略；将其部署成确定性均值动作不是等价操作，也没有成功率不降低的保证。但本结果本身不证明策略多峰、‘平均了两种动作’，或特定探索机制。

最初32个回合step0观测两组最大绝对差为0；这批初始回合确定性成功1/32、随机成功19/32。随后reset时刻不同，不把64与128个回合强行配成同一批逐轨迹对照。两组配置完全相同，actor digest相同且评测前后均未改变；源actor/data SHA256已保存。

- Same checkpoint/config/rollout seed and first32 initial states checked; later episodes reset at different times, so not trajectory-by-trajectory paired trials.
- Stochastic SAC policy is the optimized behavior; deterministic tanh(mean) is an evaluation choice and has no equal-success guarantee. A stochastic advantage alone proves neither multimodality nor mean-action averaging failure.
- Native stochastic noise has its original temporal repetition. This compares that whole sampling rule with deterministic actions, not independent per-step Gaussian noise.
- Near criterion is cumulative10, not consecutive10. Success stops recording immediately; finite-difference motion/proximity are not post-goal stability or verified contact.
- Final speed is per-transition backward finite difference at 16.667ms, never across reset; quaternion normalization and sign-invariant shortest rotation used.
- Success and failure motion cohorts remain separate. Last-up-to1s windows for short successes are shorter than one second.
- One seed1-trained policy and one rollout RNG seed on fixed object/start/goal. Sampling result does not prove generalization or a training mechanism.
- Near-dependent metrics are null if never near; aggregate n counts observations available, not substituted zeros.

## 背景，不混用checkpoint

- seed1_50M_det: 1/64（只引用原summary，未对这些回合重做运动核验）。
- seed1_50M_stoch: 40/128（只引用原summary，未对这些回合重做运动核验）。
- seed0_60M_det: 62/64（只引用原summary，未对这些回合重做运动核验）。
- seed0_60M_stoch: 127/128（只引用原summary，未对这些回合重做运动核验）。

CPU复现：`/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python /home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m/audit_seed1_sampling.py`。无GPU、无核心/配置/主报告修改。
