# 完整 STR 100M：CPU checkpoint 状态审计

2026-09-09。只读取保存的参数、优化器、BN 和 reward normalizer；CPU threads=2。没有启动 GPU、仿真、rollout、评测或训练，也没有修改核心代码、环境、配置及旧数据。

范围：full 三个 seed 各20/40/60/80/100M，共15个 checkpoint；对照为旧简单任务三个 seed 的60M checkpoint。full Actor140、Critic162；旧简单任务 Actor162、Critic162。两者任务与信息量不同，不能当作单变量对照。

## 结论

没有发现 NaN/Inf、优化器停更、BN 方差非法或显著数值爆炸。保存参数确实持续变化。最具体的尺度差异是 full reward 分母受到历史最大回报约束，达到旧简单任务的1.44–2.07倍；这是奖励压缩线索，不是已证训练失败原因。没有 full 真实 observation/replay 数组，不能从权重 alone 判定 Q 排序错误、动作 std、熵或探索失败。

## 实际更新与奖励归一化

|最终 checkpoint|native Critic调用|Actor实际更新|Critic实际更新|alpha|奖励分母D|G运行标准差|
|---|---:|---:|---:|---:|---:|---:|
|full seed0 /100M|195128|97564|195007|1.3041e−6|365.192|34.748|
|full seed1 /100M|195128|97564|195002|8.1675e−7|459.654|35.332|
|full seed2 /100M|195128|97564|195007|7.7388e−7|550.111|34.972|
|simple seed0 /60M|116998|58488|116945|6.6689e−5|252.881|170.686|
|simple seed1 /60M|116998|58493|116936|2.6640e−4|250.893|153.694|
|simple seed2 /60M|116998|58494|116931|1.3138e−4|265.750|265.750|

full 三条正常化 count 均为100,003,840。native calls=(48,830−48)×4=195,128；Actor/温度机会数为97,564，温度实际更新也均97,564。Critic AMP 跳过121/126/121次，仅0.062–0.065%，不是大面积跳步。所有18个 checkpoint 的108个组件文件通过有限值检查并记录 SHA256。

代码的实际分母是 `D=max(sqrt(G_rms_var+1e-8), G_r_max/5)`。full 全部由历史最大绝对 G 支配，最终最大值为1825.96/2298.27/2750.55；该最大值只增不减。因此分母比同一训练的 G 标准差大10.5/13.0/15.7倍，可能由稀少的大回报长期维持，不能说是当前多数 transition 的尺度。

|full seed|20M D|40M D|60M D|80M D|100M D|
|---|---:|---:|---:|---:|---:|
|0|232.931|365.192|365.192|365.192|365.192|
|1|202.186|328.787|372.004|372.004|459.654|
|2|404.142|404.142|522.138|539.230|550.111|

最终 raw +300 经缩放变成0.821/0.653/0.545，旧简单任务为1.186/1.196/1.129。full 最终 alpha 反而比旧任务低约51/326/170倍，不能把现象直接解释成“温度过高”。尺度和任务分布同时改变，不能从这张表推出资产多样性造成分母增加或分母是唯一根因。

## BN、std head、Critic 参数

- full Actor 输入 BN 最小方差为1.49e−4/1.96e−4/2.45e−4，所有层无负方差；最大输入推理增益约46.1/34.6/38.9。没有 full BN 数值爆炸的证据。旧简单任务有常量输入通道，最大增益约423/458/430，不能把不同分布的 BN 数值直接比较成好坏。
- full `std_bias` 均值0.115/0.116/0.114；旧为0.104/0.106/0.124。数量级相近。它不是实际 log_std：还需 `std_w × hidden`，再经 tanh 映射到[-10,2]并取exp。没有实际观测不能断言策略方差太小/太大。
- full Actor 参数从20M到100M相对L2变化54–55%，Critic约60–61%；不是模型没更新。最终 Critic 与 target 参数相对gap约0.8–0.9%，没有 target 完全未跟随的迹象。
- 线性输出权重行范数约1，符合原生 unit normalization。Critic support均为101个atom、[-5,5]。这不能验证实际Q是否正确或目标投影是否裁剪。
- 已完成的输出矩阵检查也不支持直接宣称“Q头坍塌”：full原始atom行余弦较高，但减去softmax不敏感的公共logit方向后，stable rank为2.43–2.56，旧简单任务为1.98–2.26。其中心化权重范数较小（6.49–6.72 vs9.00–9.17）只是参数差异，不等于对真实状态/动作的Q梯度弱或Q不准。

## 限制与复核

full `preflight_gpu.json` 只有配置和冒烟统计，未发现保存的真实 observation/replay 数组。本审计没有输入伪 observation，没有反推全状态空间Q、真实策略熵、原子概率或return ranking，也不验证已知有问题的评测结果。若要判断 Q 学错，需要真实训练/rollout状态与严格对齐的目标或后续回报；本次按用户要求不新跑这些。

结果：[checkpoint_state_audit.json](/home/abao/flashsac-robotics/diagnostics/full_nodr_20260909_1004/diagnosis/checkpoint_state_audit.json)。脚本：[audit_checkpoint_state.py](/home/abao/flashsac-robotics/diagnostics/full_nodr_20260909_1004/diagnosis/audit_checkpoint_state.py)，`--self-test` 已通过。脚本默认仅输出JSON到stdout，所有tensor计算在CPU、2线程。
