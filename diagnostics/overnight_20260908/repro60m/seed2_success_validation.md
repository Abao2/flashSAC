# Seed2 continuous60M：独立成功核验通过

纯 CPU 验证，不开新仿真、不更新模型。完整数据、配置和 SHA256 在同目录 `seed2_success_validation.json`。

| Seed2 冻结评测 | 独立成功 | timeout | 连续10个 near 样本 | 平均完成步数 |
| --- | ---: | ---: | ---: | ---: |
| 确定性 | 64/64 = 100% | 0 | 64/64 | 26.02 |
| 原生随机/global-zeta | 128/128 = 100% | 0 | 90/128 | 27.41 |

## 成功是真的，但不是稳定保持证明

从记录的自动重置前 `next_obs` 恢复物体中心、xyzw 四元数，用独立 NumPy 旋转公式重建四个 **reward 固定关键点**。不是直接取 observation 中实际物体角点的误差。

判据保持原环境：固定尺寸 `[0.141, 0.03025, 0.0271] m`，keypoint scale=1.5，最大关键点误差 ≤30 mm，回合内**累计10帧**。每near帧+100，成功总goal bonus=1000；物体中心首次抬高超过10 cm另给一次lift bonus=300。

逐回合检查的结果：

- 192个回合、5,173 transitions；所有回合恰在首次累计第10帧终止，全部 `terminated=True, truncated=False`。
- 所有回合独立重建的goal bonus=1000、一次lift bonus=300，均与原始reward components一致；raw reward累计与episode return一致。
- 每回合只有末步done；step连续；回合内 `next_obs[t] == obs[t+1]` 最大误差为0。成功计数的log编码与独立计数一致。
- 确定性64回合全部连续10个near样本；随机38回合只有累计10个、没有连续10个。10个样本的首末时间跨度约0.15秒，不能当作长时间保持。
- 所有六个checkpoint文件均有限；normalizer计数60,002,304、native update counter=116998。actor文件与评测前后digest完全一致。CPU严格加载checkpoint重算确定性动作，最大绝对误差1.40e-4。

## 达标时仍在运动

以下是物体相邻记录pose的有限差分速度；角速度采用四元数最短旋转且符号不敏感。它不是接触力或夹持力测量。

| Seed2 评测 | 达标末步线速均值 | 达标末步角速均值 | 最后5个near样本线速均值 | 终点关键点误差均值 |
| --- | ---: | ---: | ---: | ---: |
| 确定性 | 0.211 m/s | 148.7°/s | 0.258 m/s | 26.05 mm |
| 原生随机 | 0.253 m/s | 136.9°/s | 0.232 m/s | 23.82 mm |

因此可以说**完成了当前抬升/移动到goal的原始任务判据**，不能说已经静止稳定保持。自动重置后看不到成功之后的行为，也没有验证真实接触/力闭合。最后5个near样本在随机回合可能不连续。

## 必须同时保留的三训练seed结果

下面全部是连续从零60M、相同原生控制、相同评测seed0，并非只选最好结果汇报：

| 训练seed | 确定性 | 原生随机/global-zeta |
| --- | ---: | ---: |
| 0 | 62/64 = 96.88% | 127/128 = 99.22% |
| 1 | 1/64 = 1.56% | 88/128 = 68.75% |
| 2 | 64/64 = 100% | 128/128 = 100% |

Seed2可以作为当前演示checkpoint；这是看过三个结果后的选择，不能把它的100%当成算法整体表现。Seed1的确定性差异必须保留。

## 来源与范围

Checkpoint：`models/seed2/step58596`。actor digest：`b7d76bd2640c17497ca3fa2b9294c3b75356814e04ef1db0fed61bc742045a82`。

连续60,002,304 transitions、1024env、58,596 vector steps；从随机参数开始，没有checkpoint、replay或BC加载。固定eraser、固定初始姿态和goal、DR关闭、state162输入、arm filter=1、hand filter=.1；评测32env、seed0、普通reset、原生随机噪声，不含额外startup/precision/控制干预。

这里只证明这一固定模拟任务。不是完整STR benchmark、论文全部任务复现、长期稳定抓握或真机部署证明。当前reset保留原有previous-reward语义，核验没有另行修改。
