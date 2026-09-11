# 50M学习曲线独立核对：reward不等于goal成功

**hand=.1的50M独立评测中全部回合曾跨过10cm抬升阈值，但多数随后掉落；20项clean-reset checkpoint评测仍全部0 goal。其训练日志晚期出现80%左右goal窗口，与独立评测显著不一致，不能据TB宣称已获得可靠目标策略。** hand=1.0则主要提高了阈值以下的高度shaping。

## 1. 同样约1200的训练reward，含义截然不同

| 最终训练窗口分量 | hand1.0 | hand0.1 |
| --- | ---: | ---: |
| episode/return | 1159.850952 | 1199.849243 |
| episode/cumulative/fingertip_delta_rew | 26.191847 | 29.191133 |
| episode/cumulative/lifting_rew | 1159.821777 | 27.106155 |
| episode/cumulative/lift_bonus_rew | 0.000000 | 300.000000 |
| episode/cumulative/keypoint_rew | 0.000000 | 11.225805 |
| episode/cumulative/kuka_actions_penalty | -10.596605 | -14.583043 |
| episode/cumulative/hand_actions_penalty | -15.568135 | -10.382541 |
| episode/cumulative/bonus_rew | 0.000000 | 857.291687 |
| episode/final/all_goals_hit | 0.000000 | 0.802083 |
| episode/length | 600.000000 | 254.447922 |
| episode/final/done_timeout | 1.000000 | 0.197917 |

逐窗口验证：7个reward分量之和与return最大差约.0047，最大相对差约1.2e−5，符合不同求和顺序/FP32累加误差的量级；all_goals_hit与done_max_successes每个窗口完全一致。

官方当前reward代码在抬升前每步给 `20*clamp(.05+Δz,0,.5)`，跨过`.05+Δz>.15`即真实升高10cm后才给一次300，并关闭这项dense shaping。因此lifting_rew大不代表已完成任务抬升。

hand1.0最终确定性评测平均最高升高约5.12cm，仍低于10cm阈值；其rawreturn1169.67中1161.89来自这一dense项，lift/goal bonus均0。说它完全没移动物体也不准确：它学到的是阈值以下的抬高/维持，而不是已完成抬升搬运。

这些rawreturn是未折扣和，episode长度也不同。不能据长时间hover的rawreturn更高，直接宣布它是SAC最优策略；SAC用γ=.99、reward normalization及熵项。

## 2. 独立评测显示的实际阶段

| hand | 预算M | det goal；lift | stoch goal；lift | det rawreturn | stoch rawreturn |
| --- | ---: | --- | --- | ---: | ---: |
| 1.0 | 10.000 | 0/64; 0/64 | 0/128; 0/128 | 596.48 | 451.98 |
| 1.0 | 20.001 | 0/64; 0/64 | 0/128; 1/128 | 800.73 | 667.89 |
| 1.0 | 30.001 | 0/64; 0/64 | 0/128; 0/128 | 1150.10 | 1082.40 |
| 1.0 | 40.002 | 0/64; 0/64 | 0/128; 0/128 | 1159.19 | 1141.61 |
| 1.0 | 50.002 | 0/64; 0/64 | 0/128; 0/128 | 1169.67 | 1085.24 |
| 0.1 | 10.000 | 0/64; 0/64 | 0/128; 5/128 | 611.88 | 519.93 |
| 0.1 | 20.001 | 0/64; 64/64 | 0/128; 128/128 | 341.39 | 275.31 |
| 0.1 | 30.001 | 0/64; 64/64 | 0/128; 128/128 | 338.60 | 292.26 |
| 0.1 | 40.002 | 0/64; 0/64 | 0/128; 0/128 | 571.06 | 554.64 |
| 0.1 | 50.002 | 0/64; 64/64 | 0/128; 128/128 | 333.13 | 317.21 |

hand=.1在20M/30M/50M的det与stoch评测中抬升均100%；持握近掌部代理几乎/全部100%，但全部未达goal并主要等到600步。40M曾丢失抬升，之后恢复，不能把过程称为单调稳定收敛。

50M hand=.1最高升高约13.1cm，最终位置误差约17.2cm；det rawreturn333.13中有300抬升奖、0goal奖，stoch相同结论。随后物体低于初始高度的回合为det55/64、stoch127/128；reset_when_dropped=false允许它们继续到timeout。曾满足持握代理不是持续抓稳、接触确认或真机部署证明。

折扣rawreturn还揭示另一点：50M det hand1≈206.10，hand.1≈292.36；这与未折扣rawreturn1169.67对333.13的排序相反。比较奖励时必须注明折扣/长度，不能只盯一条TB return。

## 3. 训练中确有goal标志，但暂不等于checkpoint可复现

hand=.1在15.2064M第一次出现窗口平均lift bonus300；在42.8544M第一次出现正goal窗口。976个日志窗口中只有30个goal>0；这是窗口数，不是训练episode总成功率。

首次≥50%的goal窗口到49.7152M才出现；最高84.94%在49.8688M，最后窗口80.21%。同时goal bonus与done_max_successes支持“训练日志确实标了成功”，并非只有抬升奖。

但clean-reset评测所有保存点goal均0，包括50M。这是当前真正待解释的train/eval差异。可能涉及同步/异步重置、初始随机progress/首个随机action、1024训练env对32评测env、训练中的持续策略/BN更新与冻结checkpoint、仿真数值或数据分布；这里只列排查候选，未确认其中任何一个为原因。

特别是原生随机progress只在初次reset注入，不能直接用它解释50M末尾的所有差异；需要匹配协议实测。未做这项验证前，不应该宣布目标任务已可靠训成，也不应该忽略已验证的抬升技能。

## 4. 熵、alpha和学习率预算没有混算

| hand | 预算M | network calls | actor/critic/temp applied steps | actor LR | critic LR | actor scheduler step |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 1.0 | 10.000 | 19338 | 9667/19335/9669 | 0.000226170 | 0.000150037 | 9669 |
| 1.0 | 20.001 | 38870 | 19432/38858/19435 | 0.000150009 | 0.000150000 | 19435 |
| 1.0 | 30.001 | 58402 | 29198/58376/29201 | 0.000150000 | 0.000150000 | 29201 |
| 1.0 | 40.002 | 77934 | 38964/77897/38967 | 0.000150000 | 0.000150000 | 38967 |
| 1.0 | 50.002 | 97466 | 48730/97416/48733 | 0.000150000 | 0.000150000 | 48733 |
| 0.1 | 10.000 | 19338 | 9667/19335/9669 | 0.000226170 | 0.000150037 | 9669 |
| 0.1 | 20.001 | 38870 | 19429/38859/19435 | 0.000150009 | 0.000150000 | 19435 |
| 0.1 | 30.001 | 58402 | 29194/58378/29201 | 0.000150000 | 0.000150000 | 29201 |
| 0.1 | 40.002 | 77934 | 38960/77897/38967 | 0.000150000 | 0.000150000 | 38967 |
| 0.1 | 50.002 | 97466 | 48726/97415/48733 | 0.000150000 | 0.000150000 | 48733 |

LR decay_steps固定19532，但每个optimizer用自身scheduler：10M actor仍约2.2617e−4，critic已近1.5e−4；20M actor也接近end，30M起两者均1.5e−4。不是50M重启新日程，更不是10M后停止更新。AMP跳步导致实际optimizer计数略小于scheduler/attempted数，均从模型直接读取。

hand1最终H=-7.5246，alpha=7.9047823e-06；hand.1最终H=-16.6709，alpha=2.0616912e-05。hand.1首次成功窗口H≈−22.41、alpha≈2.14e−5；后期alpha在48.8448M达到最低约1.82e−5后略回升，与H落到native目标约−13.87以下的温度调节方向一致。

这给“高熵早期难形成技能、降低熵后出现动作结构”的假设提供时间相关证据，但不是因果隔离。当前另排的alpha-init配对实验才能进一步检验；本次不能把先后发生说成唯一原因。

## 结论边界

- Online success is a logged episode-final task flag, not independent checkpoint success.
- A positive-window count or unweighted mean is not an overall training episode success rate.
- Reward shaping accumulation and true goal bonus are separate; undiscounted TB return is not the SAC objective.
- Both50M conditions completed; comparisons remain one seed and one fixed task.
- Training vs clean-reset evaluation goal mismatch remains unresolved. No simulator/GPU or model update was run for this audit.

完整JSON保留所有事件窗口、连续正goal区间、每个checkpoint actual optimizer/scheduler/LR、每次评测reward分量及误差。没有修改reward、训练或评测配置。
