# 50M 连续训练预算对照

读取时间：2026-09-08T17:06:12.427778+00:00。实时队列状态：completed。

arm filter均为1.0，对照hand=1.0与0.1；seed0，固定STR单物体/初态/goal。1024env，从零连续训练50,001,920 transitions；不加载10M replay再分段续跑。前10M学习率配方保持原值，之后沿用该原日程end学习率。

每个条件完成连续50M后，才逐个评测保存的10/20/30/40/50M模型。因此目前缺评测不表示训练没成功；不要把准备阶段metadata的prepared_only当作当前运行状态。

![Training and checkpoint evaluation curves](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/curve.png)

## Checkpoint 与独立评测

| hand filter | transitions | sampling | 完成/请求 | 真实goal成功 | 抬升 | 持握代理 | 平均raw reward |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 10,000,384 | deterministic | 64/64 | 0/64 | 0/64 | 0/64 | 596.48 |
| 1.0 | 10,000,384 | stochastic | 128/128 | 0/128 | 0/128 | 0/128 | 451.98 |
| 1.0 | 20,000,768 | deterministic | 64/64 | 0/64 | 0/64 | 0/64 | 800.73 |
| 1.0 | 20,000,768 | stochastic | 128/128 | 0/128 | 1/128 | 0/128 | 667.89 |
| 1.0 | 30,001,152 | deterministic | 64/64 | 0/64 | 0/64 | 0/64 | 1150.10 |
| 1.0 | 30,001,152 | stochastic | 128/128 | 0/128 | 0/128 | 0/128 | 1082.40 |
| 1.0 | 40,001,536 | deterministic | 64/64 | 0/64 | 0/64 | 0/64 | 1159.19 |
| 1.0 | 40,001,536 | stochastic | 128/128 | 0/128 | 0/128 | 0/128 | 1141.61 |
| 1.0 | 50,001,920 | deterministic | 64/64 | 0/64 | 0/64 | 0/64 | 1169.67 |
| 1.0 | 50,001,920 | stochastic | 128/128 | 0/128 | 0/128 | 0/128 | 1085.24 |
| 0.1 | 10,000,384 | deterministic | 64/64 | 0/64 | 0/64 | 0/64 | 611.88 |
| 0.1 | 10,000,384 | stochastic | 128/128 | 0/128 | 5/128 | 4/128 | 519.93 |
| 0.1 | 20,000,768 | deterministic | 64/64 | 0/64 | 64/64 | 64/64 | 341.39 |
| 0.1 | 20,000,768 | stochastic | 128/128 | 0/128 | 128/128 | 127/128 | 275.31 |
| 0.1 | 30,001,152 | deterministic | 64/64 | 0/64 | 64/64 | 64/64 | 338.60 |
| 0.1 | 30,001,152 | stochastic | 128/128 | 0/128 | 128/128 | 128/128 | 292.26 |
| 0.1 | 40,001,536 | deterministic | 64/64 | 0/64 | 0/64 | 0/64 | 571.06 |
| 0.1 | 40,001,536 | stochastic | 128/128 | 0/128 | 0/128 | 0/128 | 554.64 |
| 0.1 | 50,001,920 | deterministic | 64/64 | 0/64 | 64/64 | 64/64 | 333.13 |
| 0.1 | 50,001,920 | stochastic | 128/128 | 0/128 | 128/128 | 128/128 | 317.21 |

## 已保存模型的实际更新次数

| hand filter | transitions | network calls | actor / critic / temp optimizer steps | alpha | reward除数 | 有限性 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 1.0 | 10,000,384 | 19338 | 9667–9667 / 19335–19335 / 9669–9669 | 0.001189 | 78.600 | True |
| 1.0 | 20,000,768 | 38870 | 19432–19432 / 38858–38858 / 19435–19435 | 0.000285 | 81.551 | True |
| 1.0 | 30,001,152 | 58402 | 29198–29198 / 58376–58376 / 29201–29201 | 0.000082 | 81.551 | True |
| 1.0 | 40,001,536 | 77934 | 38964–38964 / 77897–77897 / 38967–38967 | 0.000024 | 81.551 | True |
| 1.0 | 50,001,920 | 97466 | 48730–48730 / 97416–97416 / 48733–48733 | 0.000008 | 81.551 | True |
| 0.1 | 10,000,384 | 19338 | 9667–9667 / 19335–19335 / 9669–9669 | 0.001178 | 84.430 | True |
| 0.1 | 20,000,768 | 38870 | 19429–19429 / 38859–38859 / 19435–19435 | 0.000293 | 98.871 | True |
| 0.1 | 30,001,152 | 58402 | 29194–29194 / 58378–58378 / 29201–29201 | 0.000083 | 98.871 | True |
| 0.1 | 40,001,536 | 77934 | 38960–38960 / 77897–77897 / 38967–38967 | 0.000024 | 193.963 | True |
| 0.1 | 50,001,920 | 97466 | 48726–48726 / 97415–97415 / 48733–48733 | 0.000021 | 251.597 | True |

## 最新在线TB窗口

| hand filter | 指标 | 最新值 / transitions | 历史最小 | 历史最大 |
| --- | --- | --- | ---: | ---: |
| 1.0 | episode/return | 1159.85 / 50,001,920 | 48.55 | 1159.85 |
| 1.0 | episode/final/all_goals_hit | 0 / 50,001,920 | 0 | 0 |
| 1.0 | episode/cumulative/lift_bonus_rew | 0 / 50,001,920 | 0 | 50.5618 |
| 1.0 | actor/entropy | -7.52462 / 50,001,920 | -97.301 | 19.5182 |
| 1.0 | temperature/value | 7.90478e-06 / 50,001,920 | 7.90478e-06 | 0.0101133 |
| 0.1 | episode/return | 1199.85 / 50,001,920 | 56.7318 | 1251.24 |
| 0.1 | episode/final/all_goals_hit | 0.802083 / 50,001,920 | 0 | 0.849421 |
| 0.1 | episode/cumulative/lift_bonus_rew | 300 / 50,001,920 | 0 | 300 |
| 0.1 | actor/entropy | -16.6709 / 50,001,920 | -95.6838 | 19.6028 |
| 0.1 | temperature/value | 2.06169e-05 / 50,001,920 | 1.82056e-05 | 0.01011 |

源码捕获复核：38/39相同；差异路径和前后hash保留在summary.json，不覆盖旧捕获。

## 限制

- Missing checkpoints/evaluations mean pending, never zero success.
- Eval success is true all_goals_hit; task lift and sustained-near-palm proxy remain separate.
- Online TensorBoard values are logged window means, not independent full-budget success rates.
- TB x-axis is processed transitions; checkpoint step names are vector interactions; optimizer counters are read independently.
- Only seed0 and fixed single-task initial settings; no generalization, convergence, or paper-reproduction claim.
- Each condition trains continuously to50M before its stored10/20/30/40/50M checkpoints are evaluated. Early missing evals do not indicate failure.
- Connecting eval points are guides between measured checkpoints; online train returns and evaluation returns use different protocols/state occupancy.

JSON另存全部TB标量、checkpoint两侧真实日志窗口、完整checkpoint SHA256/optimizer计数/有限性，以及评测reward各分量。
