# BC → 原生 SAC：技能保留中途汇总

读取时间：2026-09-08T16:07:52.342729+00:00；队列状态：completed。

只读 CPU 汇总。缺文件表示未完成，不代表成功率为0。成功为真实 task all_goals_hit；抬升和持握代理分开。

## Rollout

| 条件 / snapshot | 完成/请求 | 目标成功 | 抬升 | 持握代理 | 平均reward | 平均步数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| selected_seed0_native | 128/128 | 128/128 | 128/128 | 128/128 | 1388.35 | 62.38 |
| selected_seed1_native | 128/128 | 128/128 | 128/128 | 128/128 | 1388.50 | 62.42 |
| selected_seed0_independent | 128/128 | 128/128 | 128/128 | 128/128 | 1388.02 | 62.21 |
| immediate_update10_deterministic | 64/64 | 64/64 | 64/64 | 64/64 | 1402.14 | 77.67 |
| immediate_update10_stochastic | 128/128 | 95/128 | 97/128 | 100/128 | 1192.73 | 214.20 |
| immediate_update100_deterministic | 64/64 | 0/64 | 0/64 | 0/64 | 584.71 | 600.00 |
| immediate_update100_stochastic | 128/128 | 0/128 | 0/128 | 0/128 | 572.38 | 600.00 |
| immediate_update1000_deterministic | 64/64 | 0/64 | 0/64 | 0/64 | 587.58 | 600.00 |
| immediate_update1000_stochastic | 128/128 | 0/128 | 0/128 | 0/128 | 550.96 | 600.00 |
| immediate_update2000_deterministic | 64/64 | 0/64 | 0/64 | 0/64 | 585.01 | 600.00 |
| immediate_update2000_stochastic | 128/128 | 0/128 | 0/128 | 0/128 | 556.01 | 600.00 |
| immediate_update4000_deterministic | 64/64 | 0/64 | 0/64 | 0/64 | 590.71 | 600.00 |
| immediate_update4000_stochastic | 128/128 | 0/128 | 0/128 | 0/128 | 555.96 | 600.00 |
| immediate_final_deterministic | 64/64 | 0/64 | 0/64 | 0/64 | 599.44 | 600.00 |
| immediate_final_stochastic | 128/128 | 0/128 | 0/128 | 0/128 | 556.59 | 600.00 |
| warmup2000_update10_deterministic | 64/64 | 64/64 | 64/64 | 64/64 | 1388.04 | 62.00 |
| warmup2000_update10_stochastic | 128/128 | 128/128 | 128/128 | 128/128 | 1388.35 | 62.38 |
| warmup2000_update100_deterministic | 64/64 | 64/64 | 64/64 | 64/64 | 1388.04 | 62.00 |
| warmup2000_update100_stochastic | 128/128 | 128/128 | 128/128 | 128/128 | 1388.35 | 62.38 |
| warmup2000_update1000_deterministic | 64/64 | 64/64 | 64/64 | 64/64 | 1388.04 | 62.00 |
| warmup2000_update1000_stochastic | 128/128 | 128/128 | 128/128 | 128/128 | 1388.35 | 62.38 |
| warmup2000_update2000_deterministic | 64/64 | 64/64 | 64/64 | 64/64 | 1388.04 | 62.00 |
| warmup2000_update2000_stochastic | 128/128 | 128/128 | 128/128 | 128/128 | 1388.35 | 62.38 |
| warmup2000_update4000_deterministic | 64/64 | 0/64 | 0/64 | 0/64 | 604.65 | 600.00 |
| warmup2000_update4000_stochastic | 128/128 | 0/128 | 0/128 | 0/128 | 566.48 | 600.00 |
| warmup2000_final_deterministic | 64/64 | 0/64 | 0/64 | 0/64 | 600.68 | 600.00 |
| warmup2000_final_stochastic | 128/128 | 0/128 | 0/128 | 0/128 | 566.28 | 600.00 |

## Checkpoint 更新次数 / reward normalization

每格实际 optimizer step 为 min–max，不用网络文件中恒为0的包装计数代替。

| 条件 / snapshot | network calls | actor / critic / temp 实际step | alpha | reward除数 | RMS样本数 | 全tensor有限 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| immediate/update0 | 0 | 0–0 / 0–0 / 0–0 | 0.010000 | 238.629 | 100352 | True |
| immediate/update1 | 1 | 1–1 / 1–1 / 1–1 | 0.010003 | 238.629 | 100352 | True |
| immediate/update2 | 2 | 1–1 / 2–2 / 1–1 | 0.010003 | 238.629 | 100352 | True |
| immediate/update10 | 10 | 5–5 / 10–10 / 5–5 | 0.010014 | 238.629 | 104448 | True |
| immediate/update100 | 100 | 50–50 / 100–100 / 50–50 | 0.010061 | 238.629 | 150528 | True |
| immediate/update1000 | 1000 | 500–500 / 1000–1000 / 500–500 | 0.008921 | 238.629 | 611328 | True |
| immediate/update2000 | 2000 | 1000–1000 / 2000–2000 / 1000–1000 | 0.007573 | 238.629 | 1123328 | True |
| immediate/update4000 | 4000 | 2000–2000 / 4000–4000 / 2000–2000 | 0.005703 | 238.629 | 2147328 | True |
| immediate/final | 19338 | 9668–9668 / 19334–19334 / 9669–9669 | 0.001120 | 238.629 | 10000384 | True |
| warmup2000/update0 | 0 | 0–0 / 0–0 / 0–0 | 0.010000 | 238.243 | 100352 | True |
| warmup2000/update1 | 1 | 0–0 / 1–1 / 0–0 | 0.010000 | 238.243 | 100352 | True |
| warmup2000/update2 | 2 | 0–0 / 2–2 / 0–0 | 0.010000 | 238.243 | 100352 | True |
| warmup2000/update10 | 10 | 0–0 / 10–10 / 0–0 | 0.010000 | 238.243 | 104448 | True |
| warmup2000/update100 | 100 | 0–0 / 100–100 / 0–0 | 0.010000 | 268.955 | 150528 | True |
| warmup2000/update1000 | 1000 | 0–0 / 1000–1000 / 0–0 | 0.010000 | 279.021 | 611328 | True |
| warmup2000/update2000 | 2000 | 0–0 / 2000–2000 / 0–0 | 0.010000 | 281.766 | 1123328 | True |
| warmup2000/update4000 | 4000 | 999–999 / 4000–4000 / 1000–1000 | 0.008112 | 268.955 | 2147328 | True |
| warmup2000/final | 19338 | 8661–8661 / 19338–19338 / 8669–8669 | 0.001326 | 268.955 | 10000384 | True |

## 已写入TB的窗口

| 条件 | 指标 | 首值 | 最新值 / transitions | 最小值 | 最大值 |
| --- | --- | ---: | --- | ---: | ---: |
| immediate | actor/entropy | -109.09 | 19.422 / 10000384 | -109.09 | 19.501 |
| immediate | temperature/value | 0.010003 | 0.0011218 / 10000384 | 0.0011218 | 0.010047 |
| immediate | episode/return | 1205.9 | 558.18 / 10000384 | 397.98 | 1205.9 |
| immediate | episode/final/all_goals_hit | 0.84419 | 0 / 10000384 | 0 | 0.84419 |
| warmup2000 | actor/entropy | -117.06 | 15.827 / 10000384 | -117.06 | 16.241 |
| warmup2000 | temperature/value | 0.010003 | 0.001328 / 10000384 | 0.001328 | 0.010066 |
| warmup2000 | episode/return | 1208.8 | 568.59 / 10000384 | 420.37 | 1388.5 |
| warmup2000 | episode/final/all_goals_hit | 0.8431 | 0 / 10000384 | 0 | 1 |

## 解释限制

- Partial read-only snapshot: pending outputs are not failures or zero success.
- Success is task all_goals_hit; lift and sustained-near-palm are separate flags/proxies.
- Fixed single-task reset distribution, not generalization or from-scratch learning.
- TB scalars are logged window means in environment transitions; snapshot IDs are network calls.
- TB brackets are adjacent logged windows, not exact instantaneous checkpoint entropy/alpha.
- Reward normalizer is not logged to TB; exact saved statistics and scaling divisor are read from checkpoint.
- RMS sample count approximates processed transitions; it includes initialization epsilon/count and may differ with other initializers.
- Warmup changes collected data and actor/temp optimizer schedule positions; not an isolated offline-Q test.

完整JSON含每个snapshot实际优化器计数与live audit核对、checkpoint SHA256/有限性、reward各分量、TB原始标量序列及snapshot两侧窗口。
