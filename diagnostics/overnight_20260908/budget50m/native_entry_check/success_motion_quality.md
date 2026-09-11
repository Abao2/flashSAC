# 成功动作质量：达标时是否已经停稳？

这里独立核对当前native normal final checkpoint的已有记录。**任务成功真实成立，但判据是累计10帧near，不要求连续停稳；成功立即终止，不能测量成功后的保持、接触力或稳定抓握。**

## 任务成功与连续达标分开

| 评测 | 成功/全部回合 | 成功中曾连续10帧 | 完成时间均值/中位数(s) | 最长连续near帧数：中位数/范围 |
| --- | ---: | ---: | ---: | --- |
| native_det_seed0 | 64/64 | 11/64 | 0.535/0.533 | 7.5/[6,10] |
| native_stoch_seed0 | 110/128 | 31/110 | 1.351/0.567 | 8.0/[3,10] |
| native_stoch_seed1 | 218/256 | 63/218 | 1.487/0.567 | 8.0/[4,10] |
| native_stoch_seed2 | 229/256 | 52/229 | 1.461/0.583 | 8.0/[3,10] |
| bc_det_context | 64/64 | 64/64 | 1.033/1.033 | 10.0/[10,10] |

BC仅作背景：其arm filter=.1，native=1；hand均.1。不能把完成速度/运动大小差别单独归因于算法。其他四行是同一训练checkpoint的不同采样/rollout seed，不是四个独立训练seed。所有请求回合均完成，失败仍保留在成功率分母。

## 成功终止帧仍有多少运动

下表只汇总成功回合。速度是当前policy区间(obs_t→pre-reset next_obs_t)的有限差分平均，不是瞬时PhysX速度；dt约16.667ms。末5个near帧不一定连续。

| 评测 | 终止线速度：均值/中位数/p90(m/s) | 终止角速度：均值/中位数/p90(deg/s) | 末5 near线速度均值(m/s) | 末5 near角速度均值(deg/s) |
| --- | --- | --- | ---: | ---: |
| native_det_seed0 | 0.458/0.471/0.571 | 131.9/138.5/152.1 | 0.236 | 113.6 |
| native_stoch_seed0 | 0.352/0.347/0.597 | 100.0/95.7/158.4 | 0.283 | 99.3 |
| native_stoch_seed1 | 0.345/0.331/0.590 | 101.7/102.1/148.5 | 0.291 | 99.8 |
| native_stoch_seed2 | 0.363/0.358/0.594 | 102.6/101.3/151.7 | 0.291 | 96.6 |
| bc_det_context | 0.011/0.011/0.011 | 64.0/63.8/64.6 | 0.018 | 64.9 |

这些数值描述达标时的运动，不是预先定义了合格/不合格速度阈值，也不能外推下一秒会继续稳住或掉落。

## 抬升高度、可见抬升时段和近掌部

| 评测 | 回合最高高度均值(cm) | 回合最低高度均值(cm) | 终止高度均值(cm) | >10cm最长连续采样跨度中位数(s) | 终止物体–palm距离均值(cm) | 成功中曾低于初始高度(抬升后) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native_det_seed0 | 15.64 | -0.10 | 15.60 | 0.217 | 6.03 | 0/64 |
| native_stoch_seed0 | 16.91 | -0.10 | 14.91 | 0.250 | 6.95 | 0/110 |
| native_stoch_seed1 | 16.98 | -0.10 | 15.02 | 0.250 | 7.02 | 0/218 |
| native_stoch_seed2 | 16.95 | -0.10 | 14.98 | 0.267 | 7.01 | 0/229 |
| bc_det_context | 13.82 | -0.10 | 13.82 | 0.283 | 7.40 | 0/64 |

高度相对reset初始物体中心，包含起始落到桌面的微小下降。连续k个采样点的首末时间跨度为(k−1)×dt；不能把采样点当作已经验证的连续物理保持。Palm距离不是接触/力闭合检验。

代表图选episode0：完成步数最接近det中位数，再按最长near段接近中位数、最后最低ID，未挑最好回合。图只到成功终止帧，没有后续数据。

![成功动作实际轨迹](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/native_entry_check/success_motion_trace.png)

## 核验与边界

逐回合确认：step连续；只有末步done；所有内部next_obs[t]与obs[t+1]完全相同；差分仅用同一条transition的前后观测，从不跨reset。使用success_validation中的固定reward尺寸与1.5缩放重建角点，near<=.03m，累计第10帧恰好成功终止；与success标签/每near100分goal bonus一致。记录的物体位置、终点误差、sticky lift与独立几何匹配。

- Goal success terminates immediately; there is zero recorded post-success stability horizon.
- No contact force/force-closure/slip verification. Palm proximity and height cannot certify stable grasp or deployment readiness.
- Motion metrics summarize successful episodes separately; failed episodes are not dropped from success denominators.
- Metrics requiring near samples are null for a never-near episode; aggregate n counts available values, not fabricated zeros.
- One training seed/fixed object/start/goal; rollout seed1/2 do not establish independent training reproducibility.
- BC is contextual only: its arm filter.1 differs from native1.0, so completion speed/motion are not an algorithm-only comparison.
- Finite differences average one policy interval (16.667ms), cannot resolve within-interval impulses/oscillation; shortest quaternion differences alias rotations above pi per interval.

JSON保留全部回合的near帧序号、连续段、末5帧速度、终点速度、高度和距离，以及原始文件与actor文件SHA256。四组native记录均指向normal/final/step4883、相同模型digest且评测前后不变；控制均为arm=1、hand=.1。

CPU重跑：`/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python /home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/native_entry_check/audit_success_motion.py`。本次无GPU、无环境/核心代码/main report修改。
