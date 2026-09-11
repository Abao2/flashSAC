# 最终 normal checkpoint：独立成功核验通过

2026-09-09，纯 CPU、没有新开仿真或更新参数。数据详情与 SHA256 在同目录 `success_validation.json`。

| 固定任务评测 | 独立重建成功 | 成功均为 terminated | 超时失败 | 成功终点关键点误差均值 |
| --- | ---: | ---: | ---: | ---: |
| 确定性 | 64/64 = 100% | 64，truncated 为 0 | 0 | 17.24 mm |
| 原生随机 / zeta | 110/128 = 85.94% | 110，truncated 为 0 | 18 | 20.73 mm |

## 判据具体是什么

原环境使用固定 reward 尺寸 `[0.141, 0.03025, 0.0271] m` 的四个角点，乘 `keypoint_scale=1.5`；四个角点到目标对应角点的最大距离 ≤ **30 mm**，在一个回合内**累计 10 帧**，即完成一个 goal。本任务只要求一个 goal，因此在第 10 帧终止。

不是连续 10 帧，也不是“抬起来就成功”。确定性成功中 53/64、随机成功中 79/110 没有连续 10 帧达标，只是累计达到 10 帧；这符合当前原环境判据，但不能说已持续稳定保持。

不能直接取 observation 的 `keypoints_rel_goal` 四个向量长度当 reward 判据：observation 使用实际 eraser 尺寸，reward 使用上述固定尺寸。本次先由这四点的对称均值恢复物体中心，再结合记录的 object xyzw 四元数，用独立 NumPy float64 旋转公式重建 **reward 角点**；没有复用环境的成功标记或几何函数作为计算结果。

## 已逐项核对

- 共 21,774 transitions、192 个完整回合；每回合 step 连续，只有末步 done，记录的 `next_obs` 使用自动重置前 final_obs。回合内 `next_obs[t] == obs[t+1]` 的最大误差为 0。
- 所有 174 个成功回合恰好在第 10 个累计 near 帧结束；独立计数与 episode success、NPZ 标签、`expm1(successes_obs)` 一致。成功全部 `terminated=True, truncated=False`，没有把 timeout 当 success。
- 随机失败的 18 个回合都是 600 步 timeout；累计 near 只有 4–9 帧。失败 ID 和帧数完整保存在 JSON。
- 每个 near 帧的 goal bonus 是 **100**，因此成功累计得到 **1000**；并非最后一步突然 +1000。独立累计 bonus 与每回合 `bonus_rew` 的最大误差为 0。
- 由物体高度独立重建 `0.05 + z - z_init > 0.15` 的首次跨越与 sticky lift，匹配一次性 **+300** lift bonus，误差为 0。所有成功回合确实先抬起物体。
- 原始 reward 逐步求和与 episode return 完全一致；各 reward components 合计也与 return 一致。确定性平均 return 1356.01，不是单凭这个数字判成功。
- 两评测加载相同最终 actor，前后 digest 完全一致：`f35ced3258d0d68dff361c477aeaaec8aec30a98053769644ca5408ed4b9bb9b`。在 CPU 严格加载该 checkpoint 并重新计算全部确定性动作，与记录动作平均误差 6.07e-6、最大 1.01e-4。
- 任务配置与 native normal 训练一致：arm filter=1、hand filter=.1；评测 seed0、32env、正常 reset，没有 training-startup 或新 precision 干预。实际 eval TF32 matmul=false，cudnn TF32=true，cudnn benchmark=true，matmul precision=highest。

## 这个结果来自哪里、可以说到哪一步

Checkpoint：`training/normal/models/final/step4883`。

来源是 **50,001,920 transitions 的从零 FlashSAC 训练 + 5,000,192 transitions 的 fresh-replay/new-simulator native continuation**，累计 55,002,112 transitions。源训练没有加载 checkpoint，没有 BC；后续新增 9572 次 network calls，native counter=107038，训练 audit complete。不能写成“不间断 55M 续训”。

可以确认：**FlashSAC 的这个固定 eraser 抬升/移至目标任务，已在冻结 checkpoint 的独立 rollout 中达到当前任务判据。**

仍不能外推：完整 STR 任务集、随机初态/物体、论文复现、连续稳定抓握、接触力闭合或真机可部署。这里只有一个训练 seed；DR 关闭；当前 reset 仍保留 previous-reward 的既有语义。额外评测 seed 可检查随机 rollout 稳健性，但不能替代独立训练 seed。

## 暂停的准备

`scripts/diagnose_str_live_freeze.py` 仅保留未运行、未测试初稿，已标注 UNRUN。因这个最终 checkpoint 已成功，live-freeze 优先级暂停；没有生成队列或启动 GPU，不应直接执行该初稿。
