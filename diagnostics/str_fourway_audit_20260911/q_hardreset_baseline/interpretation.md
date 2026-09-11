# 最后一次有界硬重置重放：120 步配对通过

**结果：通过这次短前缀重放检查。不是 Q 验证通过，也不是训练问题已经解决。**

32 个环境；参考与重复各 120 步。两边都先调用标准 `sim.reset(soft=False)`，使用原 STR helper 重施材质，再设置同一 seed 并执行原 task reset。参考执行冻结 500M Actor 的确定性动作，重复执行其逐步原样动作带。没有优化器、训练或额外动作分支。

- 从 step 0 到 step 120 的每个检查点，**32/32 环境均通过 `1e-5` 全部可见状态门限，Python／NumPy／Torch CPU/CUDA RNG 哈希一致**。检查覆盖机器人与物体状态、目标、手臂／手指控制目标、队列、reward/goal trackers，另包括真实关节控制缓冲与 body state。
- 终止与时间截断事件逐步相同；reward 最大绝对误差 `9.059906e-6`；模型 digest 未改变。
- 两次硬重置后的机器人／物体／桌子／目标的材料、质量、惯量，以及机器人 Kp/Kd、位置限位、力矩上限、速度上限，共 17 项数组与最初环境参数**逐元素相同（最大误差 0）**；joint/body 顺序相同。
- 总壁钟 `303.79 秒`，主要为场景初始化；仅执行这一次有界尝试，进程已正常退出。

这与前一次普通 `env.reset()` 在 prefix 60/120 完全无法配对形成鲜明对照，说明**在这一诊断中，硬重置并保持物理参数的协议能够恢复短前缀可重复性**。但它同时重建多类内部状态，不能进一步锁定某个 PhysX contact/solver cache 是唯一原因，也不能据此认定原训练的正常 episode reset 有 bug。

**仍未验证：** 600 步随机延续的 mean-repeat 方差是否下降；在恢复配对的搬运中途状态上，Q 排序是否与真实后果一致。原 `q_probe_500m` 的这些结论仍然不确定，不能用本结果替代完整 Q 动作对照。没有再开 7 分支试验，没有放宽状态门限，没有修改训练代码或 checkpoint。

证据文件：`summary.json`、`reference_trace.npz`、`repeat_trace.npz`、`original_physical_parameters.npz`、`reference_physical_parameters.npz`、`repeat_physical_parameters.npz`。实现：`/home/abao/flashsac-robotics/scripts/check_str_hardreset_replay.py`。
