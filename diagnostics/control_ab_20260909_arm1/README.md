# 完整 STR：只改机械臂控制的单 seed 对照

状态：100M已完成，没有追加seed或预算。最终结果见 [FINAL_RESULT.md](FINAL_RESULT.md)；PROGRESS.md为早先60M快照。

用户于 2026-09-09 授权：按前述建议测试，且只需一个 seed。仅在本机运行仿真训练，不使用真机，不启动独立评测，不恢复已停止的多 seed 队列。

## 唯一任务改动

- 基线：`simtoolreal_full_nodr`，`arm_moving_average=0.1`。
- 新实验：`simtoolreal_full_arm1`，`arm_moving_average=1.0`。
- 其余保持：seed0、100,003,840 transitions、2048 env、原 FF140 actor / critic162、全部1200资产的完整任务分布、随机初态和位置/旋转目标、50 goal链、原reward/reset/curriculum、手部EMA=.1、DR关闭、gamma=.99、n_step=1、10M replay、从头训练。
- 不使用 LSTM，不改网络，不改抬升参考高度，不改奖励。不把这个诊断配置当作严格论文或真机部署配置。

基线 checkpoint：`/home/abao/flashsac-robotics/diagnostics/full_nodr_20260909_1004/models/seed0/step48830`。

## 运行位置

配置：`/home/abao/flashsac-robotics/configs/simtoolreal_full_arm1.yaml`。

systemd 服务：`flashsac-str-arm1-seed0-20260909.service`。只启动一次、不自动重启，最长运行2小时；正常预计约45分钟。进度必须以实际事件文件/日志为准。

日志：本目录 `train.log`。

最终 checkpoint：本目录 `models/seed0/step48830/`。

TensorBoard（整行复制）：

```bash
/home/abao/play2perfect/.venv_isaacsim/bin/tensorboard --logdir=/home/abao/flashsac-robotics/diagnostics/control_ab_20260909_arm1/tb --host=127.0.0.1 --port=6009
```

浏览器：`http://127.0.0.1:6009`。若6009已有其他TB占用，改用6010。

对比依据：相同 transitions 下的累计指尖奖励、连续抬升奖励、一次性抬升奖励、目标进步奖励和每episode完成goal数。一次性抬升奖励/300不等于稳定抓握成功率；不能只看总reward。

配置一致性检查：

```bash
/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python -m pytest -q /home/abao/flashsac-robotics/tests/test_str_arm1_config.py
```

这次单变量对照检验慢控制是否限制当前100M预算内的学习；无改善不排除更长预算或其他因素。
