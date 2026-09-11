# FlashSAC完整STR：手响应速度单变量实验

用户授权继续训练，2026-09-10启动准备。唯一任务变化为hand_moving_average由.1改1；arm=1、hand绝对目标映射、reward、reset、6类/1200物体分布、DR关闭、140维actor/162维critic、FlashSAC算法均保持。

候选从零seed0训练100,003,840 transitions，2048env；与已完成的control_ab_20260909_arm1同预算比较。不是100M checkpoint续训，也不是恢复旧replay。单seed结果仅用于筛选下一步，不能当统计显著性证明。

## 执行与产物

本机唯一GPU串行：先train_str_holding.py训练，再play_str_full_diagnostic.py同协议回放。各阶段上限见manifest.json，总批次截止北京时间2026-09-10 05:05。队列只终止其自身子进程，不操作其他GPU用户或远程任务。systemd单元名flashsac-str-hand1-seed0-20260910.service，Restart=no。

- 运行状态：queue_status.json。
- checkpoint：candidate/models/seed0/step9766、19532、29298、39064、48830（20/40/60/80/100M）。
- 最终replay：candidate/models/seed0/step48830/replay_buffer.pt，约25.5GB。
- TB事件：candidate/runs；原始运行日志train.log。
- 成功标记：candidate/training_complete.json必须在完整checkpoint及replay落盘后写出。
- 回放：play/summary.json、play/paired_play.mp4、play/play_complete.json；64个首episode/模式，确定性和原生随机各20秒上限，.075 tolerance。不冒充论文24任务评测。

## 怎么判断改善

先看episode/final/successes提高、episode/final/done_fall下降，再看height_proxy_longest_above_10cm_s和height_proxy_above_10cm_1s。高度指标是相对每次reset高度的几何量，不是传感器证实的稳定抓握。新增统计在原reward hook返回后只读记录，不进入reward/observation/action。

最终与基线play_20260909_v4的同协议回放对比；不能把新训练窗口均值与旧64回合评测直接视作A/B。基线确定性/随机分别：抬升62/59，连续高于10cm至少1秒8/7，至少1goal为11/10，下落终止54/55（每组64）。

原生FlashSAC的网络、奖励、步长和优化器未改。replay加载改成CPU暂存再写入已分配GPU buffer，避免两份25.5GB同时在GPU；需要足够主机内存。即使保存replay，环境/RNG/curriculum仍未快照，后续续训也不能宣称逐步精确恢复。

## 本机看曲线

`/home/abao/play2perfect/.venv_isaacsim/bin/tensorboard --logdir=/home/abao/flashsac-robotics/diagnostics/hand_response_20260910/tb --host=127.0.0.1 --port=6008`

tb内两条只读目录链接分别指向已有hand.1基线和本次hand1候选。纵轴episode/final/height_proxy_*仅候选有；旧基线需用已保存play比较。
