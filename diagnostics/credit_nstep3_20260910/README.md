# 完整 STR + FlashSAC：三步回报单变量实验

已于2026-09-10 03:37完成100M及配对play；模型/replay/视频均验证。最终结论与曲线见[FINAL_RESULT.md](FINAL_RESULT.md)。本批自动续查已暂停，GPU计算进程已退出，未追加训练。

2026-09-10。本轮是夜间最后一组新增100M；连同hand_response_20260910已完成100M，总计200,007,680 transitions（按完整并行步向上取整）。不是无限训练，也不代表100M足以完整收敛。

## 唯一实验变量

对照为control_ab_20260909_arm1：arm EMA=1、hand EMA=.1、n_step=1。本组仅n_step=3；重新从零seed0训练，不载入checkpoint或旧replay。保持gamma=.99、网络、原reward、随机初始状态和goal、50-goal链、原成功判据、2048环境、完整6类/1200资产、DR关闭及100M预算。

现有scripts/run_isaaclab.sh采用n_step=3，而STR配置覆盖为1。假设是让稀疏目标奖励与抬升后果更直接影响前面的动作价值；这不是已确认根因，n-step也会改变off-policy偏差/方差。它不改变环境单步reward，不等于把抬升奖励放大三倍。

hand=1上组退步，因此本组恢复较好的hand=.1。不要把本组和hand=1组称为单变量对照：公平基准是arm1/hand.1/nstep1。

## 运行与检查

- 配置：../../configs/simtoolreal_full_arm1_nstep3.yaml；完整命令在manifest.json。
- 本机systemd：flashsac-str-nstep3-seed0-20260910.service。GPU串行，不动远端或真机。
- 状态：queue_status.json；日志：train.log、play.log。
- 模型：candidate/models/seed0/step48830；同时保存最终replay约25.48GB。
- TB：candidate/runs；合并入口沿用../hand_response_20260910/tb，新增arm1_hand01_nstep3。
- 完成后自动play：64配对首episode/模式，确定性与原生随机两组；play/paired_play.mp4与summary.json。不是论文评测。

预计训练加play约50–60分钟，以实际吞吐为准。04:00后不启动新训练；队列05:05前停止自身计算。标记文件需检查status=complete及完整checkpoint/video，不以进程退出或reward上升当成功。

先看goal、真实fall/hand_far终止，再看连续高于初始10cm的持续时间。高度只是几何代理，不证明稳定抓握；done_dropped关闭时其0值不能解释为不掉落。

## 可复查的预检

test_str_nstep3_config.py锁定完整任务/算法差异，并用真实TorchBuffer验证5次add产生的3个滑动窗口：终止/时间截断不会把reset后的100奖励或错误next_obs串入前一episode，discount使用实际有效步数。其余现有n-step、replay加载和height旁路测试一起运行，结果见preflight.json。

模型和replay虽可续读，环境/RNG/curriculum没有快照；不可宣称逐步精确恢复。未提交或推送Git。
