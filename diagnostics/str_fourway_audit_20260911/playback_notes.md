# 本机查看两个 policy

在 **abao 本机图形桌面的终端**运行，不需要激活 conda。先进入：

```bash
cd /home/abao/flashsac-robotics
```

看本轮暂选的 FlashSAC checkpoint：

```bash
bash scripts/play_str_behavior.sh best
```

关闭前一个窗口后，再看官方 pretrained checkpoint：

```bash
bash scripts/play_str_behavior.sh official
```

- **两个都是 KUKA + Sharpa，不是 Wuji。** 都是确定性推理、1 个环境，不训练，也不加载训练 replay。
- `best` 当前指 **500M**：仅在本轮 100M / 500M 两个候选中暂选；没有证明统计显著领先、全 checkpoint 最优或已经收敛。它使用随机初始状态与随机目标链、无 DR、arm EMA=1 / hand EMA=0.1，固定 tolerance=`0.02905653603374958`。
- `official` 使用官方 config/model、LSTM 与观测归一化，播放 DexToolBench `claw_hammer / swing_down` 完整轨迹，arm/hand EMA 均为 0.1。**它与 Flash 的随机目标不是同一个评测协议，只用于看行为。** 两个入口均为 Isaac Lab，不是 legacy Isaac Gym。
- Flash 启动仍会生成/转换完整 **1,200 个物体资产**，本机可能约需 **5 分钟**。先看终端是否继续输出资产转换进度，不要重复开启第二份。进入播放后按仿真速度节流。
- 一次只开一个 Isaac 窗口。脚本选择了现有 Python，并固定使用 STR 仓库的 `rl_games`；无需重新安装依赖。窗口关闭路径有退出逻辑，终端也可按 `Ctrl+C` 停止。

## 已有视频与验证边界

[500M 配对测试视频](/home/abao/flashsac-robotics/diagnostics/str_fourway_audit_20260911/matched_play/500M/paired_play.mp4) · [100M 配对测试视频](/home/abao/flashsac-robotics/diagnostics/str_fourway_audit_20260911/matched_play/100M/paired_play.mp4)

视频均已用 ffprobe 验证为 H.264、960×720、30 fps、40 秒：前 20 秒确定性，后 20 秒随机策略；四个预选视角。视频中发生 reset 后的新回合不计入首回合评测结果。

已检查脚本语法、配置、官方 CPU 推理/LSTM reset 与上述离屏视频。**官方 GUI 已实际完成一个回合：37/37 个目标，环境报告 10.6 秒，进程正常退出（exit 0）。** 这是单任务、单回合测试，不是完整 benchmark，也没有证明与 Flash 的随机目标任务可直接比较。[官方 GUI 实测记录](/home/abao/flashsac-robotics/diagnostics/str_fourway_audit_20260911/official_gui_smoke.json)

**Flash 的 `best` GUI 命令本身仍未实际运行**；其 100M/500M 匹配离屏渲染已经完成。手动关窗退出也尚未单独实测。GUI 的单环境初始状态不保证等于视频里某个环境。

[配对评测与选择依据](/home/abao/flashsac-robotics/diagnostics/str_fourway_audit_20260911/matched_play/comparison.json) · [官方 CPU 核验](/home/abao/flashsac-robotics/diagnostics/str_fourway_audit_20260911/official_cpu_check_fixed_path.json)
