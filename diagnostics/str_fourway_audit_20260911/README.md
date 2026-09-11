# FlashSAC / SimToolReal 四项诊断（2026-09-11）

## 先看结论

**本轮未恢复训练。当前 Lab / noDR / arm1 配置下，没有发现 Flash wrapper 接错动作、观测、reward 或终止信号；策略能完成部分目标，但尚未稳定完成任务。Q 是否为主要瓶颈仍未确定。**

- 用户重启后，NVIDIA 内核/用户库统一为 580.178.04，Torch CUDA 恢复。
- 独立原生 Lab / wrapper：32环境×600步，59类字段逐元素相等，最大差异0。
- 同起点、同 tolerance 比较：500M 确定性22/64回合至少达到一个目标，100M为16/64；不能称完整任务成功率。
- Q长对照完成，但重复基线漂移较大，60/120步状态配对失败，不能据此宣判Q错。hard-reset后120步重复检查通过，但未重跑完整Q长对照。
- 官方 checkpoint 本地 GUI 单回合实测：claw_hammer / swing_down 完成37/37 goals，正常退出。不是全基准评测。

本轮只新增/维护诊断脚本、播放脚本和结果；没有修改生产训练算法、环境reward、checkpoint、系统驱动；未提交/推送Git。已有工作树改动保留。

## 1. 环境接口：本次覆盖范围内通过

两个独立进程分别从“native config + gym.make”和Flash工厂创建当前STR Lab任务。原生入口没有调用Flash环境工厂。使用冻结100M actor产生动作带，wrapper重放相同动作，不学习。

配置：KUKA+Sharpa（不是Wuji），arm EMA=1、hand EMA=.1，noDR，固定 tolerance=0.02905653603374958。保留完整1200资产模板池，本次实际选择六类32个实例。每环境600个policy step。

检查结果：

| 项目 | 结果 |
| --- | --- |
| 59类数组：初始/逐步obs、action、关节target、实际关节/物体/goal、reward分项、reset、terminated/truncated、final_obs | 逐元素最大差异0 |
| 配置、资产序列、actor SHA、动作带SHA、初始Python/NumPy/Torch CPU/CUDA RNG | 全一致 |
| 实际覆盖 | 30次terminated、11次truncated、33次goal hit、41条有效final_obs |
| final_obs与新reset观测 | 41/41不同，包括全部11次超时 |
| 从reset前状态独立重算7组terminal观测字段 | 最大误差2.384e-7 |

因跨入口已经完全相等，没有额外运行native-self来消除“跨入口差异”的歧义；不能说该项运行过。

CPU回归还通过26个测试和10个subtests：actor/critic切片、timeout final_obs、终止mask、n-step discount、reset/reward和课程配置等。真实仿真验证与CPU测试分别记录，不混称。

**范围限制：**这是当前Lab/config/轨迹的验证，不是官方legacy Gym等价证明，也不覆盖全1200资产、开启DR的所有分支、训练初始化特有的随机episode计数或长期收敛。

证据：
- [接口详细说明](interface/README.md)
- [逐字段比较](interface/native_vs_wrapper.json)
- [控制与final_obs独立分析](interface/control_analysis.json)

## 2. 经验数据：不是完全没有成功经历

扫描100M与500M checkpoint保留的replay，各10,000,000行；不是累计全部训练历史。

| replay记录处于什么状态 | 100M | 500M |
| --- | ---: | ---: |
| 回合内曾触发抬升标志 | 84.44% | 83.97% |
| 此回合已经完成≥1个目标 | 57.79% | 41.84% |
| 已完成≥5个目标 | 28.79% | 14.16% |
| 已完成≥10个目标 | 14.86% | 4.63% |
| 已完成≥20个目标 | 4.88% | 0.65% |

**这是replay行占比，不是episode成功率。** n=3记录重叠、长回合贡献更多行；100M/500M的训练tolerance不同，不能据此直接断言相同难度下退化。

字段依据当前obs定义计算：successes=round(expm1(观测字段))，不是凭reward猜成功。lifted_object是回合内锁存标志，不等于现在仍抓稳。

全量未见超出[-1,1]的action或term/trunc同时为真。每份抽2,048行，actor与critic对应字段最大差异0。抽样近饱和动作分量比例为10.17%/12.82%。

限制：replay没有完整episode ID、接触力、初始物体z与明确类别，不能从它可靠重建“抓稳后搬运”的完整轨迹数量，也不能从scale唯一识别物体类型。

证据：[replay结果](replay/result.json)、replay/100M_coverage.json、replay/500M_coverage.json。

## 3. 实际行为：500M不是完全不会，但远未稳定

保持64个相同初始状态、相同物体URDF内容、同一随机seed、同一tolerance=0.02905653603374958。每模式最多1200步/20秒，只统计第一次回合；模型digest前后不变。

| 模式 | checkpoint | 至少达到1个目标的初始回合 | 目标到达总数 | 物体高于初始10cm且连续≥1秒 |
| --- | --- | ---: | ---: | ---: |
| 确定性 | 100M | 16/64 | 67 | 25/64 |
| 确定性 | 500M | 22/64 | 89 | 25/64 |
| Flash原生随机 | 100M | 14/64 | 45 | 22/64 |
| Flash原生随机 | 500M | 17/64 | 79 | 22/64 |

“至少1目标”不是完整50目标链成功率；高度不是接触抓稳证明。100M确定性有1个初始回合在20秒处被诊断截断，其余上表模式的初始回合已结束。视频中的后续自动reset回合不进入此表。

同模式两个checkpoint的初始obs、关节、object、goal、prev/cur targets最大差异均为0；64个资产URDF内容SHA全相同。临时文件目录名不同不等于资产不同。

配对bootstrap：500M−100M的每初始回合目标数差，确定性+.344，95%区间[-.625,1.188]；随机+.531，区间[-.125,1.297]。**只是暂选500M，不是统计上已证实领先，更不是全checkpoint最优。**

证据：[对照结果](matched_play/comparison.json)、[500M视频](matched_play/500M/paired_play.mp4)、[100M视频](matched_play/100M/paired_play.mp4)。视频均40秒，前20秒确定性、后20秒随机，4个预选视角。

## 4. Q：区分“数值小”和“确实判断错”

CPU用两份replay各256个状态组成同一批512状态，冻结actor/critic/RMS：

| 指标 | 100M | 500M |
| --- | ---: | ---: |
| 每状态8个合法动作的Q极差，中位数 | .005410 | .002296 |
| 均值动作处arm dQ/da范数，中位数 | .002829 | .001110 |
| Actor pre-tanh std均值 | .6465 | .6857 |
| reward归一化分母 | 1464.189 | 1464.189 |

这说明500M在这批状态上的典型动作区分幅度更小，基础高斯噪声不为零。**Q差异小本身不是Q错误或训练失败根因的证明。**

真实Q测试：500M，32环境，prefix 0/60/120，七种单步动作（mean、mean重复、zero、arm Q+/−、hand Q+/−），然后接同一冻结随机策略及共同标准化噪声，最长600步。不用Q自己bootstrap来验证Q自己。

已得到的限制：
- prefix0所有32状态通过可见状态和RNG配对；初始最大差异5.96e-8。
- 然而同动作mean重复600步后，有限soft return差最大.7436、目标数差最大6；这是重复对照自身的差异，不能都归给动作改变。
- prefix60/120的状态与RNG检查均失败，相关分支有效配对数0，**全部排除因果解释**；未放宽1e-5门限。
- 超时/诊断时长截断的未知尾部折扣是.99^600=.002405；有限soft return仍不是真Q。
- 因此报告里的原始方向一致率不是Q准确率，不能用它宣布“critic坏了”或“critic没问题”。

8环境×30步冒烟只证明协议能运行、动作扰动确实进入控制；全部轨迹时长截断，不能用来校准Q。

最后补充的hard-reset重复验证已完成：标准sim.reset(soft=False)后重施原材料，核验17项材料/质量/惯量/PD/限位等参数与原初始化误差0、joint/body顺序一致，再seed与task reset。32环境重放同一120步动作，每步状态和RNG通过1e-5门限，终止/超时事件相同，reward最大误差9.06e-6，模型未变。\n\n这找到了改善短程反事实重放的诊断协议；不锁定具体PhysX内部缓存，不证明600步随机延续可重复，也不说明普通训练reset有bug。完整Q七分支没有据此重跑。证据：[硬复位解释](q_hardreset_baseline/interpretation.md)、[结果](q_hardreset_baseline/summary.json)。

证据：[长对照原始结果](q_probe_500m/summary.json)、[短测试解释](q_smoke_500m/smoke_interpretation.md)。

## 5. 策略与控制：目标传递正确，跟踪误差需分开理解

直接运行生产apply_action_pipeline，独立NumPy公式检查；CPU4测试通过，GPU556,800个标量动作合法且有限。

- policy dt=2×1/120=1/60秒；每步实际两次physics控制调用。
- arm：prev_target+1.5×dt×action，再限幅和平滑。
- hand：动作映射为绝对关节位置，再EMA=.1。
- 当前arm EMA=1时最大单步目标增量约.025rad；原EMA=.1时约.0025rad。
- raw→clip→独立公式→任务target→PhysX target误差全部0，target限位越界0。

这个10倍速度差是已有arm1实验明确改过的设置，不是wrapper暗中放大。这也意味着当前实验并非“控制设置与官方完全相同，只换算法”。

取reset前真实关节位置，target跟踪绝对误差：arm均值.0303rad/P95 .1022；hand均值.1476rad/P95 .7425。target不是瞬时必达位置，接触阻挡或动力学滞后会造成误差；缺少接触/力矩记录，不能仅凭这些数值判Kp/Kd错误或力矩饱和。

## 6. 两条本地播放命令

在abao图形桌面的终端：

```bash
cd /home/abao/flashsac-robotics
bash scripts/play_str_behavior.sh best
```

关掉第一个窗口后：

```bash
bash scripts/play_str_behavior.sh official
```

不用激活conda，两个都是KUKA+Sharpa、1环境、不训练。best暂选500M，使用上面的严格固定tolerance；完整1200模板池启动约5分钟，不要重复开第二份。

official使用官方config/model、原生PPO/SAPG的LSTM与归一化，播放claw_hammer/swing_down完整人类轨迹。**其goal协议与Flash随机训练目标不同，仅看行为，不能直接比较成功率。** 两者均为当前Isaac Lab，不是legacy Gym。

本次发现并修复了播放依赖路径：runtime原先解析到play2perfect的rl_games，而非STR自带版本。现在仅在播放进程PYTHONPATH前置STR的rl_games；没有全局重装依赖。这是已确认的播放入口问题，未证明它造成Flash训练失败。

官方GUI单回合已实测：37/37 goals、报告10.6秒、正常退出。Flash离屏渲染已完成；best这条GUI命令本身和手动关窗尚未单独实测。

[播放详细说明](playback_notes.md) · [官方CPU核验](official_cpu_check_fixed_path.json) · [官方GUI核验](official_gui_smoke.json)

## 7. 尚不能给出的结论

没有证据支持直接归因于“缺LSTM”，也不能仅凭小Q梯度归因于critic坏。当前应优先完成可靠的同状态Q对照，再决定是否需要价值学习或actor/历史信息的A/B；不据此盲目继续500M预算。

本轮诊断使用token-saviour与Ponytail：复用已有rollout、单独补诊断与播放入口，没有借排查扩大修改生产训练框架。


