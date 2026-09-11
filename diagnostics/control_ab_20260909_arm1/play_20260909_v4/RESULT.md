# 100M arm=1 checkpoint：配对短play结果

报告：2026-09-10。回放完成于2026-09-09约23:57（北京时间）。

## 结论

能让物体离开桌面，但主要仍是短暂挑起/甩起后失去物体，并非稳定抓取搬运。此前约92%的lift事件率不能解释成抓取成功率。确定性与原生随机动作都出现该问题，不支持“仅仅是评测没加噪声”这一解释；本次不证明唯一的训练根因，也没有测试更长预算的上限。

## 同一批64个初始状态

| 首episode指标 | 确定性动作 | 原生随机动作 |
|---|---:|---:|
| 完整记录的episode | 64 | 64 |
| 曾超过初始高度10cm | 62/64 | 59/64 |
| 连续超过初始高度10cm至少1秒 | 8/64 | 7/64 |
| 至少命中1个goal | 11/64（17.19%） | 10/64（15.63%） |
| goal总数 / 每episode均值 | 32 / 0.5000 | 28 / 0.4375 |
| 完成全部50个goal | 0/64 | 0/64 |
| 物体下落终止：局部z<0.1m | 54/64 | 55/64 |
| 手与物体过远终止 | 8/64 | 8/64 |
| 时间上限终止 | 2/64 | 2/64 |
| episode时长中位数 | 1.175秒 | 1.200秒 |
| 原始累计reward均值 | 912.58 | 863.60 |

随机组有1回合同时满足fall与hand_far，终止原因不是互斥类别；两组真正终止的episode并集都是62，另2个timeout。这里是最终checkpoint的首episode样本，不是训练90–100M期间的TB窗口均值，不能把两个reward数字直接当成同口径曲线对照。

“连续高于10cm”仅是几何持续时间，不是接触力验证的握持成功。达到goal的判据也未要求稳定抓握：使用原任务关键点容差与累计near-goal计数，不能把11/64解释成稳定完成整个任务。

## 回落与姿态检查

以首goal命中之前的轨迹分类，互斥计数为：

| 首goal结果 | 确定性 | 随机 |
|---|---:|---:|
| 命中首goal | 11 | 10 |
| 从未达到抬升高度、未命中 | 2 | 5 |
| 抬升后回落低位、未命中 | 48 | 49 |
| 曾抬升、未回落低位、仍未命中 | 3 | 0 |
| 未结束/截断 | 0 | 0 |

这里“回落低位”的人为诊断定义：曾Δz>0.10m之后，Δz≤0.02m连续5个控制步，且在首goal之前。它只证明高度回落，不单独证明失去接触；没有跨reset串接轨迹。随机组另有1个曾回落但后来命中的episode，仍归入命中类别。

未命中首goal的episode在结束时，位置误差中位数分别0.950m / 0.926m，关键点最大误差中位数1.032m / 1.024m。位置本身已经很远，不能据此归咎于“仅缺少手内旋转”。旋转角误差中位数129.68° / 125.73°仅供辅助观察，未用于替换官方成功判据。

原配置reset_when_dropped=false，所以summary.json中的ended_dropped=0不说明没有掉落。上表fall来自原环境z<0.1m的终止事件；回落来自重置前的真实高度轨迹。

## 视频

[paired_play.mp4](paired_play.mp4)：40秒，960×720，30fps，1200帧。

- 0–20秒为确定性，20–40秒为原生随机；仿真与视频均为1倍速。
- 四视角依次为hammer env1、spatula env6、eraser env8、brush env0，在运行前按类别首次出现选择，未按成功表现筛选。
- 视频中这些首episode多数在约1秒时已结束；画面之后继续展示原生自动reset后的回放，标签first episode=False明确标出。这些后续episode不进入上面的64样本统计。
- 0.3/0.6/0.9秒帧可见物体被挑起/离手、下落的过程；本次未装ContactSensor，不能给出可靠的接触力或“抓稳率”。

## 测试条件与防错检查

- checkpoint：../models/seed0/step48830/actor.pt；SHA256见metadata.json。
- 完整STR任务配置simtoolreal_full_arm1：KUKA+Sharpa、6类程序物体、随机初始姿态/随机目标、50-goal链、arm EMA=1、hand EMA=.1、DR关闭。不是旧的单eraser固定目标简单任务，也不是Wuji。
- 64个实际分配物体：brush24、screwdriver12、eraser12、hammer10、spatula4、marker2；覆盖所有6类，但只覆盖1200资产池中的64个，不是论文24任务评测或全资产基准。
- 每组最多1200步/20秒；只统计每env首episode，两组64个episode均在上限前结束，没有将未结束样本记成失败。
- tolerance冻结为训练当前的.075；有效keypoint阈值.075×1.5=.1125m；near-goal累计10帧，保持原任务非强制连续计数规则。
- 两模式重新seed、清空previous reward后full reset。joint位置/速度、object/goal root state、控制目标均完全一致；obs最大差2.38e-7。配对的是初始状态，不声称后续随机goal链完全相同。
- 仅加载actor；原生sample_actions(training=True)只用于随机采样，内部网络依然training=False。无learner/process_transition调用、无replay填充。
- actor及BN状态摘要前后完全一致：35bd16f7ea5eab7230fe71b2ffaa72fd9d28f13bbdf1c20bf1c2bf3c78060389。
- 在原_get_rewards返回后、自动reset之前复制物理量，没有重复推进reward/near-goal tracker。
- 录像使用进程内NumPy1.26；训练环境NumPy2.2.6及所有checkpoint未修改。运行参数STR_PLAY_NUMPY_SITE=/home/abao/play2perfect/.venv_isaacsim/lib/python3.11/site-packages。
- 正式成功运行总耗时439秒，其中场景加载约5分钟；没有追加训练。完成后检查GPU只剩用户ToDesk进程。

## 代码与失败尝试

新增scripts/play_str_full_diagnostic.py；Isaac wrapper只增加可选render_mode以开启离屏相机，默认训练行为不变。遵循最小修改原则复用原生采样和任务hook，没有改reward、控制器、reset或网络。

20项接口/评测/config回归测试通过；另做精确actor checkpoint的CPU确定性/随机采样与摘要检查、四相机小场景实测，并检查成品视频帧/时长。

保留相邻失败目录和日志，不当成实验结果：首次在初始化时主动重启修正跨模式inference_mode问题；v2稀疏Hydra配置默认项读取错误；v3录像模块报NumPy兼容错误。小场景复现定位到SyntheticData的OmniGraph dependency写入，换用已有NumPy1.26后实测通过。它们是回放工具失败，不是训练学习失败。

原始证据：summary.json、metadata.json、actual_task_config.json、env_assets.json、*_initial.npz、*_episodes.json、*_traces.npz。轨迹active掩码包含首episode终止帧，排除之后自动reset的episode。
