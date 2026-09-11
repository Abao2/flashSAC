# FlashSAC × STR：夜间因果诊断（2026-09-08）

## 最终结论（2026-09-09 凌晨完成）

**原生 FlashSAC 前馈网络能学会当前固定任务，不必先加 LSTM、BC 或重启 replay；但训练种子和评测采样方式显著影响结果。** 这里的“成功”只指原任务累计达标判据，不是稳定抓握、完整 STR 复现或真机可部署。

三条独立训练均从随机初始化连续运行60,002,304 transitions，1024env、replay10M、固定eraser/初态/目标、DR关闭、state162、arm系数1/hand系数.1；没有中途加载checkpoint、replay或BC。保持原生Flash网络、原任务reward和成功阈值。

| 训练seed | 50M确定性 | 50M原生随机 | 60M确定性 | 60M原生随机 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0/64 | 0/128 | 62/64 | 127/128 |
| 1 | 1/64 | 40/128 | 1/64 | 88/128 |
| 2 | 64/64 | 125/128 | 64/64 | 128/128 |

每条60M训练在本机实际约27–29分钟（实际48GB显存4090，占约31.3GB；不是标准24GB卡的容量承诺，也不是严格硬件benchmark）。全部15项训练/评测完成，36个checkpoint组件有限、计数正确、17项源码hash一致；在线TB窗口与上表的独立冻结评测分开统计。[审计与原始结果](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m/audit.md)。

![三训练种子的真实独立评测](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m/reproducibility.png)

## 回到最初的问题：不只是探索和LSTM

| 环节 | 本轮证据 | 能下的结论 |
| --- | --- | --- |
| 环境接口与控制 | 官方策略在匹配控制下成功；raw/wrapper3135条transition逐项相同 | 此固定任务入口可完成；不是全部接口/任务无误的证明 |
| 观测与网络表达 | 当前162前馈BC成功，后来原生前馈SAC从零也成功 | 此固定、无DR任务不必先加LSTM；162含历史摘要，不能外推延迟/部分观测任务 |
| 训练预算与种子 | 10M三seed对照全失败；连续60M有两条确定性高成功、另一条很低 | 不能根据早期10M宣布算法不行，也不能保证每个seed都训成 |
| 采样与闭环行为 | seed1同一模型det1/64、原生随机88/128；冻结噪声倍率.25/.5/.75/1/1.5对应33/52/69/88/103（各128） | 推理采样确实影响此模型结果；不是“增大训练探索必然有效”的因果证明，也没证明1.5最优 |
| Critic/Actor学习 | 成功BC在原生RL更新中丢技能；降alpha消除所查下界裁剪仍失败；新SAC成功时alpha反而更高 | 不是“仅熵太大”或“加预热就够”；梯度探针不能代替真实Q准确性检验 |
| 成功定义与物理质量 | 按固定reward四关键点≤30mm累计10帧判成功，成功后立即结束 | 已验证任务达标；尚未验证成功后的保持、接触力闭合或部署安全 |

### seed1为什么随机能成功，确定性却很差？

63个确定性失败回合都曾进入目标范围，最佳关键点误差1.32–1.43cm；但仅累计7–9帧（其中55回合为8帧），之后离开并超时。随机成功88回合中，72个靠再次进入补齐10帧。最初32个回合的观测完全相同，确定性1/32、随机19/32；模型权重未变。因此不是“完全抬不起来/到不了目标”，也不能仅据此断言多峰策略或均值抵消。

![同一起点下的一个确定性失败与随机成功](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m/seed1_sampling_trace.png)

追加6项采样测试全部完成；960个记录回合（含复用基线）逐帧验证成功、奖励、终止和模型不变性。独立每步噪声82/128、原生重复噪声88/128，单次差异不足以宣称某种时间规则普遍更好。[采样对照及限制](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m/seed1_sampling_grid/analysis.md)。

### 成功真实成立，但还不是稳定抓握

seed2的192个评测回合独立重算通过：首次累计第10帧true terminated、无timeout冒充成功，均有一次+300抬升奖和累计+1000目标奖；没有用物体中心距离替代原reward关键点判据。

但seed2达标末步仍有约0.21–0.25m/s运动；确定性连续10个near样本也只覆盖约0.15秒。**没有成功后的保持记录，不能说已经停稳或可以上真机。** seed2是看过三个结果后选的演示模型，不能用其100%替代三个seed的整体表现。[逐帧成功与动作质量核验](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m/seed2_success_validation.md)。

## 模型、曲线和复用入口

建议先查看连续训练seed2：[checkpoint目录](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m/models/seed2/step58596)、[完整训练配置](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m/configs/seed2.json)。这是原KUKA+Sharpa，不是Wuji；没有把结果外推到另一embodiment。

本机看三条曲线，用这一整行，不需要RUN变量或激活conda：

```bash
/home/abao/play2perfect/.venv_isaacsim/bin/tensorboard --logdir=/home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m/runs --host=127.0.0.1 --port=6008
```

浏览器打开 `http://127.0.0.1:6008`。主要看 `episode/final/all_goals_hit`、`episode/return` 和 `episode/cumulative/lift_bonus_rew`；训练窗口仍不能代替上面的checkpoint评测。若6008已被占用，换一个空闲端口。

[复现实验命令清单](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m/manifest.json)记录实际执行argv；输出路径已有结果，重跑必须使用新输出目录，避免覆盖。早先50M+冷replay5M成功模型及所有负结果也原样保留。

## 尚未解决、下一步值得做什么

- 尚未定位全部种子差异的唯一根因；没有准确的反事实Q回报校准，不能用“Q梯度大”直接宣布Q正确。
- 当前固定目标不等于官方完整初态/物体/轨迹分布，更不等于论文24任务或Wuji迁移结果。
- 接下来更有价值的是看动作视频，并单独定义和测试成功后持续保持；若要进入真机路线，再做受控的初态扰动、DR/延迟和控制一致性验证。本轮没有继续启动这些新范围的训练。
- 夜间计划中的GPU训练与评测已全部完成，进程已自行退出；没有Git提交/推送，没有操作真机或远端任务。

下文是可追溯的实验过程，按时间保留当时判断；其中旧的“正在/待验证”不代表当前状态。最终结论以上述摘要及各队列JSON为准。

## 目标与边界

用户授权在本机充分测试，2026-09-09 查看结果。目标是区分：接口/任务、奖励、数据覆盖、观测与记忆、Critic 估值、Actor 更新；不预设探索或 LSTM 是病因。

- 仅本机仿真与离线分析，不控制真机、不操作远端训练、不提交或推送 Git。
- 保留两仓库全部既有未提交修改、checkpoint 和历史日志。
- 所有新测试使用独立输出目录；不覆盖原训练，不把诊断性干预叫论文复现。
- 每组固定任务/seed/控制配置，变更因素写入配置。涉及训练的对照记录 transitions、更新次数和学习率日程。
- GPU 阶段串行调度；启动前检查显存与其他进程，不终止不属于本轮的进程。
- 计划在 2026-09-09 上午收束并汇总；不以长时间运行代替有鉴别力的证据。

## 路径

- 代码：`/home/abao/flashsac-robotics`
- STR：`/home/abao/simtoolreal`
- 输出：`/home/abao/flashsac-robotics/diagnostics/overnight_20260908`
- 进度：输出目录 `STATUS.md`
- 本文作为最终结论入口，测试进行中不把计划写成结果。

## 已有基线（不是今晚新结果）

本轮前的 state162 / fixed eraser / seed0 / 10M 诊断，最后保存 9,994,240 transitions；100 回合固定起点确定性评测成功 0/100、超过 10cm 抬升 0/100、位置误差约 15.9cm。详见 `docs/str_state_teacher.md`。

起点 Q 对动作区分弱只是线索：Q 约 4.675，不能仅依据绝对差值或一个梯度范数宣称根因。随机探索出现 lift bonus 也不能证明稳定抓握。

## 测试序列

1. **正对照/接口**：官方 checkpoint 在相同固定任务上比较 raw 与 Flash wrapper。保留原 140 维 actor 输入、归一化与 LSTM；按字段从诊断 162 维提取，不把162直接喂给官方策略。比较动作、观测、分项奖励、终止/timeout/final_obs和实际回合结果。若两入口均失败，先区分固定任务分布与接口问题。
2. **冻结策略探索**：当前 Flash checkpoint 不更新参数，分别测确定性、原生随机、幅度、噪声时间连续性；每项干预单独标记。采集动作/控制尺度及分阶段指标；物理接触未直接观测时只能报告持握代理指标。
3. **信息与表达能力**：用一致轨迹数据做按整回合留出的当前帧/容量对照/历史预测；若得到成功示范，再做前馈 BC 与历史对照，并闭环评测。预测改善不等于 LSTM 必要，BC 的 MSE 不等于闭环成功。
4. **Critic/Actor**：在各物理阶段检查 Q 动作差异、分布边界、未投影 TD 目标越界质量、Q与熵的Actor梯度。分清本轮rollout数据与训练replay；这些离线量不直接证明真实回报排名。
5. **有证据的短训练 A/B**：仅在前面定位了有效方向后追加匹配对照，避免同时改探索、奖励与网络；结果不确定时保留负结果与限制。

## 结论判据

- raw 成功、wrapper失败且输入/控制有差异：接口优先。
- 冻结Actor只改采样即增加稳定持握：支持覆盖问题，不代表已经学进策略。
- 当前162维前馈BC能闭环成功：在该任务范围内说明前馈表达可行，不保证RL从零能学到。
- 历史改善但容量/数据未控制：不能归因记忆。
- Q预测差异小但真实动作回报也相近：不能称Critic错误。
- 成功演示/有利状态注入改善：只说明干预有帮助，不是原任务从零结果。

## 今晚结果

以下为 2026-09-08 21:15 已完成结果，仍在执行后续测试。回合数是同一固定任务的重复，不是跨物体/初始分布泛化。

### 接口与官方策略控制对照

原入口和 wrapper 的 3,135 条 transition、20 个数组逐元素完全一致，最大差值0，包括obs、next_obs、执行动作、控制目标、reward、terminated、truncated。证据：`diagnostics/overnight_20260908/official_wrapper_raw_parity.json`。raw旁路的是wrapper.step，仍共享底层环境初始化；不能据此宣称全部移植已证明。相关CPU回归24 passed。

| arm系数 | hand系数 | 官方checkpoint最终目标完成 |
| --- | --- | --- |
| 1.0 | 1.0 | 0/16（8次曾超过10cm，但随后掉落） |
| 0.1 | 1.0 | 0/16 |
| 1.0 | 0.1 | 16/16 |
| 0.1 | 0.1 | 16/16，raw/wrapper均完成，约59steps |

另外官方随机策略在.1/.1完成128/128，加入只读teacher标签hook后仍一致。源码action_utils.py中arm增量为 `arm_alpha*1.5*dt*action`，.1→1同时放大增量10倍；hand是绝对关节目标平滑，.1→1去掉过渡。官方Normal→clip产生大量饱和本身不是接口错误。

结论：当前wrapper能完成此任务，官方策略对手指控制设置敏感。但Flash原先就在1.0控制下从零学习，不能直接用现成PPO策略的控制失配解释Flash失败；需从零训练A/B。

### 前馈Actor确实能闭环完成，不必先加LSTM

用官方.1/.1的128个成功随机回合采样状态，从同一次forward读取确定性教师均值动作（没有额外推进LSTM）。实际执行随机动作另存，两者RMSE=.91055，不能混用。原生Flash Actor，width128、2blocks、3000次BC更新；按完整回合留出25%；评测控制仍.1/.1：

| 输入 | 最终目标完成 | 平均完成steps | mean raw return |
| --- | --- | --- | --- |
| 仅当前162维，无LSTM | 64/64 | 62.0 | 1388.04 |
| 当前帧重复8次，容量对照 | 64/64 | 71.67 | 1394.75 |
| 真实8帧历史 | 64/64 | 60.0 | 1386.76 |

独立核对每回合：恰好累计10次keypoint误差≤.03m、+1000目标bonus、+300抬升bonus、terminated而非timeout；并非拿持握代理充当成功。当前模型输入162、293374参数；任意改前7帧动作变化0。注意当前162含previous targets、trackers和progress，不等于完全没有历史摘要。

这是BC固定任务结果，不是FlashSAC从零RL成功、论文全任务复现或真机可部署结论。三组均成功，不能用小幅步数差声称历史有优势。

保留负结果：先前只用16个近乎重复的确定性回合BC，验证动作RMSE虽.033，闭环却0/32完成；都曾超过10cm但随后掉落。扩大状态覆盖、采用正确教师标签后才完成，不能把离线拟合当闭环可靠性。

成功随机数据留出动力学normalized MSE：当前.14065、重复当前.13950、8帧历史.18557，暂无短历史优势。它不排除其他记忆设计、DR/delay或完整分布的需求。

### 探索与Critic：继续定位，不抢先下结论

已有10M Flash actor冻结后，确定性128回合和原生随机seed0的128回合均无最终目标、无任务10cm抬升。其他幅度/持续时间/控制条件仍在测试，不能据这两组排除罕见有效探索。

起点top-five atom质量高不等于该处TD目标正越界，也不能说所有状态Q都平、熵始终压过Q。高奖励事件另做逐事件检查；这些是新rollout探针而非历史replay，更不是精确实测Q。配对首动作实验会比较有限实际回报并明确未知timeout尾项。

## 后台与后续

- phase1共24项全部完成；命令与状态在输出目录`queue_phase1.json`、`queue_phase1_status.json`。
- phase2已实际启动：`flashsac-str-control-ab-20260908.service`，arm均1.0，只改hand 1.0/.1，3seed各10,000,384 transitions从零训练；网络、奖励、replay、学习率日程匹配。共6train+12eval+6critic串行任务；首训练21:25启动，显存约31.3GB。配置差异断言与41文件源码hash已保存。
- 每小时续查heartbeat `flashsac-str`，先读本文与STATUS，避免重复；2026-09-09 09:00本地时间停止新增耗时工作并汇总。
- 队列有截止时间、单任务timeout、显存检查和锁；不终止其他任务。结果以JSON与日志为准。

### 21:30新增：冻结探索结果

11条件各128回合，最终目标和持续抬升近掌代理均为0。将固定噪声持续时间设为1/4/16步，动作平均幅度基本相同，但曾离桌回合0/7/20，超过任务10cm回合0/0/2；说明时间连续性改变覆盖，仍可能是拍打/抛起。仅把幅度提高1.5倍没有解决抓取。

旧actor仅在评测切换控制.1/.1时，reward升高但也没有成功或离桌，所以reward上升不能直接代表技能进步。严格结论等从零控制A/B。

首动作实际回报7分支×32配对重复×2seed完成，全部无目标成功，后续使用当前actor而非专家；终止前final_obs、独立高斯目标策略、首动作无熵项已检查。有限600步回报未补未知尾项，不能直接充当精确Q。详见`diagnostics/overnight_20260908/critic_findings.md`。

### 查看本轮训练曲线

TensorBoard日志根目录：`/home/abao/flashsac-robotics/diagnostics/overnight_20260908/control_ab/runs/control_ab`。本机执行下面这**一行**即可（端口若占用可改）：

```bash
/home/abao/play2perfect/.venv_isaacsim/bin/tensorboard --logdir=/home/abao/flashsac-robotics/diagnostics/overnight_20260908/control_ab/runs/control_ab --host=127.0.0.1 --port=6010
```

使用同一个训练transitions横轴；训练随机策略reward/成功日志与最终单独det/stoch128回合成功率分开解释。尚未训练的seed不会凭空出现曲线。

### 21:37自动续查：第一个新训练基线完成

arm=1、hand=1、seed0从零10,000,384 transitions已完成，训练进程404s、exit0，step9766已保存；此wallclock有共享GPU条件，非独占性能基准。

| 评测方式 | 最终目标 | 任务>10cm抬升 | 曾离桌 | 持续抬升近掌代理 | raw return |
| --- | --- | --- | --- | --- | --- |
| deterministic | 0/128 | 0/128 | 0/128 | 0/128 | 617.66 |
| native stochastic | 0/128 | 2/128 | 10/128 | 0/128 | 434.87 |

这是新训练结果，不是旧模型重测。单seed与旧基线定性一致；对应hand=.1仍在训练，另外2seed尚待执行，不能提前判断A/B。服务正常，未重复启动或改核心训练代码。精确结果在control_ab/evaluations/arm1_hand1_seed0。

独立检查step9766的6个checkpoint文件完整，浮点网络/optimizer/normalizer张量全部有限；计数10,000,384确认完整预算，agent update计数19,338（学习率日程19,532是配置时长，不等于warmup后实际更新数）。旧评测16env、本轮32env，且normalizer不同，因此小幅指标变化不是严格配对改善。常见状态弱Q梯度、罕见高奖励局部裁剪的定性模式仍在；详见critic_findings.md第9节。

### 22:36续查：6次10M手指平滑A/B全部完成

两组均det/stoch各0/384最终目标，确定性均0/384抬升。随机抬升hand1为3/384，hand.1为8/384；持续抬升近掌代理1/384 vs6/384，样本少且仅3个训练seed，不称显著改善。手部平滑组reward增加的重要来源是手速度惩罚减少，不等于技能成功。

全部6个训练的TB配置、checkpoint完整性及1266浮点tensor有限性通过；41个源码SHA一致。日志有极少部分近目标奖励但没有完整目标成功。详细结果与6条曲线统计在`control_ab/analysis.md`、`control_ab/compact.json`。只改hand平滑未解决10M内学习，仍不能推出更长预算永远失败。

### 23:06：下一步先排除BC→SAC桥接的随机头混杂

原生前馈BC的动作均值/BN直接转入native Flash checkpoint后，det128/128；但BC没训练std头，开启原生随机采样即0/128，尚未发生任何SAC更新。这个失败属于新诊断初始化，不是原从零FlashSAC训练失败的新根因。

冻结同一模型，仅将采样pre-tanh噪声乘.1或.01，均128/128成功。因此留存实验必须先校准随机头，并让原生随机采样也通过；不能把原先就不可靠的随机策略训练后失败归因更新。标准差头校准保持均值与BN不变，保存独立产物，非修改核心算法。

后续预定：同一个通过冻结评测的BC初始策略，直接SAC更新 vs先2000次critic-only再正常更新，记录早期快照。另加50M从零预算对照，保持前10M原日程，避免仅凭短预算下结论。当前仍在预检，未通过门槛前不计为已启动实验。

### 23:47：校准验证通过，技能留存和预算对照已启动/排队

只校准原生std头，两种方法、目标std .05/.15，闭环native随机成功分别为eigen005 128/128、eigen015 127/128、Adam005 128/128、Adam015 121/128。这些数字是固定任务结果；.05/.15是拟合目标，不是每状态恒定sigma。selected eigen005另测seed1 native、seed0逐步独立噪声，均128/128。均值、torso和BN保持不变。

新CPU审计还表明：专家同一2048状态上，原BC条件tanh熵−179.7、校准005为−142.1，但原BC实际动作更抖。微分熵受边界压缩和各维sigma差异影响，不能把“低熵”等同安静，也不能把alpha当噪声标准差。详见`warmstart/policy_distribution_audit.md`。

实际后台顺序：

1. `flashsac-str-retention-20260908`：同一个通过门槛的BC，immediate vs critic-only2000，seed0各10M，34项含快照det/stoch与CPU critic。首个immediate已完整19338次network调用，最终checkpoint/audit完成。尚未据此给出技能留存结论。
2. `flashsac-str-budget50m-20260908`：等待1结束，两组hand1/.1各从零50M，固定原前10M学习率日程，连续不重启replay，每10M保存评测，26项。
3. `flashsac-str-startup-baseline-20260908`：等待2结束，冻结selected BC，仅复现原训练的随机初始progress和第一步随机动作，分初始批次与后续回合。用来区分启动分布和学习影响。

每组仍是独立目录、同一原网络/reward/终止流程。critic warmup同时改变后续采集的数据、Actor更新次数及对应LR位置；它不是固定离线数据上“只改Q质量”的严格因果干预。低熵BC在native最大熵目标下也未必是偏好的策略，必须结合早期参数/BN、目标裁剪、熵与真实成功变化解读。

23:53核对实际执行argv：retention的det64使用1300步、stoch128使用2500步，分别足够覆盖600步timeout的2/4波回合；此前将随机评测也视为1300的担心不符合实际执行记录。汇总仍检查真实完成/请求回合数，不把“队列exit0”当全部回合完成；补测生成器当前无不足额项。

### 23:53：技能确实在早期更新中丢失，尚待warmup对照定位机制

immediate分支第5次Actor更新后，确定性仍64/64，但native随机降至95/128；第50次Actor更新后，两种评测分别0/64、0/128。起点update0与BC完全一致，所以这不是checkpoint加载时就改变了均值。

同一512专家状态上，将更新后参数配回原BN，可以复现大部分早期动作/std变化；只换BN不能复现，故不是仅有BN漂移。第5步后实际std显著增加，之后均值也明显偏离成功策略。初始专家批次熵项原始参数梯度范数约为Q项9.6倍，但这是固定专家探针，不是历史replay批次或Adam实际更新贡献，不能据此宣布唯一根因。证据在`warmstart/early_update_audit.md`；正在进行的critic-only2000对照将进一步检验预热是否有帮助。

### 00:10：预热仍失败，并发现soft-return与Q下界的冲突

两组各10M及全部34项完成。critic-only2000期间Actor、BN、temperature和对应优化器完全冻结，快照均det64/64、native128/128；恢复更新后，下一已测快照update4000（实际Actor999步）均0目标，最终10M也为0。我们未在1–998次Actor更新之间测到失效边界，不能称它恰好在第999步才失效。

![技能留存对照：横轴为真实Actor优化次数](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/warmstart/retention_actor_steps.png)

预热结束时策略明明成功，但Q已经几乎贴着可表示区间的**−5下界**。在冻结checkpoint的native rollout常见阶段探针中，约99.66%–99.96%的目标概率质量低于−5；固定专家状态的常见阶段约98.6%–99.2%。例如native起点：未投影目标期望−6.3269，被投影至−4.99946。这是实际目标投影诊断，不只是看Q绝对值猜测。

原因线索是量级：该策略动作集中且许多维接近边界，真实tanh微分熵很负；alpha=.01带来约−1.2到−1.5的每步soft-value项，而普通步骤的归一化任务reward约.003–.006。到达目标的真正终止样本不bootstrap，没有这项，必须分开解释。此时“先把Q训好”实际是在拟合大量被下界裁剪的目标，不能将它等同于获得了准确的成功技能价值。

这直接说明**本次低熵BC初始化与当前最大熵权重/固定Q区间不匹配**，但尚不证明它是原先从零训练失败的主要原因，更不证明仅降低alpha就会成功。新短对照拟比较初始alpha=.01/.0001/.000001与是否预热2000次的交叉组合，保持网络、reward、控制、初始BC和学习率日程不变，使用独立checkpoint/目录；温度仍原生自适应。此干预同时改变Actor熵压力与Critic目标，不把效果强行归因其中一条路径。

00:14追加：该六组对照已由主任务审核并排队，服务`flashsac-str-entropy-ab-20260909`明确等待现有startup队列完成后才使用GPU。每组2,000,896 transitions，预计3714次network调用，保存解冻后第1/5/50次附近快照；.01基线也在同样短预算重跑。配置独立重解析、初始组件逐字节/逐tensor对照和CPU测试通过。当前尚未轮到GPU，不提前给出低alpha效果结论。

进一步阅读：[warmup同状态对照](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/warmstart/warmup_analysis.md)、[折扣熵与reward量级审计](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/warmstart/retention_mechanism_audit.md)、[50M预算进度与曲线](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/analysis.md)。不同抽样口径的clipping比例不得混用：同一512专家状态中501条非终止样本为83.35%，reset约99.47%；另11条真正成功终止均不bootstrap，没有next-entropy进入目标。

### 01:50：长预算、低alpha和协议排查的最终结果

50M两组连续训练和20次独立checkpoint评测已完整结束。hand1.0最终raw return1169.67（det），但没有正式10cm抬升，主要来自在阈值下持续抬高的shaping。hand.1在20/30/50M能抬升，50M det64/64、stoch128/128，但目标均0；抬升后回落至初始高度以下的分别55/64、127/128。相近或更高的总reward绝不能直接当作抓取成功；gamma=.99的折扣raw return甚至与600步总return给出不同排序。

hand.1在线日志最早在42.8544M出现目标成功，49.7152M之后出现较高窗口，末尾30个vectorsteps记录80.21%；它不是一个固定策略独立测试的80%。同一最终checkpoint的det0/64、native0/128必须同时报告。真实固定size keypoint阈值3cm，64个det回合的最佳误差仍在3.77–5.10cm，中心偶尔近于3cm也不算全keypoint达标。

![50M冻结策略的一个真实回合](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/frozen50m_episode0_trace.png)

已完成以下反证检查：

| 假设/检查 | 实测结果 | 可以排除到哪里 |
| --- | --- | --- |
| Actor-only加载与原生全量加载不同 | CPU同32/1024状态，mean/std、确定性及12次配对随机动作完全一致 | 没发现加载路径或`.eval()`模式导致的差异；不代表所有CUDA数值路径相同 |
| 成功日志算错/跨reset旧值 | 源码与CPU加权fixture通过，当前success在reset前取值，按done_count加权 | 无明显日志或平均错误，不把末窗当独立成功率 |
| 32环境和1024环境不同 | 1024普通/启动对照各2048回合，仍0目标 | 此环境数量干预没有恢复效果 |
| 启动方式、seed或训练精度设置 | 单独对照均0目标，实际flags已记录 | 这些因素单独不能解释80%与0的差别 |
| 首次obs中的历史reward不一致 | 0与真实成功terminal中位数100.1402两组各128回合，均0目标 | 此字段影响动作，但仅改变首次历史reward未解决问题 |

六组初始alpha×预热对照也已66/66完成：低alpha预热Q不再大量被−5截断，但最终仍全部det0/64、stoch0/128。预热后三种alpha在5次Actor更新时仍det64/64，约50次后全0（其中.01有一次AMP跳过，实际49步）；因此它有短期保护，不能说完全没用，更不能说Q边界或熵是唯一原因。[完整六组结论](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/entropy_ab/conclusions.md)。

当时下一项是有界的原生入口对照，以下记录其已完成结果。

### 02:55：原生SAC固定任务成功，独立重算通过

两分支都加载同一从零50M的完整checkpoint、使用新的replay/仿真进程，再收集5,000,192 transitions。唯一设计干预是是否冻结Actor与temperature；Critic在两分支仍更新，后续数据也因此不同。

| 分支 | 新Actor实际更新 | 训练末窗口goal | 最终冻结det | 最终冻结native随机 |
| --- | ---: | ---: | ---: | ---: |
| Actor/temperature冻结 | 0 | 0%（97窗口全部0） | 0/64 | 0/128 |
| 正常原生更新 | 4,786 | 98.04% | 64/64 | 110/128 |

额外随机评测种子1/2分别218/256、229/256，三个种子合计557/640=87.03%。它们是同一个训练模型的rollout种子，不是三个独立训练种子；固定确定性条件换seed仍得到同样64/64，不能当成独立泛化证据。

**不是只看success标志**：独立NumPy几何计算恢复物体中心和四元数，使用原reward的固定尺寸四个角点重建最大目标误差，逐步累计≤30mm的帧。192个初始评测回合中174个成功均恰好在第10个累计达标帧true terminated，全部有+300一次性lift奖和累计+1000目标奖，timeout没有算成功。原生随机其余18回合都是600步timeout，只有4–9次达标。模型前后digest一致，成功来源与代码/配置hash已保存。

重要限制：**原判据是累计，不是连续保持**。成功中的53/64确定性回合、79/110随机回合没有连续10帧达标。因此“满足此任务goal”是准确表述，“已经稳定抓握/保持/可部署”仍不成立。模型是原生Flash前馈网络，从零50M＋冷replay继续5M，无BC；但不能叫不间断55M，不能凭此区分额外学习预算与replay刷新各自的贡献。

这一结果改变了判断：本任务并非FlashSAC原则上不能学习，也不需要先换成STR的LSTM网络才有可能成功；已有可加载、可独立复测的成功策略。之前10M失败、BC在RL更新中退化、原50M冻结策略失败仍是真实负结果，它们对应不同训练阶段/初始化，不能混成同一个原因。

当前仍需回答的是：**从零连续训练是否可重复获得这个结果、何时出现以及会不会再退化**。已启动repro60m三个训练种子，每条从随机初始化连续60,002,304 transitions，保持原hand=.1/arm=1、网络、奖励、replay10M和固定LR日程19532；50M/60M保存后独立评测。没有中途清空replay，没有成功示范。它是下一步复现性证据，不提前报结果。

真实CUDA编译/非编译前向也已测完：配对采样最大action差约.00512、全部参数/BN不变。不是完全逐位一致，尚不能据此断言微小误差对接触没有影响；但新checkpoint已在原生训练与非编译独立评测两边成功，说明两入口并非必然无法兼容。[前向差异记录](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/compiled_forward_probe/analysis.md)。

当前结果路径：

- 成功checkpoint：`/home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/native_entry_check/training/normal/models/final/step4883`
- 成功模型TB：`/home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/native_entry_check/training/normal/runs`
- 新三seed连续训练TB：`/home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m/runs`
- TB程序：`/home/abao/play2perfect/.venv_isaacsim/bin/tensorboard`，参数仍为`--logdir=上面对应路径 --host=127.0.0.1 --port=6008`；未在本轮自动另开TB服务。

### 03:15：成功动作质量与Critic对照，不把达标说成停稳

已有成功策略的记录经逐回合有限差分核验，绝不跨reset求速度。确定性64/64满足累计判据，但只有11/64曾连续10帧near；三个随机评测种子的成功回合中，连续10帧分别31/110、63/218、52/229。确定性平均完成时间0.535秒，随机成功回合均值约1.35–1.49秒。

确定性成功最后一个控制区间的估算线速度均值0.458m/s、角速度131.9°/s；随机成功组约0.35m/s、100–103°/s。它们是16.667ms区间有限差分，不是瞬时PhysX测量，且成功立即终止，**没有任何成功后的稳定保持观察**。不能由这些速度断言下一秒必定掉落，也不能宣称已经抓稳。BC背景数据的线速度更低，但arm滤波是.1而SAC为1，不是算法单变量比较。

![一个按原判据成功的真实回合](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/native_entry_check/success_motion_trace.png)

对旧50M失败与新成功checkpoint，还在相同状态批次和相同记录/随机候选动作上比较了Critic：Q对随机动作的变化程度提高约3–5倍，但这只说明学到的函数敏感度变大，不等于估值更准确。Q/熵的原始Actor参数梯度比例并没有一致提高：失败轨迹样本17.79→18.31，成功轨迹条件样本89.91→81.69；这些梯度各在自己Actor的动作上计算，不是相同动作，也不是实际Adam更新贡献。

新checkpoint的alpha反而从2.07e−5升至5.62e−5，reward除数仅变动约0.62%；不能用“终于把熵权重降下来了”解释这两个最终模型的差别。所查样本都无Q下界裁剪，新模型成功支持集仍有约2.17%目标概率质量超过上界。因此之前低熵BC的下界问题不能套用到所有native SAC阶段。当前仍没有真实反事实回报校准，不能宣布Critic已完全正确或唯一根因已找到。[完整同状态对照与抽样限制](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/native_entry_check/critic_success_comparison.md)。
