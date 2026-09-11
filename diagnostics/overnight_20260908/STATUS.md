# 夜间诊断进度：本轮已完成

更新时间：2026-09-09 05:03 Asia/Shanghai。

## 最终状态

- 所有本轮GPU训练/评测已完成并自行退出；只读检查无本任务systemd服务、无GPU计算进程。没有终止其他任务。
- 自动续查 `flashsac-str` 已通过应用工具暂停，返回 `status=PAUSED`；原prompt/名称/周期/目标线程保留，其他自动任务不动。不要再次自动启动旧队列。
- 主报告：`/home/abao/flashsac-robotics/docs/str_overnight_20260908.md`，顶部已重写为最终结论与复用入口；下文历史记录保留。
- repro60m 15/15完成、seed1_sampling_grid 6/6完成。三条从零连续60,002,304 transitions、1024env、replay10M、固定LR19532、arm1/hand.1、无BC/模型/replay加载。
- 60M独立det/stoch：seed0=62/64、127/128；seed1=1/64、88/128；seed2=64/64、128/128。50M依次0/64和0/128、1/64和40/128、64/64和125/128。必须保留种子差异，不能只报最好100%。
- 可供演示模型：`repro60m/models/seed2/step58596`。seed2_success_validation.md/.json独立确认192/192真实goal、累计10帧四reward关键点≤30mm、+1000goal/+300lift、true terminated无timeout冒充；checkpoint计数/所有文件有限。达标仍有约.21–.25m/s运动，没有成功后稳定保持或接触力证明。
- 三训练seed审计：`repro60m/audit.md/.json`、`analysis.md/summary.json`；17源码hash不变，36checkpoint组件有限、nativecounter116998/normalizer60002304正确。单条实际训练27–29分钟（本机48GB显存，非标准24GB容量承诺）。
- seed1采样差异：所有63个det失败到过3cm范围，但只有7–9near帧；随机多数靠再次进入补足10帧。冻结std倍率.25/.5/.75/1/1.5=33/52/69/88/103（各128）；逐步独立噪声82/128，另一rolloutseed原生88/128。报告 `repro60m/seed1_sampling_grid/analysis.md`，960含基线回合独立核验通过。是推理采样实验，不证明训练探索因果/1.5最优/稳定抓握。
- 图 `repro60m/reproducibility.png`、`repro60m/seed1_sampling_trace.png`、`repro60m/seed1_sampling_grid/success_vs_sampling.png` 均已主任务目视检查。
- 先前所有正/负结果保留：官方接口/控制、BC与历史、冻结探索、10M三seed、BC留存与alpha、50M预算、原生冷replay5M、精度/加载/历史reward/编译对照。它们不是同一初始化/预算，不能混成唯一根因。
- 未运行初稿 `scripts/diagnose_str_live_freeze.py` 明确UNRUN/UNTESTED，无队列；不得直接启动。没有为成功修改核心reward/网络或10帧阈值，没有Git提交推送，没有真机/远端操作。

## 复查命令

CPU汇总：`/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python /home/abao/flashsac-robotics/scripts/summarize_str_repro60m.py --root /home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m`。

TB路径：`/home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m/runs`；程序：`/home/abao/play2perfect/.venv_isaacsim/bin/tensorboard`。主报告有不使用RUN变量的一整行命令。

下面是历史阶段索引；旧的“正在/下一步”只表示当时状态。若用户提出新实验，再明确新范围与独立输出路径，不自动重跑。

## 状态

- 本机 RTX4090 48GB，磁盘余3.1TB。其他本机任务也会用GPU；只串行本任务、不终止其他进程，记录可用显存，不将受共享影响的wallclock当性能基准。
- 两仓库存在旧改动，全部保留。
- 官方 ckpt 已找到：`/home/abao/simtoolreal/pretrained_policy/model.pth`。
- 已完成：raw/wrapper逐transition相同；官方hand=.1恢复成功；BC当前162前馈、重复当前、8帧历史均64/64固定任务真正成功；独立重算目标误差与+1000奖励核实。详情见主报告和 memory_official_filter01_labeled_full/analysis.md。
- 负结果保留：原官方filter1为0/16；小数据BC0/32；旧Flash冻结det/native seed0各128回合均未完成。
- phase1：24/24 completed，无失败。状态queue_phase1_status.json，命令queue_phase1.json。
- phase2完成：24/24 completed，6次训练全部10,000,384 transitions、两组3seed全部最终目标0（det/stoch各0/384）。新分析control_ab/analysis.md、compact.json、report.json。全部36个checkpoint文件/1266tensor有限，41源码hash匹配。单独hand滤波不足以在此预算学会；较高reward主要有手速度罚下降因素。
- phase3完成：warmstart/retention_manifest.json 34/34 completed，两组各10M/19338networkcalls。warmup前2000calls Actor/temp完全冻结，所有快照det64/64和stoch128/128；恢复更新后update4000（实际Actor999步）已det0/64/stoch0/128，最终也全0。此预热方案未保住技能，不代表所有critic预训练无效。retention_summary.md/.json完整。
- BC导出原版std未训练，det128/128但native随机0/128；此为新诊断初始化混杂，不是原scratch失败新根因。外部噪声乘.1/.01均128/128。std头独立校准后，mean/BN不变；eigen005原生seed0、seed1及每步独立噪声三组各128/128。calibrated_sampling_status.json4项和selected_baseline_status.json2项均完成。std015_eigen127/128、Adam005128/128、Adam015121/128；.05/.15是拟合目标不是恒定实际sigma。
- 新wrapper的114688-transition工程smoke完成：20次冻结后共30次更新、Actor/temp各5步，旧状态因Isaac close直接退出留running，另存posthoc_validation.json；已在独立wrapper修完成标记，CPU通过。真实immediate训练已正确写complete。
- phase4完成26/26：两组各连续50M、20个冻结评测全部goal0；hand.1在20/30/50M det/stoch抬升100%，40M却0，非单调。50M抬升后低于初始高度det55/64、stoch127/128；不能称稳定抓握。hand1的高reward主要是低于10cm阈值的持续height shaping。模型实际alpha已降至约7.9e−6/2.1e−5，常见状态不再严重Q边界裁剪，不能套用BC低熵预热解释。
- 重要未解差异：hand.1训练TB最后30个vectorstep窗口all_goals_hit=.802083，最早positive42.8544M、>=.5首次49.7152M；冻结50M det0/64、stoch0/128。日志源/加权/配置独立检查未发现明显错误，但不等于冻结策略80%成功。budget50m/learning_curve_audit.md、train_eval_mismatch_audit.md、loader_parity.md、critic_followup.md详证。
- BC启动差异2/2完成：随机初始progress+首步随机动作后det124/128、stoch122/128；初始32分别28/26成功，后续96均成功。它是loaded-agent启动方式，不是从零训练约98个随机vectorsteps的完整历史；不能用来解释50M最后窗口。
- 评测上限已核实实际执行argv：det64用1300步、stoch128用2500步，均覆盖600步回合所需波次；此前担心随机评测也用1300的记录不适用于实际执行。汇总仍检查actual requested/completed；如有partial，用prepare_str_retention_completion.py生成新full2500补测目录，不覆盖旧结果，当前无不足额项。
- 23:53关键新增：immediate在5次Actor更新后det64/64、stoch95/128；50次Actor更新后det0/64、stoch0/128。early_update_audit.md用同一512专家状态+参数/BN交叉组合定位早期变化主要来自参数而非仅BN；专家批次初始Q/entropy原始梯度范数比.104，不能冒充实际replay/Adam贡献或证明熵唯一根因。warmup2000正在训练，必须等待对照。
- 分布CPU审计warmstart/policy_distribution_audit.md：专家同状态，校准005 tanh条件熵约−142，但原BC约−180仍动作更抖；低微分熵不等于安静。旧Flash在专家OOD状态约+15.9。不能仅以总熵解释技能丢失。
- 00:10新机制证据：warmup2000成功策略的Q≈−5下界，常见阶段TD未投影目标低于−5质量约98.6%–99.96%；native起点目标期望−6.3269被投影为−4.99946，reward除数281.766、熵贡献约−1.40/step。终止成功样本无bootstrap应另列；不是整个batch/所有checkpoint都这个比例。
- alpha×warmup短对照66/66完成：六组2,000,896transitions/3714calls，所有最终det0/64、stoch0/128；低alpha消除了所查专家样本的严重下界裁剪，但仍丢技能，否定仅降alpha足够。gate0低alpha在5次Actor更新后det已0但曾抬升；gate2000三种alpha在5次后det64/64，约50次后全0（.01实际49次因AMP）。预热有短期保护，不是完全无效。entropy_ab/conclusions.md/final_audit.json已完成。
- 50M协议矩阵6/6完成：32/1024环境×startup、seed1、训练TF32/high精度组合均goal0（1024条件各2048回合）。没有解决晚期在线/冻结差异。budget50m/protocol_check/queue_status.json；新增默认关闭--training-numerics，真实flags写metadata，核心未改。
- 历史reward单因素2/2完成：budget50m/reset_reward_probe，首次reset前历史reward=0 vs真实成功terminal中位数100.14024353（obs≈1.0014），均goal0/128、抬升128/128；参数未变，真实reward计算未改。这个字段确实跨reset保留且影响动作，但仅该初值未解决差异。
- native_entry_check已完成；新非零updatecounter支持及完整optimizer恢复通过11项CPU测试与独立审查，真实GPU冻结9572次后Actor/temp不变。最终原生counter107038，正常分支Actor新增4786步；不是带原replay/物理状态的无缝续训。
- warmup_analysis.md/.json为同一512专家状态的配对证据：501非终止样本的下界clipping83.35%（reset99.47%），与phase-balanced的约99%不同口径。11条真成功终止均mask nextentropy/bootstrap且无clipping，9条非终止lift bonus另列。冻结结束Q/熵原始梯度范数比.00313；低alpha反事实投影减少不等于已经训练解决。
- 汇总入口：scripts/summarize_str_retention.py --root本目录/warmstart；scripts/summarize_str_budget50m.py --root本目录/budget50m；scripts/summarize_str_entropy_ab.py --root本目录/entropy_ab。输出analysis.md/summary.json等，pending不填零。warmstart/retention_actor_steps.png和budget50m/frozen50m_episode0_trace.png已生成。
- phase1/phase2已退出；本任务GPU顺序为retention→budget50m→startup_baseline，勿重复开GPU。只新增默认关闭startup诊断选项使rollout harness源码hash变化；核心训练/config不变，旧hash记录保留，不再声称全部诊断源码毫无变化。
- phase1冻结探索11条件各128回合，均0目标、0持续持握代理；repeat1/4/16曾离桌0/7/20，task>10cm为0/0/2。探索时间连续性确实影响覆盖，但不能把离桌当抓握。全部summary已保存。
- 首动作对照seed0/1各224回合均完成；这是有限回报，没有补timeout尾项。读critic_findings.md最新解读，不把CI不显著当Q绝对正确。
- 心跳自动续查已创建：id=flashsac-str，每小时，到2026-09-09 09:00本地停止新增耗时测试，完成报告后暂停。

## 下一轮接续

1. 检查phase1和phase2状态、日志；failed项保留错误，修脚本后用新输出目录重试，不覆盖旧结果。
2. phase1包括噪声幅度/持续时间、控制、Critic分阶段以及首动作真实有限回报。Q回报实验不能隐瞒timeout未知尾项。
3. 已完成队列勿重复提交。检查repro60m最后seed2与所有50M/60M评测，再检查显式依赖其完成的seed1_sampling_grid6项；其余队列全部完成。不能混淆BC学习、技能留存与SAC从零学习，更不能用在线窗口代替冻结goal验证。
4. 有证据再做新的短A/B。没证据不要直接改LSTM、reward、alpha；尤其先区分学习数据覆盖与Q估计。
5. 主报告追加实际结果、限制、可复现命令和下一步；早上收束后暂停本任务heartbeat，不动其他automation。
6. control_ab/source_hashes.json捕获41个相关源码/配置SHA，CONTINUE.md有只读复核命令。核心代码是本轮开始时的已有版本，未为本轮改奖励/网络/算法；不要覆盖既有dirty变化。

## 恢复入口

先读 `/home/abao/flashsac-robotics/docs/str_overnight_20260908.md`，检查本目录日志、结果与后台服务，再接续未完成部分，避免重复启动。

本任务边界：本机仿真，不碰真机/远端任务/其他本机进程；2026-09-09上午收束汇总。
