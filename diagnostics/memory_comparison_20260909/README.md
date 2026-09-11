# STR 信息与记忆对照：2026-09-09

用户授权：比较140维前馈、162维前馈、140维LSTM。**本机仿真训练，不跑独立eval，不操作真机，不开远程训练。**

## 三组是什么

|组|Actor输入/结构|Critic输入/结构|预算与状态|
|---|---|---|---|
|A：140 FF|当前原始policy140，原FlashSAC前馈|当前privileged162，原FlashSAC双Q|复用此前3个seed，各100,003,840 transitions，已完成|
|B：162 FF|把当前原critic162维提供给原FlashSAC前馈actor|仍是当前162维双Q|新训seed0/1/2，各100,003,840 transitions|
|C：140 LSTM32 + historyQ|过去32帧原policy140；逐帧LayerNorm→LSTM128；当前140+memory128→原FlashSAC主干|当前privileged162＋同一32帧140/validmask展平，共4674维；原FlashSAC双Q主干再拼29维动作|新训seed0/1/2，同预算|

所有组保持：KUKA+Sharpa，原6类/1200生成资产池，随机初态与随机位置/旋转goal链，arm=.1/hand=.1，原reward、600步每goal时限、最多50goal、.075→.01容差课程，DR/noise/delay关闭；2048env、4次Q更新/vectorstep、actor每2次Q更新一次、batch2048、replay10M、gamma=.99、nstep1、相同固定LR日程。无BC、无旧checkpoint初始化、无replay载入。

配置：

- [A](/home/abao/flashsac-robotics/configs/simtoolreal_full_nodr.yaml)
- [B](/home/abao/flashsac-robotics/configs/simtoolreal_full_state162.yaml)
- [C](/home/abao/flashsac-robotics/configs/simtoolreal_full_lstm32.yaml)

配置测试已逐字段核对：除实验输出位置、B的actor观测列表、C的时序参数外，训练配置相同。

## 这个对照能说明什么、不能说明什么

- B对A检验增加当前可用状态信息是否有帮助，不需要首先引入记忆网络。
- C检验一个**能利用有限历史的SAC版本**。32帧约0.53秒，不是官方跨整个episode保持状态的LSTM；失败不能否定更长记忆。
- C的Q也要看到历史。否则同一当前状态、不同历史下的后续policy可能不同，单独Q(s,a)会混合这些情况。这里用Q(s,H,a)，不缓存可能过时的LSTM hidden state。
- C不仅增加Actor的LSTM，也增加Critic的信息与参数，所以**不是纯Actor-only、参数量完全匹配的LSTM因果证明**。B/C各多seed的结果能提供下一轮定位线索，不能把胜负直接说成LSTM唯一原因。
- A复用历史完成运行；默认前馈路径经回归保持兼容，但原native warmup动作空间未显式seed，相同seed编号不是逐步一致的反事实轨迹。GPU仿真也非逐位确定。

方法参考：[Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs](https://arxiv.org/abs/2110.05038)、[作者的时序回放实现说明](https://github.com/twni2016/pomdp-baselines/blob/main/docs/our_details.md)。本项目的有限窗口historyQ是本次实验设计，不宣称照搬该论文或官方STR实现。

## 已经检查了什么

核心接线通过59项CPU回归，另2项汇总测试通过，包括真实SAC优化器更新，不仅是shape mock：

- 同一env的连续历史；reset隔离；中途换goal不误清历史。
- `terminated`与`truncated`都切断下一episode的历史；timeout bootstrap用结束前的final_obs，绝不接新episode。
- replay容量不是env数整倍数时的环绕；覆盖边缘被排除，不能用丢失帧伪装episode开头。
- 训练随机预热期间也记录历史；第一次调用policy时，在线窗口与replay窗口一致。
- LSTM能从过去帧得到梯度、参数实际更新；checkpoint恢复和默认FF旧schema兼容。

[GPU合成训练检查](/home/abao/flashsac-robotics/diagnostics/memory_comparison_20260909/preflight_synthetic_gpu.json)：2048env等效batch、60次真实SAC更新，AMP/编译Critic通过，LSTM权重变化、loss有限。缩小replay仅用于检查，峰值allocated约1.86GB，**不能拿它当完整仿真显存需求或学习结果**。原full仿真训练约30GB；C实际占用以日志/进程为准。

C使用eager cuDNN LSTM，Critic继续compile/AMP；与FF编译执行路径不同，墙钟时间不应强行相同。比较预算是实际transition与更新机会，不是耗时。

## 时序实现与限制

- [Actor](/home/abao/flashsac-robotics/flash_rl/agents/flashSAC/history_network.py)：每次用当前权重从零重建窗口，完整窗口BPTT；左padding通过长度对齐处理，不污染有效LSTM记忆。
- [Replay](/home/abao/flashsac-robotics/flash_rl/buffers/history_buffer.py)：仍存单transition，按需重建历史；只增加write/episode ID元数据，不将10M replay乘32。
- 满buffer后保守排除最老31×2048=63,488条anchor，约容量的0.635%，避免读到被覆盖的history。其余有效anchor均匀采样。
- replay save/load暂不支持，会明确报错；本实验原本就不保存replay。模型checkpoint正常保存。
- 通用eval/play不提供时序边界，C会明确拒绝该入口，不能静默拿单帧当LSTM输入。本次用户不要求eval；以后play需要使用显式历史滚动入口。

## 运行与保存

本次服务：`flashsac-str-memory-compare-20260909.service`。

队列顺序：B0 → C0 → B1 → C1 → B2 → C2 → CPU汇总。一次只启动一个GPU训练子进程，有36GiB空闲显存检查；只管理它自身的子进程，不终止其他用户任务。各训练最多4小时，总截止2026-09-10 15:30（北京时间）。这只是安全上限，不是预计需要24小时。

每20M保存一次，共20/40/60/80/100M五档；目录vectorstep分别9766/19532/29298/39064/48830。完成后不追加其他训练。

- [运行清单与29份源文件哈希](/home/abao/flashsac-robotics/diagnostics/memory_comparison_20260909/manifest.json)
- [即时队列状态](/home/abao/flashsac-robotics/diagnostics/memory_comparison_20260909/queue_status.json)
- 模型：`/home/abao/flashsac-robotics/diagnostics/memory_comparison_20260909/models/{state162_ff,obs140_lstm32_historyq}/seed{0,1,2}/step48830/`
- 日志：`/home/abao/flashsac-robotics/diagnostics/memory_comparison_20260909/logs/`
- [训练曲线对照报告](/home/abao/flashsac-robotics/diagnostics/memory_comparison_20260909/REPORT.md)：当前/手动刷新/最终CPU汇总时的快照，不是后台实时页面；最终服务状态以queue_status.json为准。

最终CPU汇总会写`final_cpu_complete.json`，只表示报告生成；不表示训练学会，更不表示评测成功。缺失数据标PENDING，不算0分。

## 一条命令同时看三组TB

在本机终端复制整行，不需要激活conda：

```bash
/home/abao/play2perfect/.venv_isaacsim/bin/tensorboard --logdir=/home/abao/flashsac-robotics/diagnostics/memory_comparison_20260909/tb --host=127.0.0.1 --port=6009
```

打开 `http://127.0.0.1:6009`。未开跑的seed暂时没有曲线，正常。

主要看 `episode/cumulative/lift_bonus_rew`（除300是曾触发抬升事件比例，非稳定抓握成功率）、`episode/final/successes`（每episode完成goal数）、`episode/cumulative/keypoint_rew`、`task/current_success_tolerance`。按同一transitions横坐标比较，不拿A的100M直接对比B/C刚启动。

手动刷新Markdown对照，不启用GPU：

```bash
/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python /home/abao/flashsac-robotics/scripts/summarize_str_memory_comparison.py --root /home/abao/flashsac-robotics/diagnostics/memory_comparison_20260909
```
